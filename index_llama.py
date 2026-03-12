"""
index_llama.py — Индексация документов через LlamaIndex с 6 стратегиями чанкинга.

Использование:
    python index_llama.py --method fixed
    python index_llama.py --method semantic --reset

Методы: fixed, recursive, layout, semantic, parent_child, sentence_window
Каждый метод сохраняет ноды в отдельную коллекцию Qdrant: rag_<method>
"""

import argparse
import logging
import re
from collections import defaultdict

import qdrant_client
from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    MarkdownElementNodeParser,
    SemanticSplitterNodeParser,
    SentenceSplitter,
    SentenceWindowNodeParser,
    TokenTextSplitter,
    get_leaf_nodes,
)
from llama_index.core.schema import NodeRelationship
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ── Глобальные настройки LlamaIndex ────────────────────────────────────────
Settings.embed_model = OpenAIEmbedding(
    model=config.EMBEDDING_MODEL,
    api_key=config.OPENAI_API_KEY,
)
Settings.llm = OpenAI(model=config.CHAT_MODEL, api_key=config.OPENAI_API_KEY)

METHODS = ["fixed", "recursive", "layout", "semantic", "parent_child", "sentence_window"]

# Служебные метаданные, которые НЕ должны попадать в текст для эмбеддинга.
# По умолчанию LlamaIndex включает все metadata в эмбеддинг-текст,
# что загрязняет вектор шумом вроде "doc_id: matnp\nchunk_index: 42".
EXCLUDED_EMBED_KEYS = [
    "doc_id", "source_file", "chunk_method", "chunk_index", "parent_text",
    "window", "original_text",
]


def inject_table_headers(text: str) -> str:
    """Дублирует заголовок markdown-таблицы перед каждой строкой данных.

    Требование задания: если таблица разбивается на несколько чанков,
    заголовок должен присутствовать в каждом чанке.

    До:                              После:
      | Показатель | 2024 | 2023 |     | Показатель | 2024 | 2023 |
      |------------|------|------|     |------------|------|------|
      | Выручка    | 189  | 162  |     | Выручка    | 189  | 162  |
      | EBITDA     | 47   | 39   |     | Показатель | 2024 | 2023 |
                                       |------------|------|------|
                                       | EBITDA     | 47   | 39   |

    Если сплиттер разрежет между строками — каждый чанк содержит полноценную
    мини-таблицу с заголовком, и LLM читает правильный год.
    """
    lines = text.split("\n")
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Начало таблицы: строка с | и следующая — разделитель (--|--|)
        if (
            line.strip().startswith("|")
            and i + 1 < len(lines)
            and re.match(r"^\s*\|[\s\-:\|]+\|\s*$", lines[i + 1])
        ):
            header = line
            separator = lines[i + 1]
            i += 2
            # Каждую строку данных оборачиваем в мини-таблицу: header + sep + row + blank
            while i < len(lines) and lines[i].strip().startswith("|"):
                result.append(header)
                result.append(separator)
                result.append(lines[i])
                result.append("")
                i += 1
        else:
            result.append(line)
            i += 1
    return "\n".join(result)


def build_parsers(method: str, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list:
    """Возвращает список трансформаций (парсеров) для заданного метода чанкинга.

    chunk_size / chunk_overlap переопределяют config.* (для экспериментов).
    """
    _cs = chunk_size or config.CHUNK_SIZE
    _co = chunk_overlap or config.CHUNK_OVERLAP

    if method == "fixed":
        # Жёсткое деление строго по токенам — "fixed size" в классическом смысле.
        return [TokenTextSplitter(
            chunk_size=_cs,
            chunk_overlap=_co,
        )]
    if method == "recursive":
        # Пробует делить по предложениям → словам → символам (как RecursiveCharacterTextSplitter).
        return [SentenceSplitter(
            chunk_size=_cs,
            chunk_overlap=_co,
        )]
    if method == "layout":
        return [MarkdownElementNodeParser(num_workers=4)]
    if method == "semantic":
        # Предварительный сплит нужен: SemanticSplitter эмбеддит отдельные
        # предложения, и если абзац/таблица > 8192 токенов — OpenAI вернёт 400.
        # SentenceSplitter(4096) разбивает текст на безопасные куски заранее.
        return [
            SentenceSplitter(chunk_size=4096, chunk_overlap=0),
            SemanticSplitterNodeParser(
                embed_model=OpenAIEmbedding(
                    model=config.EMBEDDING_MODEL,
                    api_key=config.OPENAI_API_KEY,
                )
            ),
        ]
    if method == "parent_child":
        return [HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512])]
    if method == "sentence_window":
        # Каждый нод — одно предложение (точный поиск).
        # В метаданных «window» хранится окно из 3 соседних предложений с каждой
        # стороны — именно оно отправляется в LLM для полного контекста.
        return [SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )]
    raise ValueError(f"Неизвестный метод: {method}")


def load_documents(method: str):
    """Загружает исходные документы и добавляет метаданные к каждому файлу."""
    path_to_docid = {doc["path"]: doc["doc_id"] for doc in config.DOCS}

    reader = SimpleDirectoryReader(
        input_files=[doc["path"] for doc in config.DOCS],
        file_metadata=lambda p: {
            "doc_id": path_to_docid.get(p, p),
            "source_file": p,
            "chunk_method": method,
        },
    )
    docs = reader.load_data()
    log.info("Загружено документов: %d.", len(docs))

    # Дублируем заголовок таблицы только для методов с токен-сплиттером.
    # layout/semantic/parent_child/sentence_window обрабатывают таблицы структурно —
    # inject_table_headers разбивает каждую строку в мини-таблицу и ломает их логику.
    if method in ("fixed", "recursive"):
        for doc in docs:
            doc.set_content(inject_table_headers(doc.get_content()))
        log.info("Заголовки таблиц продублированы.")

    return docs


def setup_vector_store(
    client: qdrant_client.QdrantClient,
    collection_name: str,
    reset: bool,
) -> QdrantVectorStore:
    """Удаляет коллекцию при reset=True, затем возвращает QdrantVectorStore."""
    if reset and client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        log.info("Коллекция '%s' удалена.", collection_name)

    return QdrantVectorStore(client=client, collection_name=collection_name)


_RE_ENGLISH_SUMMARY = re.compile(
    r"^The table\b.*?(?=\n\||\Z)",
    re.DOTALL,
)


def _clean_layout_text(text: str) -> str:
    """Убирает артефакты MarkdownElementNodeParser из текста чанка.

    1. Английское описание таблицы («The table ... with the following columns: - ...\n»)
       генерируется LLM внутри парсера и бесполезно для русскоязычного BM25 и LLM.
    2. «Unnamed: N» — pandas-артефакт для безымянных столбцов.
    3. «nan» — pandas-артефакт для пустых ячеек таблицы.
    """
    # 1. Убираем английское описание до первой строки таблицы (или до конца)
    text = _RE_ENGLISH_SUMMARY.sub("", text)
    # 2. Убираем «|Unnamed: N|» ячейки и standalone «Unnamed: N»
    text = re.sub(r"\|?\s*Unnamed:\s*\d+\s*\|?", "|", text)
    # 3. Заменяем «nan» в ячейках таблицы на пустую ячейку
    text = re.sub(r"(?<=\|)\s*nan\s*(?=\|)", "  ", text)
    # 4. Нормализуем лишние символы «|» после замен
    text = re.sub(r"\|{2,}", "|", text)
    return text.strip()


def prepare_nodes(nodes: list, method: str = "") -> list:
    """Добавляет chunk_index, очищает текст layout-нод, исключает служебные метаданные.

    Без excluded_embed_metadata_keys LlamaIndex подмешивает все метаданные
    в текст для вычисления вектора. Для parent_child это катастрофа:
    parent_text (2048 токенов) полностью забивает вектор листа (512 токенов).

    Для layout — дополнительно чистим английские описания и pandas-артефакты.
    Ноды, ставшие пустыми после очистки, отбрасываются.
    """
    counters: dict = defaultdict(int)
    result = []
    for node in nodes:
        if method == "layout":
            cleaned = _clean_layout_text(node.get_content())
            if not cleaned:
                continue  # пустой чанк после очистки — пропускаем
            node.set_content(cleaned)

        doc_id = node.metadata.get("doc_id", "unknown")
        node.metadata["chunk_index"] = counters[doc_id]
        counters[doc_id] += 1
        node.excluded_embed_metadata_keys.extend(EXCLUDED_EMBED_KEYS)
        result.append(node)

    skipped = len(nodes) - len(result)
    if skipped:
        log.info("prepare_nodes: отброшено %d пустых нод после очистки.", skipped)
    return result


def run_pipeline(
    method: str,
    reset: bool,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    collection_name: str | None = None,
) -> None:
    collection_name = collection_name or f"{config.COLLECTION_PREFIX}{method}"
    log.info("Метод='%s'  коллекция='%s'  reset=%s", method, collection_name, reset)

    docs = load_documents(method)
    client = qdrant_client.QdrantClient(path=config.QDRANT_PATH)
    vector_store = setup_vector_store(client, collection_name, reset)

    try:
        # ── Шаг 1: парсинг нод ────────────────────────────────────────────────
        if method == "parent_child":
            parser = build_parsers(method, chunk_size, chunk_overlap)[0]
            all_nodes = parser.get_nodes_from_documents(docs)
            leaf_nodes = get_leaf_nodes(all_nodes)

            # Строим карту: parent_node_id → текст родителя
            leaf_ids = {n.node_id for n in leaf_nodes}
            parent_map = {
                n.node_id: n.get_content()
                for n in all_nodes
                if n.node_id not in leaf_ids
            }

            # Сохраняем текст родителя в метаданные каждого листа
            for node in leaf_nodes:
                parent_rel = node.relationships.get(NodeRelationship.PARENT)
                if parent_rel:
                    node.metadata["parent_text"] = parent_map.get(parent_rel.node_id, "")

            nodes_to_store = leaf_nodes
            log.info(
                "parent_child: всего нод=%d, листовых нод=%d.",
                len(all_nodes),
                len(leaf_nodes),
            )
        else:
            # Цепочка парсеров: каждый получает на вход результат предыдущего
            parsers = build_parsers(method, chunk_size, chunk_overlap)
            current = docs
            for parser in parsers:
                current = parser.get_nodes_from_documents(current)
            nodes_to_store = current

        # ── Шаг 2: chunk_index + исключение метаданных из эмбеддинга ──────────
        nodes_to_store = prepare_nodes(nodes_to_store, method=method)

        # ── Шаг 3: эмбеддинг и сохранение в Qdrant ───────────────────────────
        pipeline = IngestionPipeline(
            transformations=[Settings.embed_model],
            vector_store=vector_store,
        )
        pipeline.run(nodes=nodes_to_store, show_progress=True)

        count = client.count(collection_name).count
        log.info("Готово. Коллекция '%s' содержит %d точек.", collection_name, count)
    finally:
        client.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Индексация документов в Qdrant с выбранной стратегией чанкинга."
    )
    ap.add_argument(
        "--method",
        choices=METHODS,
        required=True,
        help="Стратегия чанкинга.",
    )
    ap.add_argument(
        "--reset",
        action="store_true",
        help="Удалить существующую коллекцию перед индексацией.",
    )
    args = ap.parse_args()
    run_pipeline(method=args.method, reset=args.reset)


if __name__ == "__main__":
    main()
