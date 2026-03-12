"""
ask_llama.py — Q&A через RAG pipeline с выбором стратегии чанкинга.

Использование:
    python ask_llama.py --method fixed "Какова выручка КТЖ?"
    python ask_llama.py --method semantic --top-k 10 --show-chunks
    python ask_llama.py --method parent_child   # тестовые вопросы

Методы: fixed, recursive, layout, semantic, parent_child, sentence_window
Каждый метод читает свою коллекцию Qdrant: rag_<method>

Пайплайны:
  fixed          → Naive RAG:    embed → vector search → LLM
  остальные      → Advanced RAG: rewrite → hybrid (vector+BM25) → RRF → rerank → LLM
"""

import argparse
import json
import re
import sys

from openai import OpenAI
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

import config

# ── Клиенты ───────────────────────────────────────────────────────────────────
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass  # Colab OutStream не поддерживает reconfigure
openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
qdrant: QdrantClient | None = None


def _get_qdrant() -> QdrantClient:
    """Ленивая инициализация: клиент создаётся при первом вызове, не при импорте."""
    global qdrant
    if qdrant is None:
        qdrant = QdrantClient(path=config.QDRANT_PATH)
    return qdrant

METHODS = ["fixed", "recursive", "layout", "semantic", "parent_child", "sentence_window"]

# ── Системный промпт ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Ты — аналитик финансовых и корпоративных отчётов казахстанских компаний.

ПРАВИЛА:
1. Отвечай СТРОГО на основании предоставленного КОНТЕКСТА. Не используй никаких знаний извне.
2. Если ответа нет в контексте — напиши: «Информация в предоставленных документах отсутствует.»
3. НЕ ПРИДУМЫВАЙ факты. Никакой галлюцинации.
4. Отвечай на русском языке, кратко и по делу.
5. НЕ добавляй ссылки на источники, чанки или документы в ответ.

ДОПОЛНИТЕЛЬНЫЕ ПРАВИЛА ДЛЯ ТАБЛИЦ И ВЫЧИСЛЕНИЙ:
6. Таблицы с несколькими годами: чётко указывай, из какого столбца (какого года) берёшь число.
7. Не производи самостоятельных вычислений (%, разницы, темпы роста), если результат не указан явно в контексте.
   Если нужный % отсутствует явно, приведи исходные числа из контекста и напиши:
   «Точный % роста в документах не указан; по данным контекста: X за 2024 г. и Y за 2023 г.»
8. Если вопрос касается нескольких компаний, а данные есть только по одной — ответь по той, что есть,
   и явно укажи: «По другой компании данных в контексте нет.»
"""

CONTEXT_TEMPLATE = """КОНТЕКСТ ИЗ ДОКУМЕНТОВ:

{context_blocks}

ВОПРОС: {query}"""


# ══════════════════════════════════════════════════════════════════════════════
# ── Утилиты ───────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def embed_query(query: str) -> list[float]:
    resp = openai_client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=query,
        encoding_format="float",
    )
    return resp.data[0].embedding


def parse_payload(pl: dict, method: str) -> dict:
    """
    Извлекает чистые данные из payload Qdrant-точки.

    LlamaIndex хранит всё в JSON-строке _node_content.
    Для parent_child → LLM получает parent_text (2048 токенов).
    Для sentence_window → LLM получает window (7 предложений).
    Реранкинг ВСЕГДА по leaf_text (найденный чанк).
    """
    node = {}
    if "_node_content" in pl:
        try:
            node = json.loads(pl["_node_content"])
        except (json.JSONDecodeError, KeyError):
            pass

    meta        = node.get("metadata", {})
    leaf_text   = node.get("text", pl.get("text", ""))
    doc_id      = meta.get("doc_id", pl.get("doc_id", "unknown"))
    chunk_index = meta.get("chunk_index", pl.get("chunk_index", "?"))

    # LLM-текст: расширенный контекст при необходимости
    if method == "parent_child":
        llm_text = meta.get("parent_text") or leaf_text
    elif method == "sentence_window":
        llm_text = meta.get("window") or leaf_text
    else:
        llm_text = leaf_text

    return {
        "doc_id":      doc_id,
        "chunk_index": chunk_index,
        "leaf_text":   leaf_text,   # для реранкинга и --show-chunks
        "llm_text":    llm_text,    # для промпта LLM
    }


# ══════════════════════════════════════════════════════════════════════════════
# ── Naive RAG (только для fixed) ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def retrieve(query_vector: list[float], method: str, top_k: int) -> list[dict]:
    """Простой векторный поиск — используется только для метода fixed."""
    collection_name = f"{config.COLLECTION_PREFIX}{method}"
    results = _get_qdrant().query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        score_threshold=config.SCORE_THRESHOLD,
        with_payload=True,
    ).points
    hits = []
    for hit in results:
        data = parse_payload(hit.payload, method)
        data["score"] = round(hit.score, 4)
        hits.append(data)
    return hits


# ══════════════════════════════════════════════════════════════════════════════
# ── Advanced RAG ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# ── Lazy-load реранкера (568 MB, один раз) ────────────────────────────────────
_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        print(f"[Reranker] Загрузка модели {config.RERANK_MODEL} …")
        _reranker = CrossEncoder(config.RERANK_MODEL)
        print("[Reranker] Готово.")
    return _reranker


# ── Pre-Retrieval: Query Rewriting ────────────────────────────────────────────

def rewrite_query(user_query: str) -> list[str]:
    """
    Генерирует REWRITE_COUNT альтернативных запросов через GPT-4o-mini.
    Возвращает [original, alt1, alt2, alt3].
    """
    system = (
        "Ты — эксперт по финансовым отчётам казахстанских компаний. "
        "Перепиши вопрос пользователя альтернативными способами, "
        "используя финансовую терминологию (МСФО, выручка, EBITDA, чистая прибыль и т.д.). "
        f"Верни ровно {config.REWRITE_COUNT} вариант(а/ов), каждый на новой строке, без нумерации."
    )
    resp = openai_client.chat.completions.create(
        model=config.REWRITE_MODEL,
        temperature=0,
        messages=[
            {"role": "system",  "content": system},
            {"role": "user",    "content": user_query},
        ],
    )
    raw = resp.choices[0].message.content or ""
    alts = [line.strip() for line in raw.splitlines() if line.strip()][:config.REWRITE_COUNT]
    queries = [user_query] + alts
    print(f"[Rewrite] Запросы ({len(queries)}):")
    for i, q in enumerate(queries):
        print(f"  [{i}] {q}")
    return queries


# ── BM25 in-memory ────────────────────────────────────────────────────────────

def _tokenize_ru(text: str) -> list[str]:
    """Простой токенизатор: кириллица + латиница + цифры, lowercase."""
    return re.findall(r"[а-яёА-ЯЁa-zA-Z0-9]+", text.lower())


# Кеш: {method: {"ids": [...], "texts": [...], "bm25": BM25Okapi}}
_bm25_cache: dict[str, dict] = {}


def load_bm25_corpus(method: str, collection_name: str | None = None) -> dict:
    """
    Загружает все точки коллекции через scroll (пагинация, без векторов).
    Строит и кеширует BM25Okapi-индекс.
    """
    collection_name = collection_name or f"{config.COLLECTION_PREFIX}{method}"
    if collection_name in _bm25_cache:
        return _bm25_cache[collection_name]
    ids: list    = []
    texts: list[str] = []
    payloads: list[dict] = []

    # Scroll с пагинацией — обходим все точки коллекции
    offset = None
    while True:
        result = _get_qdrant().scroll(
            collection_name=collection_name,
            limit=config.BM25_SCROLL_LIMIT,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        batch, next_offset = result
        if not batch:
            break
        for point in batch:
            parsed = parse_payload(point.payload, method)
            ids.append(point.id)
            texts.append(parsed["leaf_text"])
            payloads.append(parsed)
        if next_offset is None:
            break
        offset = next_offset

    tokenized = [_tokenize_ru(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    print(f"[BM25] Корпус '{collection_name}': {len(ids)} точек загружено.")

    _bm25_cache[collection_name] = {"ids": ids, "texts": texts, "payloads": payloads, "bm25": bm25}
    return _bm25_cache[collection_name]


def search_bm25(query: str, method: str, top_n: int, collection_name: str | None = None) -> list[dict]:
    """BM25 поиск. Возвращает список dict с point_id и payload."""
    corpus = load_bm25_corpus(method, collection_name=collection_name)
    tokens = _tokenize_ru(query)
    scores = corpus["bm25"].get_scores(tokens)

    # Топ-N по убыванию score
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    results = []
    for rank, idx in enumerate(ranked_indices):
        if scores[idx] <= 0:
            break
        entry = dict(corpus["payloads"][idx])
        entry["point_id"] = corpus["ids"][idx]
        entry["bm25_score"] = round(float(scores[idx]), 4)
        results.append(entry)
    return results


def search_vector(query_vector: list[float], method: str, top_n: int, collection_name: str | None = None) -> list[dict]:
    """Векторный поиск для Advanced Pipeline. Возвращает список dict с point_id."""
    collection_name = collection_name or f"{config.COLLECTION_PREFIX}{method}"
    results = _get_qdrant().query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_n,
        with_payload=True,
    ).points
    hits = []
    for hit in results:
        data = parse_payload(hit.payload, method)
        data["point_id"]    = hit.id
        data["vector_score"] = round(hit.score, 4)
        hits.append(data)
    return hits


# ── RRF ───────────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
    top_n: int = 10,
) -> list[dict]:
    """
    Объединяет несколько ранжированных списков через Reciprocal Rank Fusion.
    Score(doc) = Σ 1/(k + rank_i).
    Дедупликация по point_id. Возвращает top_n уникальных кандидатов.
    """
    scores: dict[str, float] = {}
    best: dict[str, dict]    = {}  # best payload по point_id

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            pid = str(item.get("point_id", f"{item['doc_id']}_{item['chunk_index']}"))
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
            if pid not in best:
                best[pid] = item

    sorted_pids = sorted(scores, key=lambda p: scores[p], reverse=True)[:top_n]
    result = []
    for pid in sorted_pids:
        entry = dict(best[pid])
        entry["rrf_score"] = round(scores[pid], 6)
        result.append(entry)

    print(f"[RRF] {sum(len(l) for l in ranked_lists)} записей → {len(result)} уникальных кандидатов.")
    return result


def weighted_reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    weights: list[float],
    k: int = 60,
    top_n: int = 10,
) -> list[dict]:
    """
    Взвешенный RRF: score(doc) = Σ weight_i / (k + rank_i + 1).
    weights[i] соответствует ranked_lists[i].
    Используется для управления балансом vector/BM25 через alpha.
    """
    scores: dict[str, float] = {}
    best: dict[str, dict] = {}

    for ranked, weight in zip(ranked_lists, weights):
        for rank, item in enumerate(ranked):
            pid = str(item.get("point_id", f"{item['doc_id']}_{item['chunk_index']}"))
            scores[pid] = scores.get(pid, 0.0) + weight / (k + rank + 1)
            if pid not in best:
                best[pid] = item

    sorted_pids = sorted(scores, key=lambda p: scores[p], reverse=True)[:top_n]
    result = []
    for pid in sorted_pids:
        entry = dict(best[pid])
        entry["rrf_score"] = round(scores[pid], 6)
        result.append(entry)

    print(f"[Weighted RRF] {sum(len(l) for l in ranked_lists)} записей → {len(result)} уникальных кандидатов.")
    return result


# ── Кросс-документное покрытие ───────────────────────────────────────────────

# Ключевые слова запроса → doc_id коллекции
_DOC_KEYWORDS: dict[str, list[str]] = {
    "ktj":   ["ктж", "қтж", "ktj", "қазақстан темір жолы", "казакстан темир жолы"],
    "matnp": ["матен", "maten", "matnp"],
}
_MIN_PER_DOC = 3  # минимальное число чанков от каждого упомянутого документа


def _detect_doc_ids(query: str) -> list[str]:
    """Возвращает doc_id документов, упомянутых в запросе."""
    q = query.lower()
    return [doc_id for doc_id, kws in _DOC_KEYWORDS.items() if any(kw in q for kw in kws)]


def _ensure_doc_coverage(
    candidates: list[dict],
    query: str,
    method: str,
    mentioned_docs: list[str],
    collection_name: str | None = None,
) -> list[dict]:
    """
    Если документ упомянут в запросе, но слабо представлен в кандидатах —
    добирает недостающие чанки через BM25 по уже загруженному корпусу
    (без дополнительных API-вызовов).
    """
    if len(mentioned_docs) < 2:
        return candidates  # не кросс-документный запрос

    coverage: dict[str, int] = {}
    for c in candidates:
        coverage[c["doc_id"]] = coverage.get(c["doc_id"], 0) + 1

    existing_pids = {str(c.get("point_id", "")) for c in candidates}
    extra: list[dict] = []

    for doc_id in mentioned_docs:
        have = coverage.get(doc_id, 0)
        if have >= _MIN_PER_DOC:
            continue

        needed = _MIN_PER_DOC - have
        corpus = load_bm25_corpus(method, collection_name=collection_name)
        tokens = _tokenize_ru(query)
        scores = corpus["bm25"].get_scores(tokens)

        ranked = sorted(
            [
                (i, scores[i])
                for i, p in enumerate(corpus["payloads"])
                if p["doc_id"] == doc_id and str(corpus["ids"][i]) not in existing_pids
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        added = 0
        for idx, _ in ranked[:needed]:
            entry = dict(corpus["payloads"][idx])
            entry["point_id"] = corpus["ids"][idx]
            entry["rrf_score"] = 0.0
            extra.append(entry)
            existing_pids.add(str(corpus["ids"][idx]))
            added += 1

        print(f"[Coverage] '{doc_id}': было {have} → добавлено {added} чанков.")

    return candidates + extra


# ── Reranking ─────────────────────────────────────────────────────────────────

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int,
    threshold: float,
) -> list[dict]:
    """
    CrossEncoder реранкинг по оригинальному запросу и leaf_text чанка.
    Устанавливает score = rerank_score для совместимости с build_context / --show-chunks.
    """
    if not candidates:
        return []

    reranker = get_reranker()
    pairs = [(query, c["leaf_text"]) for c in candidates]
    raw_scores = reranker.predict(pairs)

    scored = list(zip(raw_scores, candidates))
    scored.sort(key=lambda x: x[0], reverse=True)

    print(f"[Reranker] Результаты (порог {threshold}):")
    results = []
    for new_pos, (score, cand) in enumerate(scored):
        old_pos = candidates.index(cand)
        direction = "↑UP" if new_pos < old_pos else ("↓DOWN" if new_pos > old_pos else "─")
        label = f"[{cand['doc_id']}, чанк {cand['chunk_index']}]"
        print(
            f"  {label} pos {old_pos + 1} → {new_pos + 1} {direction} "
            f"(rerank_score: {score:.4f})"
        )
        if score < threshold:
            print(f"    ↳ отброшен (ниже порога {threshold})")
            continue
        entry = dict(cand)
        entry["score"] = round(float(score), 4)   # единое поле score для совместимости
        results.append(entry)
        if len(results) >= top_k:
            break

    print(f"[Reranker] Финал: {len(results)} чанков передано в LLM.")
    return results


# ── Orchestrator ──────────────────────────────────────────────────────────────

def advanced_retrieve(query: str, method: str, top_k: int) -> list[dict]:
    """
    Полный Advanced RAG цикл:
      rewrite_query → 4 запроса
        для каждого: embed → search_vector + search_bm25
      → 8 ranked_lists → RRF → top-10
      → rerank(оригинальный запрос) → top-k
    """
    queries = rewrite_query(query)

    ranked_lists: list[list[dict]] = []
    for q in queries:
        vec = embed_query(q)
        ranked_lists.append(search_vector(vec, method, config.RRF_TOP_N))
        ranked_lists.append(search_bm25(q, method, config.RRF_TOP_N))

    candidates = reciprocal_rank_fusion(ranked_lists, k=config.RRF_K, top_n=config.RRF_TOP_N)

    # Кросс-документное покрытие: добираем чанки из недопредставленных документов
    mentioned_docs = _detect_doc_ids(query)
    candidates = _ensure_doc_coverage(candidates, query, method, mentioned_docs, collection_name=None)

    hits = rerank(query, candidates, top_k=top_k, threshold=config.RERANK_THRESHOLD)
    return hits


def advanced_retrieve_configurable(
    query: str,
    method: str,
    top_k: int,
    rrf_alpha: float = 0.5,
    use_reranking: bool = True,
    collection_name: str | None = None,
    rerank_threshold: float = config.RERANK_THRESHOLD,
) -> list[dict]:
    """
    Конфигурируемый Advanced RAG пайплайн.

    Параметры:
      rrf_alpha:      баланс vector/BM25 (1.0 = только vector, 0.0 = только BM25)
      use_reranking:  вкл/выкл CrossEncoder реранкинг
      collection_name: override коллекции (для экспериментов с chunk_size)
    """
    queries = rewrite_query(query)

    ranked_lists: list[list[dict]] = []
    weights: list[float] = []
    for q in queries:
        vec = embed_query(q)
        ranked_lists.append(search_vector(vec, method, config.RRF_TOP_N, collection_name=collection_name))
        weights.append(rrf_alpha)
        ranked_lists.append(search_bm25(q, method, config.RRF_TOP_N, collection_name=collection_name))
        weights.append(1.0 - rrf_alpha)

    candidates = weighted_reciprocal_rank_fusion(
        ranked_lists, weights, k=config.RRF_K, top_n=config.RRF_TOP_N,
    )

    # Кросс-документное покрытие
    mentioned_docs = _detect_doc_ids(query)
    candidates = _ensure_doc_coverage(candidates, query, method, mentioned_docs, collection_name=collection_name)

    if use_reranking:
        hits = rerank(query, candidates, top_k=top_k, threshold=rerank_threshold)
    else:
        # Без реранкинга — берём top_k по RRF-score
        hits = candidates[:top_k]
        for h in hits:
            h["score"] = h.get("rrf_score", 0.0)

    return hits


def ask_with_contexts(
    query: str,
    method: str,
    top_k: int,
    use_reranking: bool = True,
    rrf_alpha: float = 0.5,
    collection_name: str | None = None,
    rerank_threshold: float = config.RERANK_THRESHOLD,
) -> tuple[str, list[str]]:
    """
    Возвращает (answer, contexts) — формат, нужный для RAGAS-оценки.
    contexts = список llm_text из хитов (то, что видит LLM).
    """
    if method == "fixed":
        query_vec = embed_query(query)
        hits = retrieve(query_vec, method, top_k)
    else:
        hits = advanced_retrieve_configurable(
            query, method, top_k,
            rrf_alpha=rrf_alpha,
            use_reranking=use_reranking,
            collection_name=collection_name,
            rerank_threshold=rerank_threshold,
        )

    answer = generate_answer(hits, query)
    contexts = [h["llm_text"] for h in hits]
    return answer, contexts


# ══════════════════════════════════════════════════════════════════════════════
# ── Общие функции вывода ──────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def build_context(hits: list[dict]) -> str:
    blocks = []
    for h in hits:
        label = f"[{h['doc_id']}, чанк {h['chunk_index']}] (score={h['score']})"
        blocks.append(f"--- {label} ---\n{h['llm_text']}")
    return "\n\n".join(blocks)


def generate_answer(hits: list[dict], query: str) -> str:
    """Формирует контекст из хитов и генерирует ответ через LLM."""
    if not hits:
        return "Релевантные чанки не найдены. Попробуйте переформулировать вопрос."

    context = build_context(hits)
    user_msg = CONTEXT_TEMPLATE.format(context_blocks=context, query=query)

    response = openai_client.chat.completions.create(
        model=config.CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
    )
    return response.choices[0].message.content


def answer_query(query: str, method: str, top_k: int, show_chunks: bool) -> None:
    """Роутинг: fixed → Naive RAG, остальные → Advanced RAG."""
    if method == "fixed":
        # Naive RAG — без изменений
        query_vec = embed_query(query)
        hits = retrieve(query_vec, method, top_k)
    else:
        # Advanced RAG
        hits = advanced_retrieve(query, method, top_k)

    if show_chunks:
        print("\n── Найденные чанки ──────────────────────────────────")
        for h in hits:
            print(f"[{h['doc_id']}, чанк {h['chunk_index']}] score={h['score']}")
            print(h["leaf_text"][:300], "...\n")
        print("─────────────────────────────────────────────────────\n")

    answer = generate_answer(hits, query)
    print("\n── Ответ ────────────────────────────────────────────")
    print(answer)
    print("─────────────────────────────────────────────────────")


# ── ask() — упрощённый вызов для внешних скриптов ────────────────────────────
def ask(query: str, method: str, top_k: int) -> str:
    """Полный цикл: retrieval → LLM ответ (без вывода в консоль)."""
    if method == "fixed":
        query_vec = embed_query(query)
        hits = retrieve(query_vec, method, top_k)
    else:
        hits = advanced_retrieve(query, method, top_k)
    return generate_answer(hits, query)


# ══════════════════════════════════════════════════════════════════════════════
# ── Тестовые вопросы ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

TEST_QUESTIONS = [
    # Точный факт в прозе; все методы находят, sentence_window точнее
    (
        "Какой кредитный рейтинг присвоило Moody's КТЖ в 2024 году?",
        "факт-рейтинг",
    ),
    # Одно значение из таблицы; layout держит таблицу целиком
    (
        "Какова выручка КТЖ от грузовых перевозок за 2024 год?",
        "таблица-kpi",
    ),
    # Тренд за несколько лет; layout/parent_child > fixed
    (
        "Как менялась маржа EBITDA Матен Петролеум с 2022 по 2024 год?",
        "таблица-тренд",
    ),
    # Числа из нескольких абзацев; parent_child/layout > fixed
    (
        "Каков фактический объём добычи нефти на месторождении Матин в 2024 году?",
        "числа-план-факт",
    ),
    # Длинный список из прозы; semantic/parent_child > sentence_window
    (
        "Перечисли стратегические цели КТЖ в рамках Стратегии развития до 2032 года.",
        "стратегия-список",
    ),
    # Причинно-следственное; semantic/parent_child > fixed
    (
        "По какой причине выручка Матен Петролеум снизилась в 2024 году?",
        "причина-следствие",
    ),
    # Кросс-документ; тестирует coverage + оба документа
    (
        "У какой компании — КТЖ или Матен Петролеум — выше маржа EBITDA за 2024 год?",
        "кросс-EBITDA",
    ),
]


def run_test_questions(method: str, top_k: int, show_chunks: bool) -> None:
    print("═" * 60)
    print(f"  Тестовые вопросы — метод: {method}")
    print("═" * 60)
    for i, (query, tag) in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[{i}/{len(TEST_QUESTIONS)}] [{tag}] {query}")
        print("─" * 60)
        answer_query(query, method, top_k, show_chunks)
    print("\n" + "═" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# ── CLI ───────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG Q&A по годовым отчётам с выбором стратегии чанкинга"
    )
    parser.add_argument(
        "--method",
        choices=METHODS,
        required=True,
        help="Стратегия чанкинга (коллекция rag_<method>)",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Вопрос на русском языке (если не указан — запускаются тестовые вопросы)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=config.TOP_K,
        help=f"Кол-во чанков для retrieval (по умолчанию {config.TOP_K})",
    )
    parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="Показывать найденные чанки перед ответом",
    )
    args = parser.parse_args()

    if args.query:
        answer_query(args.query, args.method, args.top_k, args.show_chunks)
    else:
        run_test_questions(args.method, args.top_k, args.show_chunks)


if __name__ == "__main__":
    try:
        main()
    finally:
        if qdrant is not None:
            qdrant.close()
