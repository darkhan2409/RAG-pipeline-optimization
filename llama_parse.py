"""
pdf_to_clean_md.py
Шаг 1: Отправляет PDF в LlamaParse и получает Markdown.
Шаг 2: Очищает Markdown для RAG (убирает номера страниц, TOC, GRI-теги, HTML-теги и т.д.).

Использование:
    python pdf_to_clean_md.py input.pdf output_clean.md
    python pdf_to_clean_md.py input.pdf output_clean.md --raw-md raw_output.md
"""

import argparse
import html
import json
import logging
import mimetypes
import os
import re
import sys
import time
from pathlib import Path

import requests

# ── Настройка логгера ─────────────────────────────────────────────────────────
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Конфигурация LlamaParse ───────────────────────────────────────────────────
LLAMA_CLOUD_API_KEY = os.environ["LLAMA_CLOUD_API_KEY"]
BASE_URL = "https://api.cloud.llamaindex.ai/api/parsing"


# ── Шаг 1: Парсинг PDF → Markdown ────────────────────────────────────────────
def parse_pdf_to_markdown(input_pdf: str) -> str:
    """Загружает PDF в LlamaParse и возвращает сырой Markdown."""
    headers = {"Authorization": f"Bearer {LLAMA_CLOUD_API_KEY}"}

    log.info("Загрузка %s в LlamaParse...", input_pdf)
    mime_type = mimetypes.guess_type(input_pdf)[0] or "application/pdf"

    output_options = json.dumps({
        "markdown": {
            "annotate_links": False,
            "tables": {
                "output_tables_as_markdown": True,
                "compact_markdown_tables": True,
            },
        }
    })

    # premium_mode — агент обрабатывает каждую страницу через LLM:
    # лучше распознаёт сложные layouts, логотипы, карты и разорванные таблицы.
    # parsing_instruction направляет агента специфично под годовые отчёты.
    data = {
        "language": "ru",
        "premium_mode": "true",
        "parsing_instruction": (
            "Это годовой отчёт казахстанской компании на русском языке. "
            "Точно сохрани все финансовые таблицы с числами, заголовки разделов "
            "и структуру документа. "
            "Для карт и диаграмм выводи только текстовые подписи, без шкал осей. "
            "Логотипы и водяные знаки игнорируй. "
            "Оглавление (содержание) документа не включай в вывод."
        ),
        "output_options": output_options,
    }

    with open(input_pdf, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/upload",
            headers=headers,
            files={"file": (input_pdf, f, mime_type)},
            data=data,
            timeout=120,
        )
    resp.raise_for_status()
    job_id = resp.json()["id"]
    log.info("Job создан: %s", job_id)

    result_url = f"{BASE_URL}/job/{job_id}/result/markdown"
    log.info("Ожидание результата...")
    while True:
        resp = requests.get(result_url, headers=headers, timeout=60)
        if resp.status_code == 200:
            break
        log.info("Ещё обрабатывается, жду 5 сек...")
        time.sleep(5)

    md = resp.json().get("markdown", "")
    log.info("Получено %d символов от LlamaParse.", len(md))
    return md


# ── Шаг 2: Очистка Markdown для RAG ──────────────────────────────────────────
def clean_markdown_for_rag(content: str) -> str:
    """Убирает шум из Markdown: колонтитулы, номера страниц, TOC, GRI, HTML-теги."""

    # 0a) Декодируем HTML-сущности: &#x26; → &, &amp; → &, &lt; → < и т.д.
    content = html.unescape(content)

    # 0b) Утечки LLM-агента: агент premium mode иногда вставляет в вывод
    #     собственные комментарии или наш системный промпт — удаляем их.
    content = re.sub(r"(?m)^Я вижу[^\n]*\n(?:[^\n]+\n)*?(?=\n|$)", "", content)
    content = re.sub(r"(?m)^Пожалуйста, предоставьте[^\n]*\n?", "", content)
    content = re.sub(r"(?m)^Это годовой отчёт казахстанской компании[^\n]*\n?", "", content)

    # 0c) Колонтитулы страниц вида "# АО «...» 15" + опц. "## ГОДОВОЙ ОТЧЕТ 2024"
    content = re.sub(
        r"(?m)^#+\s+АО\s+«[^»]+»\s+\d+\s*\n(?:##?\s+ГОДОВОЙ ОТЧЕТ \d{4}\s*\n)?",
        "",
        content,
    )

    # 0d) Старый колонтитул matnp без номера страницы
    content = re.sub(r'АО «МАТЕН ПЕТРОЛЕУМ»\n# ГОДОВОЙ ОТЧЕТ 2024\n', "", content)

    # 1) Строки, состоящие только из номера страницы (1–4 цифры)
    content = re.sub(r"(?m)^\s*\d{1,4}\s*$\n?", "", content)

    # 2) GRI-теги: убираем как отдельные строки, так и префикс перед текстом
    #    "GRI", "GRI 2-22" → строка удаляется целиком
    #    "GRI     Обращение председателя" → удаляется только префикс "GRI   "
    content = re.sub(r"(?m)^\s*GRI\s*[\d]?[\d\-–]?\d*\s*$\n?", "", content)  # standalone
    content = re.sub(r"(?m)^GRI\s+", "", content)                              # inline-префикс

    # 3) Блок "Содержание" (TOC) — вырезаем до начала основного раздела
    toc_pattern = re.compile(
        r"(?ms)^\s*Содержание\s*\n+.*?(?=^\s*#\s*О\s+КОМПАНИИ\b|^\s*#\s*О\s+Компании\b|^\s*О\s+КОМПАНИИ\b|^\s*О\s+Компании\b)",
        re.MULTILINE,
    )
    content, n = toc_pattern.subn("", content)

    # fallback: если якорь "О КОМПАНИИ" не найден — ищем "Обращение"
    if n == 0:
        toc_pattern2 = re.compile(
            r"(?ms)^\s*Содержание\s*\n+.*?(?=^\s*#\s*Обращение\b|^\s*Обращение\b)",
            re.MULTILINE,
        )
        content = toc_pattern2.sub("", content)

    # 3b) TOC-блок в виде последовательности заголовков без текста между ними
    #     (10+ заголовков подряд — это содержание, не основной контент)
    content = re.sub(r"(?m)(?:^#{1,3} [^\n]+\n(?:[ \t]*\n)*){10,}", "", content)

    # 4) HTML/XML теги от LlamaParse (например <page_header>АПРЕЛЬ</page_header>)
    content = re.sub(r"<[^>]+>", "", content)

    # 5) Строки-разделители страниц (одиночные "---" на отдельной строке)
    content = re.sub(r"(?m)^---\s*$\n?", "", content)

    # 6) Шкалы осей графиков: строки только из чисел и пробелов (напр. "0   5   10   15")
    content = re.sub(r"(?m)^\s*(?:\d+\s+){3,}\d+\s*$\n?", "", content)

    # 7) OCR-артефакты логотипов: буквы через пробел ("K Z N A", "T E M I R")
    content = re.sub(r"(?m)^(?:[A-ZА-ЯЁ]\s){3,}[A-ZА-ЯЁ\w]*\s*$\n?", "", content)

    # 8) Дублированные предложения внутри абзаца: "Текст. Текст." → "Текст."
    content = re.sub(r"([^.!?\n]{20,}[.!?]) \1", r"\1", content)

    # 9) Нормализация пустых строк (не более двух подряд)
    content = re.sub(r"\n{3,}", "\n\n", content)

    # 10) Хвостовые пробелы на строках
    content = re.sub(r"[ \t]+(?=\n)", "", content)

    return content.strip() + "\n"


# ── Точка входа ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="PDF → чистый Markdown для RAG (LlamaParse + очистка)"
    )
    parser.add_argument("input_pdf", nargs="?", help="Путь к входному PDF-файлу")
    parser.add_argument("output_md", nargs="?", help="Путь к выходному очищенному Markdown-файлу")
    parser.add_argument(
        "--clean-only",
        metavar="MD_FILE",
        help="Применить только очистку к существующему .md файлу (без парсинга PDF)",
    )
    args = parser.parse_args()

    if args.clean_only:
        # Режим: только очистка существующего MD без перепарсинга
        md_path = Path(args.clean_only)
        raw = md_path.read_text(encoding="utf-8")
        clean_md = clean_markdown_for_rag(raw)
        md_path.write_text(clean_md, encoding="utf-8")
        log.info("Очистка применена: %d → %d символов → %s", len(raw), len(clean_md), md_path)
        return

    if not args.input_pdf or not args.output_md:
        parser.error("Укажи input_pdf и output_md, или используй --clean-only <file.md>")

    # Шаг 1: парсинг
    raw_markdown = parse_pdf_to_markdown(args.input_pdf)

    # Шаг 2: очистка
    log.info("Очистка Markdown...")
    clean_md = clean_markdown_for_rag(raw_markdown)

    # Сохраняем результат
    Path(args.output_md).write_text(clean_md, encoding="utf-8")
    log.info(
        "Готово! %d → %d символов → %s",
        len(raw_markdown),
        len(clean_md),
        args.output_md,
    )


if __name__ == "__main__":
    main()
