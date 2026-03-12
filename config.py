"""
config.py — Центральная конфигурация RAG pipeline.
Все параметры берутся из переменных окружения или дефолтов.
"""

import os

# ── OpenAI ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
EMBEDDING_MODEL: str = "text-embedding-3-large"
EMBEDDING_DIMS: int = 3072
CHAT_MODEL: str = "gpt-4o-mini"

# ── Qdrant ────────────────────────────────────────────────────────────────────
QDRANT_PATH: str = "./qdrant_data"   # локальное файловое хранилище
COLLECTION_NAME: str = "rag_docs"
COLLECTION_PREFIX: str = "rag_"   # prefix for LlamaIndex collections

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 1024       # макс. токенов в чанке
CHUNK_OVERLAP: int = 200     # перекрытие между чанками
CHUNK_MIN: int = 50          # минимум токенов; короче — отбрасываем
TIKTOKEN_ENCODING: str = "cl100k_base"

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K: int = 5
SCORE_THRESHOLD: float = 0.20   # минимальный cosine score

# ── Advanced RAG ──────────────────────────────────────────────────────────────
REWRITE_MODEL: str   = "gpt-4o-mini"
REWRITE_COUNT: int   = 1           # кол-во альтернативных запросов
RRF_K: int           = 60          # константа сглаживания RRF
RRF_TOP_N: int       = 20          # кандидатов после RRF, до реранкинга
RERANK_MODEL: str    = "BAAI/bge-reranker-v2-m3"
RERANK_THRESHOLD: float = 0.01     # чанки ниже порога отбрасываются
BM25_SCROLL_LIMIT: int = 2000      # макс. точек при загрузке BM25-корпуса

# ── Источники ────────────────────────────────────────────────────────────────
DOCS = [
    {"path": "matnp_clean.md", "doc_id": "matnp"},
    {"path": "ktj_clean.md",   "doc_id": "ktj"},
]
