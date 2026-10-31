"""Lazy singleton clients. Nothing is created at import time."""
from __future__ import annotations

from functools import lru_cache

import config


@lru_cache(maxsize=1)
def get_qdrant():
    from qdrant_client import QdrantClient

    if config.QDRANT_URL:
        return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    return QdrantClient(path=config.QDRANT_PATH)


@lru_cache(maxsize=1)
def get_embedder():
    from fastembed import TextEmbedding

    return TextEmbedding(config.EMBED_MODEL, cache_dir=config.FASTEMBED_CACHE)


@lru_cache(maxsize=1)
def get_llm():
    import anthropic
    from dotenv import load_dotenv

    load_dotenv(config.ROOT / ".env")
    return anthropic.Anthropic()
