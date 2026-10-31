"""All tunables for the Plant Floor Doc Assistant.

Every setting reads from environment variables first, so the same code runs
locally, in Docker, and in the cloud with no code changes.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _opt_float(name: str) -> float | None:
    v = os.getenv(name)
    return float(v) if v else None


DOMAINS = ["safety", "maintenance", "quality"]

COLLECTION = os.getenv("COLLECTION", "plant_docs")
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_data")
# If QDRANT_URL is set, use a Qdrant server instead of local mode.
QDRANT_URL: str | None = os.getenv("QDRANT_URL") or None
QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY") or None

# So the embedding model can be baked into a Docker image.
FASTEMBED_CACHE = os.getenv("FASTEMBED_CACHE", "./models")

LOG_DIR = os.getenv("LOG_DIR", "./logs")
APP_PASSWORD: str | None = os.getenv("APP_PASSWORD") or None  # gate off when unset

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = 384

LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")

K_PER_DOMAIN = 5
FINAL_K = 6
MIN_PER_DOMAIN = 2

# If set (e.g. 0.85), search only the top domain when its score >= threshold.
SINGLE_DOMAIN_THRESHOLD: float | None = _opt_float("SINGLE_DOMAIN_THRESHOLD")

MAX_CHUNK_TOKENS = 400  # approximated as words * 1.3

# Client-configured; empty = no automatic conflict resolution.
PRECEDENCE_RULES: list[dict] = []

DOMAIN_OWNERS = {
    "safety": "EHS Lead",
    "maintenance": "Maintenance Supervisor",
    "quality": "Quality Engineer",
}

CORPUS_DIR = ROOT / "corpus"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"
EVAL_DIR = ROOT / "eval"

# Router settings (app/router.py).
FEWSHOT_PATH = Path(os.getenv("FEWSHOT_PATH", str(EVAL_DIR / "fewshot.jsonl")))
ROUTER_MAX_TOKENS = int(os.getenv("ROUTER_MAX_TOKENS", "300"))
LLM_RETRIES = int(os.getenv("LLM_RETRIES", "1"))  # extra attempts after the first failure
