"""Startup concerns for the UI: password gate, index bootstrap, fail-fast checks.

Kept free of Streamlit imports so it can be unit-tested.
"""
from __future__ import annotations

import hmac
import os

import config


def password_ok(entered: str, expected: str | None) -> bool:
    """True when the gate is off (no expected password) or the entry matches."""
    if not expected:
        return True
    return hmac.compare_digest(entered.encode("utf-8"), expected.encode("utf-8"))


def index_size(client) -> int:
    if not client.collection_exists(config.COLLECTION):
        return 0
    return client.count(config.COLLECTION, exact=True).count


def ensure_index(client, embedder) -> int:
    """Build the index in-process if it is missing (Streamlit Community Cloud path).

    Skipped when a Qdrant server is configured; there the ingest is an explicit step.
    Returns the number of points.
    """
    n = index_size(client)
    if n or config.QDRANT_URL:
        return n
    from app.ingest import ingest_all

    ingest_all(client, embedder, reset=True)
    return index_size(client)


def preflight_errors(client) -> list[str]:
    """Clear startup problems; an empty list means the app can serve queries."""
    from dotenv import load_dotenv

    load_dotenv(config.ROOT / ".env")
    errors = []
    if not os.getenv("ANTHROPIC_API_KEY"):
        errors.append("ANTHROPIC_API_KEY is not set. Add it to .env or the host's secrets.")
    if index_size(client) == 0:
        errors.append(f"Qdrant collection '{config.COLLECTION}' is empty. Run `python -m app.ingest --reset`.")
    return errors
