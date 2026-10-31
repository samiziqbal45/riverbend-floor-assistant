from types import SimpleNamespace as NS

import pytest

import config
from ui import startup


def test_password_gate_logic():
    assert startup.password_ok("anything", None)
    assert startup.password_ok("", "")
    assert startup.password_ok("s3cret", "s3cret")
    assert not startup.password_ok("wrong", "s3cret")
    assert startup.password_ok("pässword", "pässword")  # non-ASCII must not raise


class FakeClient:
    def __init__(self, exists=True, n=0):
        self.exists, self.n = exists, n

    def collection_exists(self, name):
        return self.exists

    def count(self, name, exact=True):
        return NS(count=self.n)


def test_index_size():
    assert startup.index_size(FakeClient(exists=False)) == 0
    assert startup.index_size(FakeClient(n=7)) == 7


def test_ensure_index_skips_when_populated_or_server(monkeypatch):
    monkeypatch.setattr("app.ingest.ingest_all", lambda *a, **k: pytest.fail("must not ingest"))
    assert startup.ensure_index(FakeClient(n=3), None) == 3
    monkeypatch.setattr(config, "QDRANT_URL", "http://qdrant:6333")
    assert startup.ensure_index(FakeClient(n=0), None) == 0


def test_ensure_index_builds_when_missing(monkeypatch):
    monkeypatch.setattr(config, "QDRANT_URL", None)
    c = FakeClient(n=0)
    monkeypatch.setattr("app.ingest.ingest_all", lambda cl, em, reset: setattr(c, "n", 5))
    assert startup.ensure_index(c, None) == 5


def test_preflight_reports_missing_key_and_empty_index(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    errs = startup.preflight_errors(FakeClient(n=0))
    assert len(errs) == 2 and "ANTHROPIC_API_KEY" in errs[0] and "empty" in errs[1]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert startup.preflight_errors(FakeClient(n=4)) == []
