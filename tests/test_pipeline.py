"""Precedence + orchestration tests (all components faked)."""
from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

import config
from app import pipeline
from app.generate import Conflict, GenResult
from app.precedence import apply_precedence
from app.retrieve import Chunk


def chunk(domain, doc="Doc", rev="A", date="2025-01-01", owner="", src=None):
    return Chunk(f"{domain}-{doc}-{rev}", 0.8, "t", domain, doc, "Sec", 1, 1, rev, date, owner,
                 source_path=src or f"{domain}/{doc}.pdf")


def conflict(a=1, b=2):
    return Conflict("torque", "18-22", a, "15-20", b, "contradiction")


# ---------- precedence ----------

def test_default_no_rules_escalates_to_both_owners():
    chunks = [chunk("quality", owner="Quality Engineer"), chunk("maintenance", owner="Maintenance Supervisor")]
    [c] = apply_precedence([conflict()], chunks, rules=[])
    assert c.resolution is None and c.rule is None
    assert c.escalate_to == ["Quality Engineer", "Maintenance Supervisor"]


def test_default_uses_config_rules_which_are_empty():
    assert config.PRECEDENCE_RULES == []
    [c] = apply_precedence([conflict()], [chunk("quality"), chunk("maintenance")])
    assert c.resolution is None and c.escalate_to == ["Quality Engineer", "Maintenance Supervisor"]


def test_owner_falls_back_to_domain_owner_and_dedupes():
    [c] = apply_precedence([conflict()], [chunk("safety"), chunk("safety", doc="Other")], rules=[])
    assert c.escalate_to == ["EHS Lead"]


def test_latest_revision_same_doc():
    old, new = chunk("quality", "Spec", "D", "2024-01-01", src="q/spec.pdf"), chunk("quality", "Spec", "E", "2025-07-01", src="q/spec.pdf")
    [c] = apply_precedence([conflict()], [old, new], rules=[{"type": "latest_revision"}])
    assert c.rule == "latest_revision" and c.winner_cite == 2 and "rev E" in c.resolution and c.escalate_to == []


def test_latest_revision_does_not_apply_across_docs():
    [c] = apply_precedence([conflict()], [chunk("quality", "A"), chunk("maintenance", "B", date="2020-01-01")],
                           rules=[{"type": "latest_revision"}])
    assert c.resolution is None and c.escalate_to


def test_domain_order():
    chunks = [chunk("maintenance", "Guide"), chunk("safety", "Jam")]
    [c] = apply_precedence([conflict()], chunks, rules=[{"type": "domain_order", "order": ["safety", "quality", "maintenance"]}])
    assert c.rule == "domain_order" and c.winner_cite == 2 and "safety takes precedence" in c.resolution


def test_domain_order_needs_both_domains_listed_and_different():
    rule = {"type": "domain_order", "order": ["safety"]}
    assert apply_precedence([conflict()], [chunk("safety"), chunk("quality")], rules=[rule])[0].resolution is None
    rule = {"type": "domain_order", "order": ["safety", "quality"]}
    assert apply_precedence([conflict()], [chunk("safety", "A"), chunk("safety", "B")], rules=[rule])[0].resolution is None


def test_first_matching_rule_wins_and_input_not_mutated():
    chunks = [chunk("maintenance"), chunk("safety")]
    original = conflict()
    rules = [{"type": "latest_revision"}, {"type": "domain_order", "order": ["safety", "maintenance"]}]
    [c] = apply_precedence([original], chunks, rules=rules)
    assert c.rule == "domain_order" and original.resolution is None


def test_unknown_rule_type_raises():
    with pytest.raises(ValueError):
        apply_precedence([conflict()], [chunk("safety"), chunk("quality")], rules=[{"type": "nope"}])


# ---------- pipeline ----------

@dataclass
class FakeRoute:
    scores: dict
    reasoning: str = "because"

    @property
    def ranked(self):
        return sorted(self.scores, key=lambda d: self.scores[d], reverse=True)

    @property
    def primary(self):
        return self.ranked[0]

    @property
    def margin(self):
        s = sorted(self.scores.values(), reverse=True)
        return s[0] - s[1]


@pytest.fixture
def logs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOG_DIR", str(tmp_path / "logs"))
    return tmp_path / "logs"


def fakes(scores=None, conflicts=None, not_covered=False):
    calls = {}
    chunks = [chunk("quality", owner="Quality Engineer"), chunk("maintenance", owner="Maintenance Supervisor")]

    def route_fn(q):
        calls["route"] = q
        return FakeRoute(scores or {"safety": 0.05, "maintenance": 0.4, "quality": 0.55})

    def retrieve_fn(q, domains):
        calls["domains"] = domains
        return chunks

    def generate_fn(q, ch):
        return GenResult("answer [1]", [1], not_covered, [conflict()] if conflicts is None else conflicts)

    return dict(router=route_fn, retriever=retrieve_fn, generator=generate_fn), calls


def test_ask_end_to_end(logs):
    kw, calls = fakes()
    r = pipeline.ask("Cap torque is reading 16 in-lb, is that ok?", single_domain_threshold=None, **kw)
    assert calls["domains"] == ["quality", "maintenance"] == r.searched_domains
    assert r.route.primary == "quality" and r.route.confidence == 0.55 and r.route.margin == pytest.approx(0.15)
    assert r.conflicts[0].escalate_to == ["Quality Engineer", "Maintenance Supervisor"]
    assert set(r.latency_ms) == {"route", "retrieve", "generate"}
    assert r.answer == "answer [1]" and not r.not_covered


def test_single_domain_threshold_override(logs):
    kw, calls = fakes({"safety": 0.9, "maintenance": 0.06, "quality": 0.04})
    r = pipeline.ask("q", single_domain_threshold=0.85, **kw)
    assert r.searched_domains == ["safety"] and calls["domains"] == ["safety"]
    r = pipeline.ask("q", single_domain_threshold=None, **kw)
    assert len(r.searched_domains) == 2


def test_logs_written(logs):
    kw, _ = fakes()
    pipeline.ask("q1", single_domain_threshold=None, **kw)
    pipeline.ask("q2", single_domain_threshold=None, **kw)
    q = [json.loads(l) for l in (logs / "queries.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [x["question"] for x in q] == ["q1", "q2"]
    assert q[0]["searched"] == ["quality", "maintenance"] and q[0]["conflicts"][0]["topic"] == "torque"
    assert set(q[0]["latency_ms"]) == {"route", "retrieve", "generate"}
    c = [json.loads(l) for l in (logs / "conflicts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(c) == 2 and c[0]["a"]["domain"] == "quality" and c[0]["b"]["doc"] == "Doc"
    assert c[0]["escalate_to"] == ["Quality Engineer", "Maintenance Supervisor"]


def test_no_conflicts_means_no_conflict_log(logs):
    kw, _ = fakes(conflicts=[])
    pipeline.ask("q", single_domain_threshold=None, **kw)
    assert (logs / "queries.jsonl").exists() and not (logs / "conflicts.jsonl").exists()


def test_logging_failure_does_not_break_ask(logs, monkeypatch):
    def boom(_):
        raise OSError("disk full")
    monkeypatch.setattr(pipeline, "write_logs", boom)
    kw, _ = fakes()
    assert pipeline.ask("q", single_domain_threshold=None, **kw).answer


def test_write_log_false(logs):
    kw, _ = fakes()
    pipeline.ask("q", single_domain_threshold=None, write_log=False, **kw)
    assert not logs.exists()


def test_precomputed_route_skips_router(logs):
    kw, calls = fakes()
    pre = FakeRoute({"safety": 0.1, "maintenance": 0.6, "quality": 0.3})
    kw["router"] = lambda q: pytest.fail("router must not be called")
    r = pipeline.ask("q", route_result=pre, single_domain_threshold=None, **kw)
    assert r.searched_domains == ["maintenance", "quality"] == calls["domains"]
    assert r.route.primary == "maintenance" and r.latency_ms["route"] == 0.0


def test_real_components_are_the_defaults():
    import inspect

    from app import generate, retrieve, router
    d = inspect.signature(pipeline.ask).parameters
    assert d["router"].default is router.route
    assert d["retriever"].default is retrieve.retrieve
    assert d["generator"].default is generate.generate
