"""Eval harness tests: hand-made sample golden set, all stages faked."""
from __future__ import annotations

import pytest

from app.generate import Conflict, GenResult
from app.retrieve import Chunk
from eval import run_eval as ev

SAMPLE = ev.EVAL_DIR / "golden.sample.jsonl"


def item(**kw):
    base = dict(id="x", question="q", acceptable_domains=["safety"], expected_docs=["Doc A"],
                expect_conflict=False, expect_not_covered=False, category="clean")
    return {**base, **kw}


def out(it, ranked, searched, docs, conflict=None, nc=None):
    return ev.Outcome(it, ranked, searched, docs, conflict, nc)


class FakeRoute:
    def __init__(self, scores):
        self.scores = scores

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

    reasoning = "r"


# ---------- sample golden ----------

def test_sample_golden_well_formed_and_disjoint():
    g = ev.load_jsonl(SAMPLE)
    ev.validate_golden(g)
    ev.assert_disjoint(g, ev.load_jsonl(ev.FEWSHOT))
    assert len(g) == 5 and {x["category"] for x in g} == set(ev.CATEGORIES)


def test_overlap_detected():
    with pytest.raises(SystemExit, match="overlap"):
        ev.assert_disjoint([item(question="Hello  World")], [{"question": "hello world"}])


def test_validate_rejects_bad_items():
    with pytest.raises(SystemExit, match="missing"):
        ev.validate_golden([{"id": "a", "question": "q"}])
    with pytest.raises(SystemExit, match="category"):
        ev.validate_golden([item(category="weird")])
    with pytest.raises(SystemExit, match="duplicate"):
        ev.validate_golden([item(), item()])


# ---------- pure metrics ----------

def test_top1_top2_recall():
    o = out(item(acceptable_domains=["safety", "quality"]), ["safety", "maintenance", "quality"],
            ["safety", "maintenance"], ["Doc A manual"])
    assert ev.top1_ok(o) and not ev.top2_ok(o) and ev.recall_ok(o) == 1.0
    assert ev.recall_ok(out(item(expected_docs=[]), [], [], [])) is None
    assert ev.recall_of(["A", "B"], ["a doc"]) == 0.5


def test_prf_and_percentile():
    r = ev.prf([(True, True), (True, False), (False, True), (False, False)])
    assert r["precision"] == 0.5 and r["recall"] == 0.5
    assert ev.prf([(False, False)])["precision"] is None
    assert ev.percentile([1, 2, 3, 4, 100], 50) == 3 and ev.percentile([1, 2, 3, 4, 100], 95) == 100
    assert ev.percentile([], 50) is None


def test_summarize_refusal_and_conflict():
    outs = [
        out(item(expect_not_covered=True, expected_docs=[]), ["safety"], ["safety"], [], False, True),
        out(item(expect_not_covered=True, expected_docs=[]), ["safety"], ["safety"], [], False, False),
        out(item(expect_conflict=True), ["safety"], ["safety"], ["Doc A"], True, False),
        out(item(), ["safety"], ["safety"], ["Doc A"], True, True),  # false conflict + false refusal
    ]
    s = ev.summarize(outs)
    assert s["refusal"] == 0.5 and s["false_refusal"] == 0.5
    assert s["conflict"]["precision"] == 0.5 and s["conflict"]["recall"] == 1.0


# ---------- end to end with fakes ----------

DOCS = {"s1": "Personal Protective Equipment (PPE) Matrix by Task", "s2": "RP-200 Centrifugal Pump Manual"}


def chunk(doc, domain="safety"):
    return Chunk(f"id-{doc}-{domain}", 0.8, "t", domain, doc, "S", 1, 1, "A", "2025-01-01", "Owner")


class Harness:
    def __init__(self):
        self.route_calls, self.retrieve_calls = [], []

    def router(self, q):
        self.route_calls.append(q)
        if "forklift" in q or "vacation" in q:
            return FakeRoute({"safety": 0.5, "maintenance": 0.4, "quality": 0.1})
        if "PPE" in q:
            return FakeRoute({"safety": 0.9, "maintenance": 0.06, "quality": 0.04})
        if "RP-200 mechanical" in q:
            return FakeRoute({"safety": 0.1, "maintenance": 0.85, "quality": 0.05})
        return FakeRoute({"safety": 0.5, "maintenance": 0.45, "quality": 0.05})

    def retriever(self, q, domains):
        self.retrieve_calls.append((q, tuple(domains)))
        docs = {"safety": ["Personal Protective Equipment (PPE) Matrix by Task", "Conveyor Jam Clearing Safety Procedure",
                           "Chemical Spill and Sanitizer Handling Response"],
                "maintenance": ["RP-200 Centrifugal Pump Manual", "Conveyor Maintenance and Jam Troubleshooting Guide"]}
        return [chunk(d, dom) for dom in domains for d in docs.get(dom, [])]

    def generator(self, q, chunks):
        if "forklift" in q or "vacation" in q:
            return GenResult("not covered", [], True, [])
        if "jog" in q:
            return GenResult("conflict", [1, 2], False, [Conflict("jog", "a", 1, "b", 2)])
        return GenResult("answer [1]", [1], False, [])


def test_full_run_on_sample_with_fakes(tmp_path):
    h = Harness()
    golden = ev.load_jsonl(SAMPLE)
    outs, routes = ev.evaluate(golden, router=h.router, retriever=h.retriever, generator=h.generator, verbose=False)
    assert len(h.route_calls) == 5  # one routing call per question
    by = {o.item["id"]: o for o in outs}
    assert by["s1"].searched == ["safety", "maintenance"] and by["s1"].conflict is False
    assert by["s4"].conflict is True and by["s5"].not_covered is True
    assert all(set(o.latency) == {"route", "retrieve", "generate"} for o in outs)

    n_before = len(h.retrieve_calls)
    rows = ev.sweep(golden, routes, retriever=h.retriever)
    assert [r["threshold"] for r in rows] == ev.SWEEP
    assert len(h.route_calls) == 5  # the sweep reuses cached routes, never re-routes
    assert len(h.retrieve_calls) > n_before  # ...but does re-run retrieval
    assert rows[0]["pct_single"] == 0.0  # off => always two domains
    assert rows[-1]["pct_single"] == pytest.approx(0.2)  # only s1 (0.9) goes single at 0.9
    # a lower threshold can only send MORE queries to single-domain search
    pcts = [r["pct_single"] for r in rows[1:]]  # thresholds 0.6, 0.7, 0.8, 0.9
    assert pcts == sorted(pcts, reverse=True)

    path = ev.write_outputs(outs, rows, tmp_path / "res", golden_path=str(SAMPLE), skip_generation=False)
    text = path.read_text(encoding="utf-8")
    assert (tmp_path / "res" / "threshold_sweep.png").stat().st_size > 1000
    for section in ("## Summary", "## Latency (ms)", "## Threshold sweep", "## Failures", "conflict P"):
        assert section in text


def test_failures_listed_with_expected_vs_actual(tmp_path):
    h = Harness()
    golden = [dict(ev.load_jsonl(SAMPLE)[0], expect_conflict=True)]  # s1 will not produce a conflict
    outs, _ = ev.evaluate(golden, router=h.router, retriever=h.retriever, generator=h.generator, verbose=False)
    [(i, q, e, a)] = ev.failures(outs)
    assert (i, e, a) == ("s1", "conflict=True", "conflict=False") and "PPE" in q


def test_skip_generation_never_calls_generator():
    h = Harness()
    gen = lambda *a: pytest.fail("generator must not run")  # noqa: E731
    outs, routes = ev.evaluate(ev.load_jsonl(SAMPLE), skip_generation=True, router=h.router,
                               retriever=h.retriever, generator=gen, verbose=False)
    assert all(o.conflict is None and "retrieve" in o.latency and "generate" not in o.latency for o in outs)
    text = ev.report(outs, ev.sweep(ev.load_jsonl(SAMPLE), routes, retriever=h.retriever), skip_generation=True)
    assert "conflict P" not in text and "--skip-generation" in text


def test_errors_are_recorded_not_raised():
    def bad_router(q):
        raise RuntimeError("api down")
    outs, routes = ev.evaluate(ev.load_jsonl(SAMPLE)[:2], router=bad_router, retriever=lambda *a: [],
                               generator=lambda *a: None, verbose=False)
    assert routes == {} and all(o.error for o in outs)
    assert ev.failures(outs)[0][2] == "pipeline completes"
    rows = ev.sweep(ev.load_jsonl(SAMPLE)[:2], routes, retriever=lambda *a: [])
    assert rows[0]["top2_coverage"] == 0.0


def test_main_cli_flags(tmp_path, monkeypatch):
    h = Harness()
    monkeypatch.setattr(ev, "_default_router", h.router)
    monkeypatch.setattr(ev, "_default_retriever", h.retriever)
    rc = ev.main(["--golden", str(SAMPLE), "--skip-generation", "--limit", "3", "--results-dir", str(tmp_path)])
    assert rc == 0 and len(h.route_calls) == 3
    assert (tmp_path / "report.md").exists() and (tmp_path / "threshold_sweep.png").exists()


def test_main_rejects_overlap_with_fewshot(tmp_path):
    fewshot_q = ev.load_jsonl(ev.FEWSHOT)[0]["question"]
    bad = tmp_path / "g.jsonl"
    bad.write_text(__import__("json").dumps(item(question=fewshot_q)) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="overlap"):
        ev.main(["--golden", str(bad), "--skip-generation", "--results-dir", str(tmp_path)])
