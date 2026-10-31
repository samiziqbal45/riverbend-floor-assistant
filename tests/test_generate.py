"""Generation tests with a mocked LLM (two tools: report_conflicts, then answer)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app import generate as G
from app.retrieve import Chunk


def chunk(i, domain="safety", p1=2, p2=2, text="text"):
    return Chunk(f"id{i}", 0.9, text, domain, f"Doc {i}", "Sec > Sub", p1, p2, "B", "2025-06-10", "Owner")


def conf(a=1, b=2, topic="torque", kind="contradiction"):
    return {"topic": topic, "claim_a": "18-22", "cite_a": a, "claim_b": "15-20", "cite_b": b, "kind": kind}


class FakeLLM:
    """Dispatches on the requested tool. `answers`/`detects` are consumed in order;
    an Exception entry is raised."""

    def __init__(self, answers=(), detects=None):
        self.answers = list(answers)
        self.detects = list(detects) if detects is not None else None
        self.calls = {"answer": 0, "report_conflicts": 0}
        self.requests = {"answer": [], "report_conflicts": []}
        self.messages = self

    def create(self, **kw):
        name = kw["tools"][0]["name"]
        self.calls[name] += 1
        self.requests[name].append(kw)
        if name == "report_conflicts":
            out = {"conflicts": []} if self.detects is None else self.detects.pop(0)
        else:
            out = self.answers.pop(0)
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(content=[SimpleNamespace(type="tool_use", name=name, input=out)])


GOOD = {"answer": "Lock out first [1]. Torque differs [1][2].", "citations_used": [1, 2], "not_covered": False,
        "conflicts": [conf()]}


def gen(chunks, answers, detects=None):
    llm = FakeLLM(answers, detects)
    return G.generate("q", chunks, llm=llm), llm


# ---------- formatting & prompts ----------

def test_context_formatting():
    ctx = G.format_context([chunk(1, p1=3, p2=3, text="alpha"), chunk(2, "maintenance", p1=1, p2=2, text="beta")])
    assert ctx.startswith("[1] (safety | Doc 1 | Sec > Sub | p. 3 | rev B, effective 2025-06-10)\nalpha")
    assert "[2] (maintenance | Doc 2 | Sec > Sub | pp. 1–2 | rev B, effective 2025-06-10)\nbeta" in ctx


def test_system_prompt_has_rules_and_owners():
    sp = G.system_prompt()
    assert "MUST respond by calling the `answer` tool" in sp
    assert "ONLY" in sp and "The documents disagree on <topic>:" in sp and "Confirm with <owner> before proceeding." in sp
    assert "Quality Engineer" in sp and "EHS Lead" in sp
    assert "`answer` and `conflicts` must agree" in sp and "separate verification pass" in sp


def test_detect_prompt_favours_recall_and_covers_kinds():
    p = G.DETECT_PROMPT
    assert "report_conflicts" in p and "numeric" in p and "procedural" in p and "revision" in p
    assert "missing a real conflict is far worse" in p


# ---------- two-pass flow ----------

def test_detection_runs_first_and_feeds_the_answer_call():
    r, llm = gen([chunk(1), chunk(2, "quality")], [{**GOOD, "conflicts": []}], [{"conflicts": [conf()]}])
    assert llm.calls == {"answer": 1, "report_conflicts": 1}
    det = llm.requests["report_conflicts"][0]
    assert det["tool_choice"] == {"type": "auto"} and det["system"] == G.DETECT_PROMPT
    assert "[2] (quality" in det["messages"][0]["content"]
    ans = llm.requests["answer"][0]
    assert ans["tool_choice"] == {"type": "auto"} and ans["tools"][0]["name"] == "answer"
    body = ans["messages"][0]["content"]
    assert "Conflicts found by a separate verification pass" in body and "18-22" in body and "Question: q" in body
    assert "temperature" not in ans  # this SDK/model rejects it


def test_detected_conflict_survives_when_answer_pass_omits_it():
    """The bug this pass fixes: prose says 'disagree' but the structured list is empty."""
    answer = {**GOOD, "answer": "The documents disagree on torque [1][2].", "conflicts": []}
    r, _ = gen([chunk(1), chunk(2)], [answer], [{"conflicts": [conf()]}])
    assert [(c.topic, c.cite_a, c.cite_b) for c in r.conflicts] == [("torque", 1, 2)]
    assert r.conflict_check_ok and r.dropped == []


def test_merge_dedupes_by_unordered_pair_and_keeps_extras():
    answer = {**GOOD, "conflicts": [conf(2, 1, topic="same pair, reversed"), conf(1, 3, topic="extra")]}
    r, _ = gen([chunk(1), chunk(2), chunk(3)], [answer], [{"conflicts": [conf(1, 2)]}])
    assert [c.topic for c in r.conflicts] == ["torque", "extra"]


def test_no_conflicts_anywhere():
    r, llm = gen([chunk(1), chunk(2)], [{**GOOD, "answer": "Fine [1].", "conflicts": []}])
    assert r.conflicts == [] and r.conflict_check_ok
    assert "found no conflicts" in llm.requests["answer"][0]["messages"][0]["content"]


def test_single_chunk_skips_detection():
    r, llm = gen([chunk(1)], [{**GOOD, "conflicts": []}])
    assert llm.calls["report_conflicts"] == 0 and r.conflict_check_ok


def test_detection_failure_degrades_gracefully(caplog):
    with caplog.at_level("WARNING"):
        r, llm = gen([chunk(1), chunk(2)], [GOOD], [RuntimeError("a"), RuntimeError("b")])
    assert llm.calls["report_conflicts"] == 2  # retried once
    assert r.conflict_check_ok is False and "conflict detection pass failed" in r.dropped
    assert [c.topic for c in r.conflicts] == ["torque"]  # answer-pass conflicts still reported
    assert "conflict detection unavailable" in caplog.text


def test_detection_retry_then_success():
    r, llm = gen([chunk(1), chunk(2)], [{**GOOD, "conflicts": []}], [RuntimeError("boom"), {"conflicts": [conf()]}])
    assert llm.calls["report_conflicts"] == 2 and r.conflict_check_ok and len(r.conflicts) == 1


def test_not_covered_discards_conflicts():
    answer = {"answer": "Not covered.", "citations_used": [], "not_covered": True, "conflicts": []}
    r, _ = gen([chunk(1), chunk(2)], [answer], [{"conflicts": [conf()]}])
    assert r.not_covered and r.conflicts == []


def test_bad_detector_conflicts_are_dropped():
    bad = {"conflicts": [conf(1, 9), conf(1, 1), {"topic": "x"}, conf(1, 2, kind="weird")]}
    r, _ = gen([chunk(1), chunk(2)], [{**GOOD, "conflicts": []}], [bad])
    assert [(c.cite_a, c.cite_b, c.kind) for c in r.conflicts] == [(1, 2, "contradiction")]


# ---------- validation of the answer pass ----------

def test_valid_result():
    r, _ = gen([chunk(1), chunk(2, "quality")], [GOOD])
    assert r.citations_used == [1, 2] and not r.not_covered and r.dropped == []
    assert r.conflicts[0].topic == "torque" and (r.conflicts[0].cite_a, r.conflicts[0].cite_b) == (1, 2)


def test_invalid_citations_dropped_with_warning(caplog):
    bad = {**GOOD, "answer": "Do X [1]. Also Y [9].", "citations_used": [1, 9, 0, 1],
           "conflicts": [conf(), conf(1, 7, topic="t2")]}
    with caplog.at_level("WARNING"):
        r, _ = gen([chunk(1), chunk(2)], [bad])
    assert r.citations_used == [1]
    assert "[9]" not in r.answer and "[1]" in r.answer
    assert [c.topic for c in r.conflicts] == ["torque"]
    assert len(r.dropped) >= 3 and "out-of-range" in caplog.text


def test_unknown_kind_coerced():
    r, _ = gen([chunk(1), chunk(2)], [{**GOOD, "conflicts": [conf(kind="weird")]}])
    assert r.conflicts[0].kind == "contradiction"


def test_malformed_output_raises():
    with pytest.raises(G.GenerationError):
        gen([chunk(1)], [{"nope": 1}])


def test_answer_retry_then_success_and_failure():
    r, llm = gen([chunk(1)], [RuntimeError("boom"), GOOD])
    assert r.answer and llm.calls["answer"] == 2
    with pytest.raises(G.GenerationError, match="2 attempts"):
        gen([chunk(1)], [RuntimeError("a"), RuntimeError("b")])


def test_text_only_reply_is_retried_then_errors():
    class TextLLM(FakeLLM):
        def create(self, **kw):
            self.calls[kw["tools"][0]["name"]] += 1
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="plain answer")])

    llm = TextLLM()
    with pytest.raises(G.GenerationError, match="without calling"):
        G.generate("q", [chunk(1)], llm=llm)
    assert llm.calls["answer"] == 2


def test_truncated_output_is_reported_as_such_and_retried():
    """A response cut off at max_tokens yields an empty tool input; say so instead of 'malformed'."""
    class CutLLM(FakeLLM):
        def create(self, **kw):
            name = kw["tools"][0]["name"]
            self.calls[name] += 1
            return SimpleNamespace(stop_reason="max_tokens",
                                   content=[SimpleNamespace(type="tool_use", name=name, input={})])

    llm = CutLLM()
    with pytest.raises(G.GenerationError, match="max_tokens"):
        G.generate("q", [chunk(1)], llm=llm)
    assert llm.calls["answer"] == 2
    assert G.MAX_TOKENS >= 4096


def test_warns_when_answer_reports_disagreement_without_conflict(caplog):
    out = {**GOOD, "answer": "The documents disagree on torque: A [1] vs B [2].", "conflicts": []}
    with caplog.at_level("WARNING"):
        r, _ = gen([chunk(1), chunk(2)], [out])
    assert r.conflicts == [] and "no conflict was recorded" in caplog.text


def test_no_chunks_is_not_covered_without_llm_call():
    r, llm = gen([], [])
    assert r.not_covered and sum(llm.calls.values()) == 0
