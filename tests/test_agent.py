"""Agent loop tests: scripted LLM, fake searcher, no network."""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

import config
from app import agent as A
from app.generate import ANSWER_TOOL, DETECT_TOOL
from app.retrieve import Chunk


def chunk(cid, domain="quality", doc="Doc", section="Sec", text="body", score=0.8):
    return Chunk(cid, score, text, domain, doc, section, 1, 1, "B", "2025-01-01", "Owner")


def search(query="q", domain="quality", k=None, tid="t1"):
    args = {"query": query, "domain": domain, **({"k": k} if k else {})}
    return NS(type="tool_use", id=tid, name="search_documents", input=args)


def answer(text="done [1]", cites=(1,), nc=False, conflicts=(), tid="a1"):
    return NS(type="tool_use", id=tid, name="answer",
              input={"answer": text, "citations_used": list(cites), "not_covered": nc, "conflicts": list(conflicts)})


def say(text):
    return NS(type="text", text=text)


class ScriptedLLM:
    """Each agent turn pops the next scripted content list. detect_conflicts calls are answered separately."""

    def __init__(self, script, detects=None):
        self.script, self.detects = list(script), list(detects or [])
        self.turns, self.detect_calls = [], 0
        self.messages = self

    def create(self, **kw):
        if kw["tools"][0]["name"] == DETECT_TOOL["name"]:
            self.detect_calls += 1
            out = self.detects.pop(0) if self.detects else {"conflicts": []}
            if isinstance(out, Exception):
                raise out
            return NS(content=[NS(type="tool_use", id="d", name="report_conflicts", input=out)])
        self.turns.append({"tools": [t["name"] for t in kw["tools"]], "messages": list(kw["messages"]),
                           "system": kw["system"], "tool_choice": kw["tool_choice"]})
        out = self.script.pop(0)
        if isinstance(out, Exception):
            raise out
        return NS(content=out)


class FakeSearcher:
    def __init__(self, by_domain):
        self.by_domain, self.calls = by_domain, []

    def search(self, query, domain, k):
        self.calls.append((query, domain, k))
        if isinstance(self.by_domain, Exception):
            raise self.by_domain
        if domain == "all":
            return sorted([c for v in self.by_domain.values() for c in v], key=lambda c: -c.score)[:k]
        return self.by_domain.get(domain, [])[:k]


Q1, Q2 = chunk("c1", "quality", "Incoming Inspection", "Bottles", "check neck finish"), chunk("c2", "quality", "Incoming Inspection", "Caps")
M1 = chunk("m1", "maintenance", "CX-4 Manual", "Calibration", "set heads to 15-20")


def run(script, by_domain=None, detects=None, **kw):
    llm, s = ScriptedLLM(script, detects), FakeSearcher(by_domain if by_domain is not None else {"quality": [Q1, Q2], "maintenance": [M1]})
    return A.run_agent("question?", llm=llm, searcher=s, **kw), llm, s


def tool_results(turn):
    return turn["messages"][-1]["content"]


# ---------- the core loop ----------

def test_search_then_answer_feeds_results_back_as_tool_result():
    res, llm, s = run([[search("bottle checks", "quality")], [answer("Check neck finish [1].", cites=[1])]])
    assert s.calls == [("bottle checks", "quality", A.DEFAULT_RESULTS)]
    assert len(llm.turns) == 2 and res.llm_calls == 3  # 2 agent turns + the conflict check
    [tr] = tool_results(llm.turns[1])
    assert tr["type"] == "tool_result" and tr["tool_use_id"] == "t1" and "is_error" not in tr
    assert "[1] (quality | Incoming Inspection | Bottles" in tr["content"] and "check neck finish" in tr["content"]
    assert [c.id for c in res.chunks] == ["c1", "c2"] and res.citations_used == [1]
    assert res.steps[0].returned == [1, 2] and res.steps[0].new == [1, 2]
    assert llm.turns[0]["tool_choice"] == {"type": "auto"} and llm.turns[0]["tools"] == ["search_documents", "answer"]


def test_citation_numbers_are_stable_across_searches_and_repeats_are_not_resent():
    res, llm, _ = run([
        [search("a", "quality")],
        [search("b", "all", tid="t2")],  # returns M1, Q1, Q2 sorted by score: M1 is new, Q1/Q2 are repeats
        [answer("x [3]", cites=[3])],
    ], by_domain={"quality": [Q1, Q2], "maintenance": [chunk("m1", "maintenance", score=0.9, text="NEW TEXT")]})
    [tr] = tool_results(llm.turns[2])
    assert [c.id for c in res.chunks] == ["c1", "c2", "m1"]
    assert "[3] (maintenance" in tr["content"] and "NEW TEXT" in tr["content"]
    assert tr["content"].count("(already shown above)") == 2 and "check neck finish" not in tr["content"]
    assert res.steps[1].new == [3] and res.steps[1].returned == [3, 1, 2]


def test_several_tool_calls_in_one_turn_are_answered_together():
    res, llm, s = run([[search("a", "quality", tid="t1"), search("b", "maintenance", tid="t2")], [answer()]])
    results = tool_results(llm.turns[1])
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"] and len(s.calls) == 2


def test_duplicate_search_is_not_executed_twice():
    res, llm, s = run([[search("Bottle  checks", "quality")], [search("bottle checks", "quality", tid="t2")], [answer()]])
    assert len(s.calls) == 1
    assert "already ran this exact search" in tool_results(llm.turns[2])[0]["content"]
    assert res.steps[1].note == "duplicate"


def test_invalid_domain_and_query_are_returned_as_errors_not_raised():
    res, llm, s = run([[search("x", "finance")], [search("  ", "quality", tid="t2")], [answer(nc=True, cites=[])]])
    assert s.calls == []
    assert tool_results(llm.turns[1])[0]["is_error"] and "domain" in tool_results(llm.turns[1])[0]["content"]
    assert tool_results(llm.turns[2])[0]["is_error"] and "query" in tool_results(llm.turns[2])[0]["content"]


def test_search_failure_is_reported_to_the_model():
    res, llm, _ = run([[search()], [answer(nc=True, cites=[])]], by_domain=RuntimeError("qdrant down"))
    assert tool_results(llm.turns[1])[0]["is_error"] and "RuntimeError" in tool_results(llm.turns[1])[0]["content"]


def test_k_is_clamped():
    _, _, s = run([[search(k=99)], [answer()]])
    assert s.calls[0][2] == A.MAX_RESULTS


def test_no_results_message():
    _, llm, _ = run([[search(domain="safety")], [answer(nc=True, cites=[])]])
    assert "No results" in tool_results(llm.turns[1])[0]["content"]


# ---------- guardrails ----------

def test_last_turn_only_offers_answer():
    s1, s2 = search("a", "quality"), search("b", "maintenance", tid="t2")
    res, llm, _ = run([[s1], [s2], [answer()]], max_steps=3)
    assert llm.turns[0]["tools"] == ["search_documents", "answer"]
    assert llm.turns[2]["tools"] == ["answer"]  # budget turn cannot be spent searching


def test_no_answer_within_budget_raises():
    with pytest.raises(A.AgentError, match="no answer after 2 turns"):
        run([[search("a")], [say("still thinking")], [say("x")]], max_steps=2)


def test_answer_without_searching_is_pushed_back_once():
    res, llm, _ = run([[answer("from memory", cites=[])], [search()], [answer("grounded [1]")]])
    pb = tool_results(llm.turns[1])[0]
    assert pb["tool_use_id"] == "a1" and pb["is_error"] and "not searched" in pb["content"]
    assert res.answer == "grounded [1]"


def test_answer_without_searching_twice_raises():
    with pytest.raises(A.AgentError, match="without any search"):
        run([[answer("memory", cites=[])], [answer("memory again", cites=[], tid="a2")]])


def test_not_covered_without_searching_is_allowed():
    res, _, _ = run([[answer("Not covered.", cites=[], nc=True)]])
    assert res.not_covered and res.chunks == []


def test_text_only_turn_is_nudged_once_then_errors():
    res, llm, _ = run([[say("hmm")], [search()], [answer()]])
    assert "Continue:" in llm.turns[1]["messages"][-1]["content"]
    with pytest.raises(A.AgentError, match="without calling"):
        run([[say("hmm")], [say("hmm again")]])


def test_llm_failure_is_retried_then_raises():
    res, llm, _ = run([RuntimeError("boom"), [search()], [answer()]], verify_conflicts=False)
    assert res.llm_calls == 2  # the failed attempt isn't a completed turn
    with pytest.raises(A.AgentError, match="2 attempts"):
        run([RuntimeError("a"), RuntimeError("b")])


def test_invalid_citations_are_dropped_by_the_shared_validator():
    res, _, _ = run([[search()], [answer("x [1] [9]", cites=[1, 9])]])
    assert res.citations_used == [1] and "[9]" not in res.answer


# ---------- conflicts ----------

CONFLICT = {"topic": "torque", "claim_a": "18-22", "cite_a": 1, "claim_b": "15-20", "cite_b": 3, "kind": "contradiction"}


def test_separate_conflict_pass_runs_over_the_collected_evidence_and_precedence_escalates():
    res, llm, _ = run([[search("a", "quality")], [search("b", "maintenance", tid="t2")],
                       [answer("The documents disagree [1][3]", cites=[1, 3], conflicts=[])]],
                      detects=[{"conflicts": [CONFLICT]}])
    assert llm.detect_calls == 1 and res.llm_calls == 4
    [c] = res.conflicts
    assert (c.cite_a, c.cite_b) == (1, 3) and c.escalate_to == ["Owner"]  # chunk owners, deduped
    assert res.conflict_check_ok


def test_verify_conflicts_can_be_disabled_and_single_chunk_skips_it():
    _, llm, _ = run([[search()], [answer()]], verify_conflicts=False)
    assert llm.detect_calls == 0
    _, llm, _ = run([[search()], [answer()]], by_domain={"quality": [Q1]})
    assert llm.detect_calls == 0


def test_conflict_detection_failure_degrades():
    res, _, _ = run([[search()], [answer(conflicts=[{**CONFLICT, "cite_b": 2}])]],
                    detects=[RuntimeError("a"), RuntimeError("b")])
    assert not res.conflict_check_ok and "conflict detection pass failed" in res.dropped
    assert len(res.conflicts) == 1  # the model's own conflict is kept


def test_not_covered_discards_conflicts():
    res, llm, _ = run([[search()], [answer("Not covered", cites=[], nc=True, conflicts=[CONFLICT])]],
                      detects=[{"conflicts": [CONFLICT]}])
    assert res.not_covered and res.conflicts == [] and llm.detect_calls == 0


# ---------- prompt, tools, Searcher ----------

def test_prompt_and_tool_schema():
    p = A.agent_prompt(5)
    assert "Search before answering" in p and "at most 5 turns" in p and "ONLY the numbered documentation excerpts" in p
    props = A.SEARCH_TOOL["input_schema"]["properties"]
    assert props["domain"]["enum"] == [*config.DOMAINS, "all"] and A.SEARCH_TOOL["input_schema"]["required"] == ["query", "domain"]
    assert A.SEARCH_TOOL["name"] == "search_documents" and ANSWER_TOOL["name"] == "answer"


def test_searcher_all_merges_domains_by_score(monkeypatch):
    import numpy as np

    hits = {"safety": [chunk("s1", "safety", score=0.5)], "maintenance": [chunk("m1", "maintenance", score=0.9)],
            "quality": [chunk("q1", "quality", score=0.7)]}
    monkeypatch.setattr(A, "search_domain", lambda client, qvec, d, k: hits[d])
    s = A.Searcher(client=object(), embedder=NS(embed=lambda texts: iter([np.zeros(3)])))
    assert [c.id for c in s.search("x", "all", 2)] == ["m1", "q1"]
    assert [c.id for c in s.search("x", "safety", 5)] == ["s1"]
