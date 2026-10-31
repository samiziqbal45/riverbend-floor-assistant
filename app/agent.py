"""Agentic variant of the pipeline: the MODEL decides what to search, through a real tool.

Unlike app.pipeline (a fixed route -> retrieve -> generate sequence), here the model is given a
`search_documents` tool, our code executes each call against Qdrant and feeds the results back,
and the loop continues until the model calls `answer`.

    from app.agent import run_agent
    res = run_agent("What inspection checks should be done for bottles?")
    res.answer, res.conflicts, res.steps

What is reused unchanged: Chunk / search_domain (retrieval), ANSWER_TOOL + validation + conflict
detection (app.generate), and apply_precedence. What is new: the loop, the search tool, the
evidence registry that keeps [n] citation numbers stable across searches, and the guardrails.

Guardrails: bounded turns, a last turn that only offers `answer`, no repeated identical searches,
no ungrounded answers (the model must search first), tool errors returned to the model instead of
raised, and the same separate conflict-detection pass as the fixed pipeline (the model alone was
unreliable at filling the structured conflict list).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import config
from app import clients
from app import generate as G
from app.generate import ANSWER_TOOL, Conflict, GenerationError
from app.precedence import apply_precedence
from app.retrieve import Chunk, search_domain

log = logging.getLogger(__name__)

# TODO: move to config.py once approved (config.py is owned by another session).
MAX_STEPS = 6  # LLM turns, including the final `answer` turn
DEFAULT_RESULTS = 5
MAX_RESULTS = 8
MAX_TOKENS = 2000
RETRIES = 1
SEARCH = "search_documents"


class AgentError(RuntimeError):
    pass


SEARCH_TOOL = {
    "name": SEARCH,
    "description": (
        "Search the plant's controlled documents. Returns the best-matching excerpts, each numbered [n] "
        "(cite these numbers in your answer). Domains: safety (hazard controls, lockout/tagout, PPE, "
        "emergency response), maintenance (operation, repair, troubleshooting, preventive maintenance), "
        "quality (specifications, tolerances, inspection, SPC, nonconformance and holds), or all."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for, in plain words. Be specific."},
            "domain": {"type": "string", "enum": [*config.DOMAINS, "all"]},
            "k": {"type": "integer", "minimum": 1, "maximum": MAX_RESULTS,
                  "description": f"How many excerpts to return (default {DEFAULT_RESULTS})."},
        },
        "required": ["query", "domain"],
    },
}

AGENT_PREFIX = """You are a documentation research assistant for manufacturing floor supervisors. You have a tool, `search_documents`, that searches the plant's controlled documents. Search results are numbered [n]; those numbers are what you cite.

How to work:
1. Search before answering. Never answer from memory; you have no knowledge of this plant's documents except what the tool returns.
2. Search the domain that should answer the question. When the topic could ALSO be governed by another domain (a specification in quality and a setting in maintenance; a repair procedure in maintenance and a hazard rule in safety), search that domain as well: documents from different owners sometimes disagree, and finding that out is part of your job. If one domain clearly covers the question and nothing suggests another does, one search is enough.
3. If results are weak or off-topic, rephrase and search again (different wording, a narrower question, another domain). Do not repeat an identical search. You have at most {max_steps} turns, the last of which must be the `answer` call.
4. When you have what you need, call the `answer` tool. If nothing relevant turned up after a reasonable search, call it with not_covered=true.

The rules for the final answer follow."""


def agent_prompt(max_steps: int = MAX_STEPS) -> str:
    return AGENT_PREFIX.format(max_steps=max_steps) + "\n\n" + G.system_prompt()


# ---------- search execution (plain Python; the model never touches Qdrant) ----------

class Searcher:
    """Embeds a query and searches one domain, or all of them merged by score."""

    def __init__(self, client=None, embedder=None):
        self._client, self._embedder = client, embedder

    def search(self, query: str, domain: str, k: int) -> list[Chunk]:
        client = self._client or clients.get_qdrant()
        embedder = self._embedder or clients.get_embedder()
        qvec = [float(x) for x in next(iter(embedder.embed([query])))]
        domains = list(config.DOMAINS) if domain == "all" else [domain]
        hits = [h for d in domains for h in search_domain(client, qvec, d, k)]
        return sorted(hits, key=lambda c: c.score, reverse=True)[:k]


@dataclass
class Evidence:
    """Every chunk the agent has seen, numbered by first appearance so [n] stays stable."""
    chunks: list[Chunk] = field(default_factory=list)
    _index: dict[str, int] = field(default_factory=dict)

    def add(self, c: Chunk) -> tuple[int, bool]:
        if c.id in self._index:
            return self._index[c.id], False
        self.chunks.append(c)
        self._index[c.id] = len(self.chunks)
        return len(self.chunks), True


def _header(n: int, c: Chunk) -> str:
    return f"[{n}] ({c.domain} | {c.doc} | {c.section} | {G._pages(c)} | rev {c.revision}, effective {c.effective_date})"


@dataclass
class SearchStep:
    turn: int
    query: str
    domain: str
    k: int
    returned: list[int]  # [n] labels of everything returned
    new: list[int]  # labels first seen in this call
    note: str = ""  # "duplicate", "error: ...", "no results"


def execute_search(args: dict, searcher, evidence: Evidence, seen: set, turn: int) -> tuple[str, SearchStep, bool]:
    """Run one search tool call. Returns (text for the model, trace record, is_error)."""
    query = args.get("query")
    domain = args.get("domain")
    k = args.get("k", DEFAULT_RESULTS)
    if not isinstance(query, str) or not query.strip():
        return "Error: `query` must be a non-empty string.", SearchStep(turn, str(query), str(domain), 0, [], [], "error: bad query"), True
    if domain not in (*config.DOMAINS, "all"):
        return (f"Error: `domain` must be one of {[*config.DOMAINS, 'all']}, got {domain!r}.",
                SearchStep(turn, query, str(domain), 0, [], [], "error: bad domain"), True)
    k = max(1, min(MAX_RESULTS, k if isinstance(k, int) and not isinstance(k, bool) else DEFAULT_RESULTS))

    key = (" ".join(query.lower().split()), domain)
    if key in seen:
        return ("You already ran this exact search; its results are above. Rephrase, try another domain, "
                "or call `answer`."), SearchStep(turn, query, domain, k, [], [], "duplicate"), False
    seen.add(key)

    try:
        hits = searcher.search(query, domain, k)
    except Exception as e:  # return the failure to the model so it can recover
        log.warning("search_documents failed: %s", e)
        return f"Error: search failed ({type(e).__name__}). Try again or call `answer`.", \
            SearchStep(turn, query, domain, k, [], [], f"error: {type(e).__name__}"), True
    if not hits:
        return f'No results for "{query}" in {domain}.', SearchStep(turn, query, domain, k, [], [], "no results"), False

    lines, returned, new = [f'Results for "{query}" in {domain}:'], [], []
    for c in hits:
        n, is_new = evidence.add(c)
        returned.append(n)
        if is_new:
            new.append(n)
            lines.append(f"{_header(n, c)}\n{c.text}")
        else:
            lines.append(f"{_header(n, c)}\n(already shown above)")
    return "\n\n".join(lines), SearchStep(turn, query, domain, k, returned, new), False


# ---------- the loop ----------

@dataclass
class AgentResult:
    question: str
    answer: str
    conflicts: list[Conflict]  # post-precedence
    not_covered: bool
    citations_used: list[int]
    chunks: list[Chunk]  # evidence, in [n] order
    steps: list[SearchStep]
    llm_calls: int
    latency_ms: float
    conflict_check_ok: bool = True
    dropped: list[str] = field(default_factory=list)


def _create(llm, system: str, tools: list[dict], messages: list):
    last: Exception | None = None
    for _ in range(RETRIES + 1):
        try:
            return llm.messages.create(model=config.LLM_MODEL, max_tokens=MAX_TOKENS, system=system, tools=tools,
                                       tool_choice={"type": "auto"}, messages=messages)
        except Exception as e:  # noqa: BLE001
            last = e
            log.warning("agent LLM call failed: %s", e)
    raise AgentError(f"LLM call failed after {RETRIES + 1} attempts: {last}") from last


def run_agent(question: str, *, llm=None, searcher=None, max_steps: int = MAX_STEPS,
              verify_conflicts: bool = True) -> AgentResult:
    t0 = time.perf_counter()
    llm = llm or clients.get_llm()
    searcher = searcher or Searcher()
    system = agent_prompt(max_steps)
    evidence, steps, seen = Evidence(), [], set()
    messages: list = [{"role": "user", "content": f"Question: {question}"}]
    calls, nudged, pushed_back, final = 0, False, False, None

    for turn in range(1, max_steps + 1):
        # Last turn: only `answer` is offered, so the budget cannot be exhausted without an answer.
        tools = [ANSWER_TOOL] if turn == max_steps else [SEARCH_TOOL, ANSWER_TOOL]
        resp = _create(llm, system, tools, messages)
        calls += 1
        messages.append({"role": "assistant", "content": resp.content})
        uses = [b for b in resp.content if b.type == "tool_use"]

        answer_call = next((b for b in uses if b.name == ANSWER_TOOL["name"]), None)
        if answer_call is not None:
            raw = answer_call.input
            if not evidence.chunks and not (isinstance(raw, dict) and raw.get("not_covered")) and not pushed_back:
                # Ungrounded: the model answered without searching. Push back once.
                pushed_back = True
                messages.append({"role": "user", "content": [{
                    "type": "tool_result", "tool_use_id": answer_call.id, "is_error": True,
                    "content": "You have not searched yet. Use `search_documents` first; answers must come from search results."}]})
                continue
            final = raw
            break

        searches = [b for b in uses if b.name == SEARCH]
        if not searches:
            if nudged:
                raise AgentError("model ended its turn without calling `search_documents` or `answer`")
            nudged = True
            messages.append({"role": "user", "content": "Continue: call `search_documents` or, if you are done, the `answer` tool."})
            continue

        results = []
        for b in searches:  # several tool calls in one turn are answered together
            text, step, is_err = execute_search(b.input, searcher, evidence, seen, turn)
            steps.append(step)
            res = {"type": "tool_result", "tool_use_id": b.id, "content": text}
            if is_err:
                res["is_error"] = True
            results.append(res)
        messages.append({"role": "user", "content": results})

    if final is None:
        raise AgentError(f"no answer after {max_steps} turns")

    res = G._validate(final, len(evidence.chunks))
    if not evidence.chunks and not res.not_covered:
        raise AgentError("model answered without any search results")

    conflict_ok = True
    if res.not_covered:
        res.conflicts = []
    else:
        detected: list[Conflict] = []
        if verify_conflicts and len(evidence.chunks) >= 2:
            try:
                detected = G.detect_conflicts(question, evidence.chunks, llm)
                calls += 1
            except GenerationError as e:
                conflict_ok = False
                res.dropped.append("conflict detection pass failed")
                log.warning("conflict detection unavailable: %s", e)
        res.conflicts = G._merge_conflicts(detected, res.conflicts)

    return AgentResult(
        question=question, answer=res.answer, conflicts=apply_precedence(res.conflicts, evidence.chunks),
        not_covered=res.not_covered, citations_used=res.citations_used, chunks=evidence.chunks, steps=steps,
        llm_calls=calls, latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        conflict_check_ok=conflict_ok, dropped=res.dropped,
    )
