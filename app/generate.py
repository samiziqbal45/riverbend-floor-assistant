"""Grounded generation over all retrieved chunks, with a dedicated conflict-detection pass.

generate() makes two LLM calls:
  1. detect_conflicts(): a focused check that returns structured conflicts.
  2. the answer call, which is told about the verified conflicts so its prose and its
     `conflicts` field come from the same list.
If the detection pass fails, generation degrades gracefully (conflict_check_ok=False).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import config
from app import clients
from app.retrieve import Chunk

log = logging.getLogger(__name__)

# TODO: move to config.py once approved (config.py is owned by another session).
MAX_TOKENS = 4096  # answers that restate several conflicts ran close to 1500 and got cut off
RETRIES = 1  # extra attempts after the first failure
KINDS = ("contradiction", "scope_difference", "revision_mismatch")


class GenerationError(RuntimeError):
    pass


CONFLICT_ITEM = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"},
        "claim_a": {"type": "string"}, "cite_a": {"type": "integer"},
        "claim_b": {"type": "string"}, "cite_b": {"type": "integer"},
        "kind": {"type": "string", "enum": list(KINDS)},
    },
    "required": ["topic", "claim_a", "cite_a", "claim_b", "cite_b", "kind"],
}

ANSWER_TOOL = {
    "name": "answer",
    "description": "Return the grounded answer, any document conflicts, and whether the question is covered.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "description": "Concise, step-oriented answer with inline [n] citations."},
            "citations_used": {"type": "array", "items": {"type": "integer"}},
            "not_covered": {"type": "boolean"},
            "conflicts": {"type": "array", "items": CONFLICT_ITEM},
        },
        "required": ["answer", "citations_used", "not_covered", "conflicts"],
    },
}

DETECT_TOOL = {
    "name": "report_conflicts",
    "description": "Report every pair of excerpts that conflict on something relevant to the question (empty list if none).",
    "input_schema": {
        "type": "object",
        "properties": {"conflicts": {"type": "array", "items": CONFLICT_ITEM}},
        "required": ["conflicts"],
    },
}

DETECT_PROMPT = """You audit manufacturing documentation for conflicts. You are given numbered excerpts and a floor supervisor's question.

You MUST respond by calling the `report_conflicts` tool exactly once. Do not reply with plain text.

Report every pair of excerpts (cite_a, cite_b) that give values, limits, or instructions that cannot both be followed for the situation in the question. Look for:
- numeric differences: ranges or limits that differ or only partly overlap (e.g. 18-22 vs 15-20)
- procedural differences: one excerpt allows or recommends what another prohibits, or requires something different
- revision differences: two revisions of the SAME document state the same topic differently
Set kind=contradiction for any conflict between two different documents (even if their revision dates differ); use kind=revision_mismatch ONLY for two revisions of the same document, and kind=scope_difference only as described below.
Be thorough: missing a real conflict is far worse than reporting a borderline one, especially when it concerns safety or product specifications. Compare excerpts from DIFFERENT documents and domains carefully.

Do not report excerpts that are merely complementary, or that concern different equipment or conditions - unless they look like they conflict and you can explain the difference, in which case use kind=scope_difference. Only report conflicts relevant to answering the question. Give each claim as a short, faithful paraphrase of the excerpt. Use ONLY the excerpts. Report an empty list if there are none."""

SYSTEM_PROMPT = """You answer questions from manufacturing floor supervisors using ONLY the numbered documentation excerpts provided.

You MUST respond by calling the `answer` tool exactly once. Do not reply with plain text.

Rules:
- Use ONLY the excerpts. No outside knowledge, no guessing, no general best practice.
- Cite every claim inline as [n], where n is the excerpt number.
- If the excerpts do not answer the question, set not_covered=true, leave conflicts empty, and in `answer` say it is not covered and name the kind of document that would be needed. Do not answer from general knowledge.
- If two excerpts disagree, NEVER merge, average, or pick one. Add an entry to `conflicts` and, in `answer`, write: "The documents disagree on <topic>:" then both positions with their citations, then "Confirm with <owner> before proceeding." Name the owners of the two documents' domains (owners below).
- If the user message lists conflicts found by a separate verification pass, treat them as confirmed: present each one per the rule above and include each in `conflicts`.
- `answer` and `conflicts` must agree: every disagreement you describe in `answer` MUST also be an entry in `conflicts`, and every entry in `conflicts` must be described in `answer`. Never report one without the other. If `conflicts` is empty, `answer` must not say the documents disagree.
- A conflict can be a numeric difference (ranges that differ), a procedural difference (one allows what the other forbids), or a revision difference. Check carefully for these, especially across different domains and documents with different revision dates.
- If two excerpts only appear to disagree because they cover different conditions, explain the distinction in `answer` and report it with kind=scope_difference.
- kind=revision_mismatch is ONLY for the same topic stated differently by different revisions of the SAME document. Conflicts between two different documents are kind=contradiction, even when their revision dates differ.
- Audience: supervisors on the floor. Short, step-oriented, no preamble.

Escalation owners by domain: {owners}"""


def _pages(c: Chunk) -> str:
    return f"p. {c.page_start}" if c.page_start == c.page_end else f"pp. {c.page_start}–{c.page_end}"


def format_context(chunks: list[Chunk]) -> str:
    blocks = [
        f"[{i}] ({c.domain} | {c.doc} | {c.section} | {_pages(c)} | rev {c.revision}, effective {c.effective_date})\n{c.text}"
        for i, c in enumerate(chunks, 1)
    ]
    return "\n\n".join(blocks)


def system_prompt() -> str:
    owners = "; ".join(f"{d}: {o}" for d, o in config.DOMAIN_OWNERS.items())
    return SYSTEM_PROMPT.format(owners=owners)


@dataclass
class Conflict:
    topic: str
    claim_a: str
    cite_a: int
    claim_b: str
    cite_b: int
    kind: str = "contradiction"
    # filled by app.precedence
    resolution: str | None = None  # human-readable outcome when a client rule resolved it
    rule: str | None = None  # which rule type resolved it
    winner_cite: int | None = None
    escalate_to: list[str] = field(default_factory=list)


@dataclass
class GenResult:
    answer: str
    citations_used: list[int]
    not_covered: bool
    conflicts: list[Conflict]
    dropped: list[str] = field(default_factory=list)  # human-readable validation notes
    conflict_check_ok: bool = True  # False if the dedicated detection pass failed


def _parse_conflicts(items, n_chunks: int, dropped: list[str]) -> list[Conflict]:
    """Validate raw conflict dicts: drop malformed / out-of-range ones, coerce unknown kinds."""
    ok = lambda i: isinstance(i, int) and not isinstance(i, bool) and 1 <= i <= n_chunks  # noqa: E731
    out: list[Conflict] = []
    for c in items or []:
        try:
            a, b = c["cite_a"], c["cite_b"]
            conf = Conflict(str(c["topic"]), str(c["claim_a"]), a, str(c["claim_b"]), b,
                            str(c.get("kind", "contradiction")))
        except (KeyError, TypeError, AttributeError):
            dropped.append("malformed conflict")
            log.warning("dropping malformed conflict %r", c)
            continue
        if not (ok(a) and ok(b)) or a == b:
            dropped.append(f"conflict '{conf.topic}' (bad citation)")
            log.warning("dropping conflict %r: bad citation", conf.topic)
            continue
        if conf.kind not in KINDS:
            log.warning("unknown conflict kind %r; treating as contradiction", conf.kind)
            conf.kind = "contradiction"
        out.append(conf)
    return out


def _merge_conflicts(primary: list[Conflict], extra: list[Conflict]) -> list[Conflict]:
    """Union, deduped by unordered citation pair; `primary` wins."""
    seen: set[frozenset] = set()
    out: list[Conflict] = []
    for c in [*primary, *extra]:
        key = frozenset((c.cite_a, c.cite_b))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _validate(raw: dict, n_chunks: int) -> GenResult:
    if not isinstance(raw, dict) or not isinstance(raw.get("answer"), str):
        raise GenerationError(f"malformed tool output: {raw!r}"[:300])
    dropped: list[str] = []
    ok = lambda i: isinstance(i, int) and not isinstance(i, bool) and 1 <= i <= n_chunks  # noqa: E731

    cites = []
    for i in raw.get("citations_used") or []:
        if ok(i):
            if i not in cites:
                cites.append(i)
        else:
            dropped.append(f"citation {i!r}")
            log.warning("dropping out-of-range citation %r (have %d excerpts)", i, n_chunks)

    def strip_bad(m: re.Match) -> str:
        nums = [int(x) for x in re.findall(r"\d+", m.group(0))]
        if all(1 <= x <= n_chunks for x in nums):
            return m.group(0)
        dropped.append(f"inline citation {m.group(0)}")
        log.warning("dropping out-of-range inline citation %s", m.group(0))
        return ""

    answer = re.sub(r"\s?\[\d+(?:\s*,\s*\d+)*\]", strip_bad, raw["answer"])
    conflicts = _parse_conflicts(raw.get("conflicts"), n_chunks, dropped)

    return GenResult(answer.strip(), sorted(cites), bool(raw.get("not_covered", False)), conflicts, dropped)


def _tool_call(llm, system: str, user: str, tool: dict) -> dict:
    """One tool-use request (tool_choice=auto; forced tool use is unsupported), retried once."""
    last: Exception | None = None
    for _ in range(RETRIES + 1):
        try:
            resp = llm.messages.create(
                model=config.LLM_MODEL, max_tokens=MAX_TOKENS, system=system, tools=[tool],
                tool_choice={"type": "auto"}, messages=[{"role": "user", "content": user}],
            )
            if getattr(resp, "stop_reason", None) == "max_tokens":
                raise GenerationError(f"`{tool['name']}` output hit max_tokens ({MAX_TOKENS}) and was cut off")
            block = next((b for b in resp.content if b.type == "tool_use" and b.name == tool["name"]), None)
            if block is None:
                raise GenerationError(f"model replied without calling the `{tool['name']}` tool")
            return block.input
        except Exception as e:  # network/API/shape errors: retry once, then surface clearly
            last = e
            log.warning("%s attempt failed: %s", tool["name"], e)
    raise GenerationError(f"{tool['name']} failed after {RETRIES + 1} attempts: {last}") from last


def detect_conflicts(question: str, chunks: list[Chunk], llm) -> list[Conflict]:
    """Dedicated conflict check over all chunks. Raises GenerationError if the call fails."""
    user = f"Excerpts:\n\n{format_context(chunks)}\n\nQuestion: {question}"
    raw = _tool_call(llm, DETECT_PROMPT, user, DETECT_TOOL)
    return _parse_conflicts(raw.get("conflicts"), len(chunks), [])


def _conflicts_block(conflicts: list[Conflict]) -> str:
    if not conflicts:
        return "A separate verification pass found no conflicts relevant to this question."
    lines = [f'- {c.topic} ({c.kind}): [{c.cite_a}] says "{c.claim_a}" vs [{c.cite_b}] says "{c.claim_b}"'
             for c in conflicts]
    return "Conflicts found by a separate verification pass (confirmed):\n" + "\n".join(lines)


def generate(question: str, chunks: list[Chunk], llm=None) -> GenResult:
    if not chunks:
        return GenResult("Not covered: no relevant documentation was found for this question.", [], True, [])
    llm = llm or clients.get_llm()

    detected: list[Conflict] = []
    check_ok = True
    if len(chunks) >= 2:  # a conflict needs two excerpts
        try:
            detected = detect_conflicts(question, chunks, llm)
        except GenerationError as e:  # degrade: the answer pass can still report conflicts itself
            check_ok = False
            log.warning("conflict detection unavailable: %s", e)

    user = f"Excerpts:\n\n{format_context(chunks)}\n\n{_conflicts_block(detected)}\n\nQuestion: {question}"
    res = _validate(_tool_call(llm, system_prompt(), user, ANSWER_TOOL), len(chunks))
    if res.not_covered:  # nothing relevant found, so conflicts among unrelated excerpts don't matter
        res.conflicts = []
    else:
        res.conflicts = _merge_conflicts(detected, res.conflicts)
    res.conflict_check_ok = check_ok
    if not check_ok:
        res.dropped.append("conflict detection pass failed")
    if not res.conflicts and re.search(r"documents disagree", res.answer, re.I):
        res.dropped.append("answer mentions a disagreement but no conflict was recorded")
        log.warning("answer text reports a disagreement but no conflict was recorded; nothing will be escalated")
    return res
