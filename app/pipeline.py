"""ask(): route -> filtered retrieval -> one generation call -> precedence -> logging."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import config
from app import retrieve as retrieve_mod  # module access so choose_domains stays patchable
from app.generate import Conflict, generate
from app.precedence import apply_precedence
from app.retrieve import Chunk, retrieve
from app.router import Route, route

log = logging.getLogger(__name__)
_UNSET = object()


@dataclass
class RouteInfo:
    scores: dict[str, float]
    reasoning: str
    primary: str
    confidence: float  # normalized score of the primary domain
    margin: float  # top1 - top2


@dataclass
class AskResult:
    question: str
    route: RouteInfo
    searched_domains: list[str]
    chunks: list[Chunk]
    answer: str
    conflicts: list[Conflict]  # post-precedence
    not_covered: bool
    citations_used: list[int] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)  # route, retrieve, generate

    def to_dict(self) -> dict:
        return asdict(self)


def _log_dir() -> Path:
    p = Path(config.LOG_DIR)
    return p if p.is_absolute() else config.ROOT / p


def _append(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _conflict_record(ts: str, question: str, c: Conflict, chunks: list[Chunk]) -> dict:
    def side(n: int, claim: str) -> dict:
        ch = chunks[n - 1]
        return {"claim": claim, "cite": n, "domain": ch.domain, "doc": ch.doc, "section": ch.section,
                "pages": [ch.page_start, ch.page_end], "revision": ch.revision,
                "effective_date": ch.effective_date, "source_path": ch.source_path}

    return {"ts": ts, "question": question, "topic": c.topic, "kind": c.kind,
            "a": side(c.cite_a, c.claim_a), "b": side(c.cite_b, c.claim_b),
            "resolution": c.resolution, "rule": c.rule, "escalate_to": c.escalate_to}


def write_logs(result: AskResult) -> None:
    """One line per query in queries.jsonl; one per conflict in conflicts.jsonl (doc-quality report)."""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    d = _log_dir()
    d.mkdir(parents=True, exist_ok=True)
    _append(d / "queries.jsonl", {
        "ts": ts, "question": result.question, "route": asdict(result.route),
        "searched": result.searched_domains,
        "sources": [{"domain": c.domain, "doc": c.doc, "section": c.section, "score": round(c.score, 4)}
                    for c in result.chunks],
        "conflicts": [asdict(c) for c in result.conflicts],
        "not_covered": result.not_covered, "latency_ms": result.latency_ms,
    })
    for c in result.conflicts:
        _append(d / "conflicts.jsonl", _conflict_record(ts, result.question, c, result.chunks))


def _timed(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return out, round((time.perf_counter() - t0) * 1000, 1)


def ask(question: str, *, router=route, retriever=retrieve, generator=generate,
        route_result: Route | None = None, single_domain_threshold=_UNSET,
        write_log: bool = True) -> AskResult:
    """Run the full pipeline.

    router/retriever/generator are injectable stages. Pass `route_result` to skip the
    router call and reuse a precomputed Route (its route latency is then reported as 0).
    `single_domain_threshold` overrides config.SINGLE_DOMAIN_THRESHOLD for this call.
    """
    threshold = config.SINGLE_DOMAIN_THRESHOLD if single_domain_threshold is _UNSET else single_domain_threshold

    if route_result is not None:
        rt, t_route = route_result, 0.0
    else:
        rt, t_route = _timed(router, question)
    domains = retrieve_mod.choose_domains(rt, threshold)
    chunks, t_ret = _timed(retriever, question, domains)
    gen, t_gen = _timed(generator, question, chunks)
    conflicts = apply_precedence(gen.conflicts, chunks)

    result = AskResult(
        question=question,
        route=RouteInfo(dict(rt.scores), rt.reasoning, rt.primary,
                        round(rt.scores[rt.primary], 4), round(rt.margin, 4)),
        searched_domains=domains, chunks=chunks, answer=gen.answer, conflicts=conflicts,
        not_covered=gen.not_covered, citations_used=gen.citations_used,
        latency_ms={"route": t_route, "retrieve": t_ret, "generate": t_gen},
    )
    if write_log:
        try:
            write_logs(result)
        except OSError as e:  # logging must never break an answer
            log.warning("could not write logs: %s", e)
    return result
