"""Dual filtered retrieval: one Qdrant search per routed domain, merged with a per-domain floor."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import config


@dataclass
class Chunk:
    id: str
    score: float
    text: str
    domain: str
    doc: str
    section: str
    page_start: int
    page_end: int
    revision: str
    effective_date: str
    owner: str
    chunk_index: int = 0
    source_path: str = ""

    @classmethod
    def from_point(cls, point) -> "Chunk":
        p = point.payload
        return cls(
            id=str(point.id), score=float(point.score), text=p["text"], domain=p["domain"],
            doc=p["doc"], section=p["section"], page_start=p["page_start"], page_end=p["page_end"],
            revision=p["revision"], effective_date=p["effective_date"], owner=p["owner"],
            chunk_index=p.get("chunk_index", 0), source_path=p.get("source_path", ""),
        )


class RouteLike(Protocol):
    scores: dict[str, float]

    @property
    def ranked(self) -> list[str]: ...


def choose_domains(route: RouteLike, threshold: float | None = config.SINGLE_DOMAIN_THRESHOLD) -> list[str]:
    """Top-2 domains by default; top-1 only if a threshold is set and the top score reaches it."""
    ranked = route.ranked
    if threshold is not None and route.scores[ranked[0]] >= threshold:
        return ranked[:1]
    return ranked[:2]


def search_domain(client, qvec: list[float], domain: str, k: int = config.K_PER_DOMAIN) -> list[Chunk]:
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    res = client.query_points(
        config.COLLECTION,
        query=qvec,
        query_filter=Filter(must=[FieldCondition(key="domain", match=MatchValue(value=domain))]),
        limit=k,
        with_payload=True,
    )
    return [Chunk.from_point(p) for p in res.points]


def merge(per_domain: dict[str, list[Chunk]], final_k: int = config.FINAL_K,
          min_per_domain: int = config.MIN_PER_DOMAIN) -> list[Chunk]:
    """Floor first (top N of each domain), then fill to final_k by raw score; dedupe; sort desc.

    A domain with no hits simply contributes nothing, so the other domain fills the slots.
    """
    picked: dict[str, Chunk] = {}
    for hits in per_domain.values():
        for c in sorted(hits, key=lambda c: c.score, reverse=True)[:min_per_domain]:
            picked.setdefault(c.id, c)
    rest = sorted(
        (c for hits in per_domain.values() for c in hits if c.id not in picked),
        key=lambda c: c.score, reverse=True,
    )
    for c in rest:
        if len(picked) >= final_k:
            break
        picked[c.id] = c
    return sorted(picked.values(), key=lambda c: c.score, reverse=True)


def retrieve(question: str, domains: list[str], client=None, embedder=None) -> list[Chunk]:
    if client is None:
        from app.clients import get_qdrant
        client = get_qdrant()
    if embedder is None:
        from app.clients import get_embedder
        embedder = get_embedder()
    qvec = [float(x) for x in next(iter(embedder.embed([question])))]  # embed once
    per_domain = {d: search_domain(client, qvec, d) for d in domains}
    return merge(per_domain)
