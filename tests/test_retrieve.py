"""Retrieval tests with a temp Qdrant and hand-made vectors (no model)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

import config
from app import retrieve as R


def vec(angle_deg: float) -> list[float]:
    """Unit vector in the first two dims; cosine to vec(0) is cos(angle)."""
    v = np.zeros(config.EMBED_DIM)
    a = np.deg2rad(angle_deg)
    v[0], v[1] = np.cos(a), np.sin(a)
    return v.tolist()


def payload(domain, n):
    return {"text": f"{domain} chunk {n}", "domain": domain, "doc": f"{domain} doc", "section": "S",
            "page_start": 1, "page_end": 1, "revision": "A", "effective_date": "2025-01-01",
            "owner": "Owner", "chunk_index": n, "source_path": f"{domain}/x.pdf"}


@pytest.fixture
def client(tmp_path):
    c = QdrantClient(path=str(tmp_path / "q"))
    c.create_collection(config.COLLECTION, vectors_config=VectorParams(size=config.EMBED_DIM, distance=Distance.COSINE))
    # safety chunks are all CLOSER to the query (angle ~0) than maintenance chunks
    pts = [PointStruct(id=i + 1, vector=vec(i * 2), payload=payload("safety", i)) for i in range(6)]
    pts += [PointStruct(id=100 + i, vector=vec(40 + i * 2), payload=payload("maintenance", i)) for i in range(6)]
    c.upsert(config.COLLECTION, points=pts)
    yield c
    c.close()


class FakeEmbedder:
    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        for _ in texts:
            yield np.array(vec(0))


@dataclass
class FakeRoute:
    scores: dict

    @property
    def ranked(self):
        return sorted(self.scores, key=lambda d: self.scores[d], reverse=True)


def test_floor_guarantees_secondary_domain(client):
    emb = FakeEmbedder()
    chunks = R.retrieve("q", ["safety", "maintenance"], client=client, embedder=emb)
    domains = [c.domain for c in chunks]
    assert len(chunks) == config.FINAL_K
    assert domains.count("maintenance") >= config.MIN_PER_DOMAIN  # despite lower raw scores
    assert "safety" in domains
    assert emb.calls == 1  # query embedded once
    assert [c.score for c in chunks] == sorted((c.score for c in chunks), reverse=True)


def test_fill_is_by_raw_score(client):
    chunks = R.retrieve("q", ["safety", "maintenance"], client=client, embedder=FakeEmbedder())
    # 2 maintenance (floor) + 4 safety (best remaining by score)
    assert sum(c.domain == "safety" for c in chunks) == 4


def test_dedupe():
    c = R.Chunk("a", 0.9, "t", "safety", "d", "s", 1, 1, "A", "2025-01-01", "o")
    d = R.Chunk("a", 0.9, "t", "safety", "d", "s", 1, 1, "A", "2025-01-01", "o")
    out = R.merge({"safety": [c], "maintenance": [d]})
    assert [x.id for x in out] == ["a"]


def test_empty_domain_filled_from_other(client):
    chunks = R.retrieve("q", ["safety", "quality"], client=client, embedder=FakeEmbedder())  # quality has none
    assert len(chunks) == config.K_PER_DOMAIN  # all 5 safety hits are used, capped by K_PER_DOMAIN
    assert {c.domain for c in chunks} == {"safety"}


def test_all_domains_empty(client):
    assert R.retrieve("q", ["quality"], client=client, embedder=FakeEmbedder()) == []


def test_chunk_has_payload_fields(client):
    c = R.retrieve("q", ["safety"], client=client, embedder=FakeEmbedder())[0]
    assert c.domain == "safety" and c.owner == "Owner" and c.revision == "A" and c.id and c.score > 0.9


def test_choose_domains_default_top2():
    r = FakeRoute({"safety": 0.2, "maintenance": 0.5, "quality": 0.3})
    assert R.choose_domains(r, threshold=None) == ["maintenance", "quality"]


def test_choose_domains_threshold():
    r = FakeRoute({"safety": 0.9, "maintenance": 0.06, "quality": 0.04})
    assert R.choose_domains(r, threshold=0.85) == ["safety"]
    assert R.choose_domains(r, threshold=0.95) == ["safety", "maintenance"]
    mixed = FakeRoute({"safety": 0.5, "maintenance": 0.4, "quality": 0.1})
    assert R.choose_domains(mixed, threshold=0.85) == ["safety", "maintenance"]
