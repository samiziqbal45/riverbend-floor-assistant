"""Router -> dual filtered retrieval -> single grounded generation.

pip install qdrant-client fastembed anthropic
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

import anthropic
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams,
)

DOMAINS = ["safety", "maintenance", "quality"]
COLLECTION = "plant_docs"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # 384-dim, runs locally on CPU
LLM_MODEL = "claude-sonnet-5-5"

K_PER_DOMAIN = 5      # candidates pulled from each routed domain
FINAL_K = 6           # chunks passed to generation
MIN_PER_DOMAIN = 2    # keep the secondary domain from being crowded out

embedder = TextEmbedding(EMBED_MODEL)
qdrant = QdrantClient(path="./qdrant_data")  # or ":memory:"; swap for url=... later
llm = anthropic.Anthropic()


# ---------- ingestion ----------

def ensure_collection() -> None:
    if not qdrant.collection_exists(COLLECTION):
        qdrant.create_collection(
            COLLECTION, vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )


def ingest(chunks: list[dict]) -> None:
    """chunks: [{"text", "domain", "doc", "section"}, ...]"""
    ensure_collection()
    vectors = list(embedder.embed([c["text"] for c in chunks]))
    qdrant.upsert(
        COLLECTION,
        points=[
            PointStruct(id=str(uuid.uuid4()), vector=v.tolist(), payload=c)
            for c, v in zip(chunks, vectors)
        ],
    )


# ---------- routing ----------

ROUTE_TOOL = {
    "name": "route",
    "description": "Score how likely each documentation domain answers the question.",
    "input_schema": {
        "type": "object",
        "properties": {
            **{d: {"type": "number", "minimum": 0, "maximum": 1} for d in DOMAINS},
            "reasoning": {"type": "string"},
        },
        "required": [*DOMAINS, "reasoning"],
    },
}

ROUTER_PROMPT = """You route manufacturing floor questions to documentation.
Domains:
- safety: hazard controls, lockout/tagout, PPE, emergency and incident procedures
- maintenance: equipment operation, repair, troubleshooting, PM schedules, parts
- quality: specs, tolerances, inspection methods, defect handling, nonconformance
Give each domain a probability (they should sum to ~1). Score by relevance only.

Question: {q}"""


@dataclass
class Route:
    scores: dict[str, float]
    reasoning: str

    @property
    def ranked(self) -> list[str]:
        return sorted(self.scores, key=self.scores.get, reverse=True)

    @property
    def primary(self) -> str:
        return self.ranked[0]


def route(question: str) -> Route:
    resp = llm.messages.create(
        model=LLM_MODEL,
        max_tokens=300,
        tools=[ROUTE_TOOL],
        tool_choice={"type": "tool", "name": "route"},
        messages=[{"role": "user", "content": ROUTER_PROMPT.format(q=question)}],
    )
    out = next(b.input for b in resp.content if b.type == "tool_use")
    total = sum(out[d] for d in DOMAINS) or 1.0
    return Route({d: out[d] / total for d in DOMAINS}, out["reasoning"])


# ---------- retrieval ----------

def search_domain(qvec: list[float], domain: str, k: int) -> list[dict]:
    hits = qdrant.query_points(
        COLLECTION,
        query=qvec,
        query_filter=Filter(must=[FieldCondition(key="domain", match=MatchValue(value=domain))]),
        limit=k,
    ).points
    return [{**h.payload, "id": h.id, "score": h.score} for h in hits]


def retrieve(question: str, domains: list[str]) -> list[dict]:
    qvec = next(embedder.embed([question])).tolist()
    per_domain = {d: search_domain(qvec, d, K_PER_DOMAIN) for d in domains}

    # Guarantee each routed domain a floor, then fill by raw similarity.
    picked: dict[str, dict] = {}
    for hits in per_domain.values():
        for h in hits[:MIN_PER_DOMAIN]:
            picked[h["id"]] = h
    rest = sorted(
        (h for hits in per_domain.values() for h in hits if h["id"] not in picked),
        key=lambda h: h["score"], reverse=True,
    )
    for h in rest[: max(0, FINAL_K - len(picked))]:
        picked[h["id"]] = h
    return sorted(picked.values(), key=lambda h: h["score"], reverse=True)


# ---------- generation ----------

ANSWER_PROMPT = """You answer questions from plant floor supervisors using ONLY the
documentation excerpts below. Cite excerpts inline as [1], [2], etc.
If the excerpts don't answer the question, say it isn't covered in the
documentation and name what document would be needed. Do not guess.
Be concise and actionable; supervisors are on the floor.

Excerpts:
{context}

Question: {q}"""


def generate(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i}] ({c['domain']} | {c['doc']} | {c['section']})\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )
    resp = llm.messages.create(
        model=LLM_MODEL,
        max_tokens=800,
        messages=[{"role": "user", "content": ANSWER_PROMPT.format(context=context, q=question)}],
    )
    return resp.content[0].text


# ---------- entry point ----------

def ask(question: str) -> dict:
    r = route(question)
    domains = r.ranked[:2]
    chunks = retrieve(question, domains)
    return {
        "primary_domain": r.primary,
        "confidence": round(r.scores[r.primary], 2),
        "searched": domains,
        "scores": r.scores,
        "router_reasoning": r.reasoning,
        "answer": generate(question, chunks),
        "sources": chunks,
    }
