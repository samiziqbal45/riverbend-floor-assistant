"""Trace how a question turns into retrieved chunks, step by step, using the real functions.

    python scripts/trace_retrieval.py "What inspection checks should be done for bottles?"
    python scripts/trace_retrieval.py "..." --scores quality=0.94,maintenance=0.04,safety=0.02   # skip the router call

Embedded Qdrant allows one process at a time. If the app is running, point this at a copy:
    QDRANT_PATH=<copy of qdrant_data> python scripts/trace_retrieval.py "..."
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from app import retrieve as R  # noqa: E402
from app.clients import get_embedder, get_qdrant  # noqa: E402


def line(c: R.Chunk, tag: str = "") -> str:
    return f"{c.score:.4f}  {c.domain:<11} {c.doc[:40]:<40} > {c.section[:34]:<34} p.{c.page_start}  {tag}"


def preview(c: R.Chunk, n: int = 150) -> str:
    return " ".join(c.text.split())[:n]


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--scores", help="skip the router: e.g. quality=0.94,maintenance=0.04,safety=0.02")
    ap.add_argument("--threshold", type=float, default=config.SINGLE_DOMAIN_THRESHOLD)
    args = ap.parse_args()
    q = args.question

    print(f"QUESTION: {q}\n")
    print("== 1. ROUTE ==")
    if args.scores:
        scores = {k: float(v) for k, v in (p.split("=") for p in args.scores.split(","))}
        route = SimpleNamespace(scores=scores, ranked=sorted(scores, key=scores.get, reverse=True))
        print("(scores supplied by hand, router not called)")
    else:
        from app.router import route as do_route
        route = do_route(q)
        print(f"reasoning: {route.reasoning}")
    s = sorted(route.scores.items(), key=lambda kv: -kv[1])
    print("scores:", "  ".join(f"{d}={v:.3f}" for d, v in s), f"| margin={s[0][1] - s[1][1]:.3f}")

    domains = R.choose_domains(route, args.threshold)
    print(f"\n== 2. CHOOSE DOMAINS == threshold={args.threshold}")
    print(f"-> {domains}  ({'single-domain: top score >= threshold' if len(domains) == 1 else 'top-2 by default; no threshold, or top score below it'})")

    client, emb = get_qdrant(), get_embedder()
    qvec = [float(x) for x in next(iter(emb.embed([q])))]

    print(f"\n== 3. FILTERED SEARCH per domain (K_PER_DOMAIN={config.K_PER_DOMAIN}) ==")
    per: dict[str, list[R.Chunk]] = {}
    for d in domains:
        per[d] = R.search_domain(client, qvec, d)
        print(f"\n[{d}] filter domain=='{d}'")
        for i, c in enumerate(per[d], 1):
            print(f"  #{i} {line(c)}")

    print(f"\n== 4. MERGE (MIN_PER_DOMAIN={config.MIN_PER_DOMAIN}, FINAL_K={config.FINAL_K}) ==")
    floor: dict[str, str] = {}
    for d, hits in per.items():
        for i, c in enumerate(sorted(hits, key=lambda c: c.score, reverse=True)[: config.MIN_PER_DOMAIN], 1):
            floor.setdefault(c.id, f"FLOOR (top {i} of {d}, kept regardless of score)")
    final = R.merge(per)
    kept_ids = {c.id for c in final}
    rest = sorted((c for h in per.values() for c in h if c.id not in floor), key=lambda c: -c.score)
    print("step A: floor = the top", config.MIN_PER_DOMAIN, "of EACH searched domain,", len(floor), "chunks")
    print(f"step B: fill the remaining {max(0, config.FINAL_K - len(floor))} slot(s) with the best-scoring leftovers from either domain")
    print("\nFINAL CONTEXT sent to the LLM (sorted by score):")
    for i, c in enumerate(final, 1):
        print(f"  [{i}] {line(c, floor.get(c.id, 'FILL (best remaining by score)'))}")
    left = [c for c in rest if c.id not in kept_ids]
    if left:
        print("considered but cut:", "; ".join(f"{c.domain}:{c.section[:24]} {c.score:.3f}" for c in left))

    other = [c for c in final if c.domain != route.ranked[0]]
    if other:
        print(f"\n== 5. WHAT THE {route.ranked[1].upper()} CHUNKS ACTUALLY SAY (secondary-domain chunks in the context) ==")
        for c in other:
            print(f"  {c.score:.4f} {c.doc[:38]} > {c.section[:30]}\n      \"{preview(c)}...\"")

    print("\n== 6. COUNTERFACTUALS ==")
    solo = R.merge({route.ranked[0]: R.search_domain(client, qvec, route.ranked[0])})
    print(f"(a) single-domain [{route.ranked[0]}] only -> {len(solo)} chunks, lowest score {min(c.score for c in solo):.4f}, "
          f"all {route.ranked[0]}: {all(c.domain == route.ranked[0] for c in solo)}")
    res = client.query_points(config.COLLECTION, query=qvec, limit=config.FINAL_K, with_payload=True)
    glob = [R.Chunk.from_point(p) for p in res.points]
    print(f"(b) plain UNFILTERED top-{config.FINAL_K} over the whole index (no routing, no floor):")
    for i, c in enumerate(glob, 1):
        print(f"  #{i} {line(c)}")
    in_glob = {c.id for c in glob}
    forced = [c for c in final if c.id not in in_glob]
    print(f"\nchunks in the final context that an unfiltered search would NOT have returned: {len(forced)}")
    for c in forced:
        print(f"  - {line(c, floor.get(c.id, 'FILL'))}")
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
