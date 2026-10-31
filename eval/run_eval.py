"""Golden-set evaluation.

    python -m eval.run_eval [--golden PATH] [--skip-generation] [--limit N]

Each question is routed ONCE; that Route is reused for the main run and for every
threshold in the sweep (the sweep re-runs retrieval only).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402

EVAL_DIR = ROOT / "eval"
GOLDEN = EVAL_DIR / "golden.jsonl"
FEWSHOT = EVAL_DIR / "fewshot.jsonl"
RESULTS = EVAL_DIR / "results"
SWEEP = [None, 0.6, 0.7, 0.8, 0.9]
CATEGORIES = ["clean", "ambiguous", "conflict", "not_covered"]
REQUIRED = {"id", "question", "acceptable_domains", "expected_docs",
            "expect_conflict", "expect_not_covered", "category"}


@dataclass
class Outcome:
    item: dict
    ranked: list[str] = field(default_factory=list)
    searched: list[str] = field(default_factory=list)
    docs: list[str] = field(default_factory=list)  # doc titles of retrieved chunks
    conflict: bool | None = None  # None when generation was skipped / failed
    not_covered: bool | None = None
    latency: dict = field(default_factory=dict)
    error: str | None = None


# ---------- loading & validation ----------

def load_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_golden(golden: list[dict]) -> None:
    ids = set()
    for g in golden:
        missing = REQUIRED - set(g)
        if missing:
            raise SystemExit(f"golden item {g.get('id')!r} missing fields: {sorted(missing)}")
        if g["category"] not in CATEGORIES:
            raise SystemExit(f"golden item {g['id']!r}: unknown category {g['category']!r}")
        if g["id"] in ids:
            raise SystemExit(f"duplicate golden id {g['id']!r}")
        ids.add(g["id"])


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def assert_disjoint(golden: list[dict], fewshot: list[dict]) -> None:
    """The router's few-shot examples must not appear in the golden set."""
    overlap = {_norm(g["question"]) for g in golden} & {_norm(f["question"]) for f in fewshot}
    if overlap:
        raise SystemExit(f"golden/fewshot question overlap: {sorted(overlap)}")


# ---------- metrics (pure) ----------

def top1_ok(o: Outcome) -> bool:
    return bool(o.ranked) and o.ranked[0] in o.item["acceptable_domains"]


def top2_ok(o: Outcome) -> bool:
    return set(o.item["acceptable_domains"]) <= set(o.searched)


def recall_of(expected: list[str], docs: list[str]) -> float | None:
    """Fraction of expected_docs found (case-insensitive substring) among retrieved doc titles."""
    if not expected:
        return None
    got = [d.lower() for d in docs]
    return sum(any(e.lower() in g for g in got) for e in expected) / len(expected)


def recall_ok(o: Outcome) -> float | None:
    return recall_of(o.item["expected_docs"], o.docs)


def prf(pairs: list[tuple[bool, bool]]) -> dict:
    """pairs of (expected, predicted)."""
    tp = sum(e and p for e, p in pairs)
    fp = sum((not e) and p for e, p in pairs)
    fn = sum(e and (not p) for e, p in pairs)
    return {"precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None, "tp": tp, "fp": fp, "fn": fn}


def percentile(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    return s[min(len(s) - 1, round(p / 100 * (len(s) - 1)))]


def _mean(xs):
    xs = list(xs)
    return statistics.mean(xs) if xs else None


def summarize(outs: list[Outcome]) -> dict:
    gen = [o for o in outs if o.conflict is not None]
    expect_nc = [o for o in gen if o.item["expect_not_covered"]]
    other = [o for o in gen if not o.item["expect_not_covered"]]
    return {
        "n": len(outs),
        "top1": _mean(top1_ok(o) for o in outs),
        "top2": _mean(top2_ok(o) for o in outs),
        "recall": _mean(r for o in outs if (r := recall_ok(o)) is not None),
        "conflict": prf([(o.item["expect_conflict"], o.conflict) for o in gen]) if gen else None,
        "refusal": _mean(o.not_covered for o in expect_nc),  # refused when it should
        "false_refusal": _mean(o.not_covered for o in other),  # refused when it shouldn't
    }


def fmt(x) -> str:
    return "n/a" if x is None else f"{x:.2f}"


# ---------- running ----------

def _default_router(q):
    from app import router
    return router.route(q)


def _default_retriever(q, domains):
    from app import retrieve
    return retrieve.retrieve(q, domains)


def run_item(item: dict, route, *, retriever, generator, skip_generation: bool) -> Outcome:
    """Evaluate one question given its (cached) Route."""
    from app import pipeline, retrieve

    out = Outcome(item, ranked=list(route.ranked))
    if skip_generation:
        out.searched = retrieve.choose_domains(route, None)
        t = time.perf_counter()
        chunks = retriever(item["question"], out.searched)
        out.latency["retrieve"] = (time.perf_counter() - t) * 1000
        out.docs = [c.doc for c in chunks]
        return out
    r = pipeline.ask(item["question"], retriever=retriever, generator=generator, route_result=route,
                     single_domain_threshold=None, write_log=False)
    out.searched, out.docs = r.searched_domains, [c.doc for c in r.chunks]
    out.conflict, out.not_covered = bool(r.conflicts), r.not_covered
    out.latency.update({k: v for k, v in r.latency_ms.items() if k != "route"})
    return out


def evaluate(golden: list[dict], *, skip_generation: bool = False, router=None, retriever=None,
             generator=None, verbose: bool = True) -> tuple[list[Outcome], dict]:
    """Route each question once (cached in `routes`), then run retrieval/generation."""
    from app.generate import generate

    router = router or _default_router
    retriever = retriever or _default_retriever
    generator = generator or generate
    routes: dict = {}
    outs: list[Outcome] = []
    for item in golden:
        try:
            t = time.perf_counter()
            route = router(item["question"])
            route_ms = (time.perf_counter() - t) * 1000
            routes[item["id"]] = route
            o = run_item(item, route, retriever=retriever, generator=generator, skip_generation=skip_generation)
            o.latency["route"] = route_ms
        except Exception as e:  # one bad call must not sink the whole run
            o = Outcome(item, error=f"{type(e).__name__}: {e}")
        outs.append(o)
        if verbose:
            tag = f"ERROR {o.error}" if o.error else f"top={o.ranked[0]:<11} searched={o.searched}"
            print(f"{item['id']:<4} {item['category']:<12} {tag}")
    return outs, routes


def sweep(golden: list[dict], routes: dict, retriever=None) -> list[dict]:
    """For each threshold, re-choose domains from the cached routes and re-run retrieval only."""
    from app import retrieve

    retriever = retriever or _default_retriever
    cache: dict[tuple, list[str]] = {}
    rows = []
    for th in SWEEP:
        cov, single, rec = [], [], []
        for item in golden:
            route = routes.get(item["id"])
            if route is None:  # routing failed: counts as a miss
                cov.append(False)
                single.append(False)
                continue
            doms = retrieve.choose_domains(route, th)
            key = (item["question"], tuple(doms))
            if key not in cache:
                cache[key] = [c.doc for c in retriever(item["question"], doms)]
            cov.append(set(item["acceptable_domains"]) <= set(doms))
            single.append(len(doms) == 1)
            if (r := recall_of(item["expected_docs"], cache[key])) is not None:
                rec.append(r)
        rows.append({"threshold": th, "top2_coverage": statistics.mean(cov), "pct_single": statistics.mean(single),
                     "recall": statistics.mean(rec) if rec else None})
    return rows


# ---------- reporting ----------

def _th(t) -> str:
    return "off" if t is None else str(t)


def plot_sweep(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r["pct_single"] * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(xs, [r["top2_coverage"] for r in rows], marker="o", label="domain coverage")
    if all(r["recall"] is not None for r in rows):
        ax.plot(xs, [r["recall"] for r in rows], marker="s", linestyle="--", label="recall@K")
    for r, x in zip(rows, xs):
        ax.annotate(_th(r["threshold"]), (x, r["top2_coverage"]), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("% of queries using single-domain search")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title("SINGLE_DOMAIN_THRESHOLD sweep (labels = threshold)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def failures(outs: list[Outcome]) -> list[tuple[str, str, str, str]]:
    """(id, question, expected, actual) for every check that failed."""
    rows = []
    for o in outs:
        it = o.item
        if o.error:
            rows.append((it["id"], it["question"], "pipeline completes", o.error))
            continue
        checks = [
            (top1_ok(o), f"top-1 in {it['acceptable_domains']}", f"top-1 {o.ranked[0] if o.ranked else None}"),
            (top2_ok(o), f"searched {it['acceptable_domains']}", f"searched {o.searched}"),
            (recall_ok(o) in (None, 1.0), f"docs {it['expected_docs']}", f"docs {sorted(set(o.docs))}"),
        ]
        if o.conflict is not None:
            checks += [
                (o.conflict == it["expect_conflict"], f"conflict={it['expect_conflict']}", f"conflict={o.conflict}"),
                (o.not_covered == it["expect_not_covered"], f"not_covered={it['expect_not_covered']}",
                 f"not_covered={o.not_covered}"),
            ]
        rows += [(it["id"], it["question"], e, a) for ok, e, a in checks if not ok]
    return rows


def report(outs: list[Outcome], sweep_rows: list[dict], *, golden_path: str = "", skip_generation: bool = False) -> str:
    gen_cols = not skip_generation
    head = ["category", "n", "top-1", "top-2 cov.", "recall@K"]
    if gen_cols:
        head += ["conflict P", "conflict R", "refusal R", "false refusal"]
    lines = ["# Eval report", "",
             f"- golden: `{golden_path}` ({len(outs)} questions), generated {datetime.now():%Y-%m-%d %H:%M}",
             f"- mode: {'route + retrieve only (--skip-generation)' if skip_generation else 'full pipeline'}",
             f"- K_PER_DOMAIN={config.K_PER_DOMAIN}, FINAL_K={config.FINAL_K}, MIN_PER_DOMAIN={config.MIN_PER_DOMAIN}",
             f"- errors: {sum(bool(o.error) for o in outs)}", "",
             "## Summary", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for c in ["all", *CATEGORIES]:
        sub = outs if c == "all" else [o for o in outs if o.item["category"] == c]
        if not sub:
            continue
        s = summarize(sub)
        row = [c, str(s["n"]), fmt(s["top1"]), fmt(s["top2"]), fmt(s["recall"])]
        if gen_cols:
            cf = s["conflict"] or {}
            row += [fmt(cf.get("precision")), fmt(cf.get("recall")), fmt(s["refusal"]), fmt(s["false_refusal"])]
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## Latency (ms)", "", "| stage | p50 | p95 |", "|---|---|---|"]
    for stage in ("route", "retrieve", "generate"):
        v = [o.latency[stage] for o in outs if stage in o.latency]
        if v:
            lines.append(f"| {stage} | {percentile(v, 50):.0f} | {percentile(v, 95):.0f} |")
    lines += ["", "## Threshold sweep", "",
              "| SINGLE_DOMAIN_THRESHOLD | top-2 coverage | % single-domain | recall@K |", "|---|---|---|---|"]
    lines += [f"| {_th(r['threshold'])} | {r['top2_coverage']:.2f} | {r['pct_single'] * 100:.0f}% | {fmt(r['recall'])} |"
              for r in sweep_rows]
    lines += ["", "![threshold sweep](threshold_sweep.png)", "", "## Failures", ""]
    fails = failures(outs)
    if fails:
        lines += [f"- **{i}** \"{q}\" — expected: {e}; actual: {a}" for i, q, e, a in fails]
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def write_outputs(outs, sweep_rows, results_dir: Path, *, golden_path: str, skip_generation: bool) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_sweep(sweep_rows, results_dir / "threshold_sweep.png")
    path = results_dir / "report.md"
    path.write_text(report(outs, sweep_rows, golden_path=golden_path, skip_generation=skip_generation),
                    encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="python -m eval.run_eval")
    ap.add_argument("--golden", type=Path, default=GOLDEN, help="golden set (default eval/golden.jsonl)")
    ap.add_argument("--skip-generation", action="store_true", help="route + retrieve only (cheap)")
    ap.add_argument("--limit", type=int, help="only the first N questions")
    ap.add_argument("--results-dir", type=Path, default=RESULTS)
    args = ap.parse_args(argv)

    golden = load_jsonl(args.golden)
    validate_golden(golden)
    assert_disjoint(golden, load_jsonl(FEWSHOT))
    if args.limit:
        golden = golden[: args.limit]

    outs, routes = evaluate(golden, skip_generation=args.skip_generation)
    rows = sweep(golden, routes)
    path = write_outputs(outs, rows, args.results_dir, golden_path=str(args.golden),
                         skip_generation=args.skip_generation)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
