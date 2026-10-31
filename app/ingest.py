"""PDF -> sections -> chunks -> embeddings -> Qdrant.

CLI:
    python -m app.ingest [--reset]
    python -m app.ingest --inspect <pdf>
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import uuid
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

import config

HEADER_FOOTER_FRAC = 0.06
REPEAT_FRAC = 0.5
FONT_HEADING_RATIO = 1.12
BOLD_HEADING_MAX_CHARS = 80
MIN_SECTION_WORDS = 8  # drops front-matter fragments such as the bare title


class IngestError(RuntimeError):
    pass


# ---------- data ----------

@dataclass
class Element:
    """One paragraph-level unit on a page (text block or rendered table)."""
    page: int
    y0: float
    text: str
    kind: str = "text"  # "text" | "table"
    size: float = 0.0
    bold: bool = False
    nlines: int = 1


@dataclass
class Section:
    path: str
    level: int
    paras: list[tuple[str, int]] = field(default_factory=list)  # (text, page)


@dataclass
class DocChunk:
    section: str
    text: str
    page_start: int
    page_end: int


@dataclass
class Extraction:
    sections: list[Section]
    strategy: str  # "toc" | "font"


@dataclass
class PdfEntry:
    path: Path
    source_path: str  # "<domain>/<file>.pdf"
    meta: dict


# ---------- manifest ----------

def load_manifest(path: Path = config.MANIFEST_PATH) -> dict:
    if not Path(path).exists():
        raise IngestError(f"manifest not found: {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def discover_pdfs(corpus_dir: Path, manifest: dict, domains: list[str]) -> list[PdfEntry]:
    """Pair every corpus/<domain>/*.pdf with its manifest entry; fail loudly on mismatches."""
    corpus_dir = Path(corpus_dir)
    entries: list[PdfEntry] = []
    seen: set[str] = set()
    for sub in sorted(p for p in corpus_dir.iterdir() if p.is_dir() and p.name != "source"):
        pdfs = sorted(sub.glob("*.pdf"))
        if not pdfs:
            continue
        if sub.name not in domains:
            raise IngestError(f"directory '{sub.name}' has PDFs but is not a configured domain {domains}")
        for pdf in pdfs:
            meta = manifest.get(pdf.name)
            if meta is None:
                raise IngestError(f"{sub.name}/{pdf.name} has no entry in the manifest")
            if meta.get("domain") not in domains:
                raise IngestError(f"{pdf.name}: manifest domain {meta.get('domain')!r} not in {domains}")
            if meta["domain"] != sub.name:
                raise IngestError(f"{pdf.name}: in folder '{sub.name}' but manifest says '{meta['domain']}'")
            missing = {"doc", "domain", "revision", "effective_date", "owner"} - set(meta)
            if missing:
                raise IngestError(f"{pdf.name}: manifest entry missing {sorted(missing)}")
            seen.add(pdf.name)
            entries.append(PdfEntry(pdf, f"{sub.name}/{pdf.name}", meta))
    for name in sorted(set(manifest) - seen):
        print(f"WARNING: manifest entry '{name}' has no PDF", file=sys.stderr)
    return entries


# ---------- extraction ----------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _is_bold(span: dict) -> bool:
    return bool(span["flags"] & 16) or "bold" in span["font"].lower()


def _block_element(block: dict, page_no: int) -> Element | None:
    lines = []
    for ln in block["lines"]:
        t = "".join(s["text"] for s in ln["spans"]).strip()
        if t:
            lines.append(t)
    spans = [s for ln in block["lines"] for s in ln["spans"] if s["text"].strip()]
    if not lines or not spans:
        return None
    if len(lines) > 1 and re.fullmatch(r"\d+", lines[0]):  # numbered-list marker
        lines[0] = lines[0] + "."
    text = " ".join(lines)
    return Element(
        page=page_no, y0=block["bbox"][1], text=text,
        size=max(s["size"] for s in spans),
        bold=all(_is_bold(s) for s in spans), nlines=len(lines),
    )


def _table_text(page: pymupdf.Page, table) -> str:
    rows = []
    for row in table.rows:
        cells = []
        for c in row.cells:
            if c is None:
                continue
            r = pymupdf.Rect(c) + (1, 1, -1, -1)
            cells.append(_ws(page.get_text("text", clip=r)))
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _page_elements(page: pymupdf.Page, page_no: int) -> list[Element]:
    h = page.rect.height
    tables = []
    try:
        tables = list(page.find_tables().tables)
    except Exception:  # find_tables is best-effort
        tables = []
    boxes = [pymupdf.Rect(t.bbox) for t in tables]

    elems: list[Element] = []
    for b in page.get_text("dict", sort=True)["blocks"]:
        if b["type"] != 0:
            continue
        x0, y0, x1, y1 = b["bbox"]
        if y1 < h * HEADER_FOOTER_FRAC or y0 > h * (1 - HEADER_FOOTER_FRAC):
            continue
        center = pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2)
        if any(center in box for box in boxes):
            continue
        e = _block_element(b, page_no)
        if e:
            elems.append(e)
    for t, box in zip(tables, boxes):
        if box.y1 < h * HEADER_FOOTER_FRAC or box.y0 > h * (1 - HEADER_FOOTER_FRAC):
            continue
        txt = _table_text(page, t)
        if txt:
            elems.append(Element(page_no, box.y0, txt, kind="table"))
    return sorted(elems, key=lambda e: e.y0)


def _strip_repeating(pages: list[list[Element]]) -> list[list[Element]]:
    n = len(pages)
    if n < 2:
        return pages
    counts: Counter[str] = Counter()
    for elems in pages:
        for key in {_norm(e.text) for e in elems if e.kind == "text" and e.nlines <= 2}:
            counts[key] += 1
    repeated = {k for k, c in counts.items() if c / n > REPEAT_FRAC}
    return [[e for e in elems if not (e.kind == "text" and e.nlines <= 2 and _norm(e.text) in repeated)]
            for elems in pages]


def _weighted_median_size(elems: list[Element]) -> float:
    sizes: list[float] = []
    for e in elems:
        if e.kind == "text":
            sizes.extend([round(e.size, 1)] * len(e.text))
    return statistics.median(sizes) if sizes else 10.0


def _heading_levels_toc(elems_by_page: list[list[Element]], toc: list) -> dict[int, int]:
    """Map id(element) -> level using the PDF outline."""
    levels: dict[int, int] = {}
    used: set[int] = set()
    # Outline pages in real PDFs are sometimes off by one, so match on the
    # listed page first, then the neighbours, and take entries in outline order.
    for pi, elems in enumerate(elems_by_page, start=1):
        for e in elems:
            if e.kind != "text":
                continue
            matches = [i for i, (_, title, pg) in enumerate(toc)
                       if i not in used and abs(pg - pi) <= 1 and _norm(title) == _norm(e.text)]
            if matches:
                i = min(matches, key=lambda k: (abs(toc[k][2] - pi), k))
                used.add(i)
                levels[id(e)] = toc[i][0]
    for i, (lvl, title, pg) in enumerate(toc):
        if i not in used:
            print(f"WARNING: outline entry '{title}' (p.{pg}) not found in page text", file=sys.stderr)
    return levels


def _heading_levels_font(elems_by_page: list[list[Element]]) -> dict[int, int]:
    all_elems = [e for p in elems_by_page for e in p]
    body = _weighted_median_size(all_elems)
    heads = []
    for e in all_elems:
        if e.kind != "text":
            continue
        big = e.size > body * FONT_HEADING_RATIO
        bold_short = e.bold and e.nlines == 1 and len(e.text) <= BOLD_HEADING_MAX_CHARS
        if big or bold_short:
            heads.append(e)
    sizes = sorted({round(e.size, 1) for e in heads if e.size > body * FONT_HEADING_RATIO}, reverse=True)
    # a single, largest heading on page 1 is the document title, not a section
    if sizes:
        top = [e for e in heads if round(e.size, 1) == sizes[0]]
        if len(top) == 1 and top[0].page == 1 and len(sizes) > 1:
            heads.remove(top[0])
            sizes = sizes[1:]
    levels: dict[int, int] = {}
    for e in heads:
        s = round(e.size, 1)
        levels[id(e)] = sizes.index(s) + 1 if s in sizes else len(sizes) + 1
    return levels


def extract_document(pdf_path: Path) -> Extraction:
    doc = pymupdf.open(str(pdf_path))
    try:
        pages = [_page_elements(doc[i], i + 1) for i in range(len(doc))]
        pages = _strip_repeating(pages)
        toc = doc.get_toc()
    finally:
        doc.close()

    if toc:
        strategy, levels = "toc", _heading_levels_toc(pages, toc)
    else:
        strategy, levels = "font", _heading_levels_font(pages)

    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    cur = Section("Front matter", 0)
    sections.append(cur)
    for elems in pages:
        for e in elems:
            lvl = levels.get(id(e))
            if lvl is not None:
                while stack and stack[-1][0] >= lvl:
                    stack.pop()
                stack.append((lvl, e.text.strip()))
                cur = Section(" > ".join(t for _, t in stack), lvl)
                sections.append(cur)
            else:
                cur.paras.append((e.text, e.page))
    sections = [s for s in sections if sum(len(t.split()) for t, _ in s.paras) >= MIN_SECTION_WORDS]
    return Extraction(sections, strategy)


# ---------- chunking ----------

def _tokens(text: str) -> float:
    return len(text.split()) * 1.3


def _split_oversize(text: str, page: int, max_tokens: int) -> list[tuple[str, int]]:
    """Split a single paragraph/table that alone exceeds max_tokens."""
    if _tokens(text) <= max_tokens:
        return [(text, page)]
    lines = text.split("\n")
    if len(lines) > 1:
        header = lines[0] if " | " in lines[0] else None
        body = lines[1:] if header else lines
        out, cur = [], []
        for ln in body:
            trial = ([header] if header else []) + cur + [ln]
            if cur and _tokens("\n".join(trial)) > max_tokens:
                out.append(("\n".join(([header] if header else []) + cur), page))
                cur = []
            cur.append(ln)
        if cur:
            out.append(("\n".join(([header] if header else []) + cur), page))
        return out
    words = text.split()
    step = max(1, int(max_tokens / 1.3))
    return [(" ".join(words[i:i + step]), page) for i in range(0, len(words), step)]


def chunk_sections(sections: list[Section], max_tokens: int = config.MAX_CHUNK_TOKENS) -> list[DocChunk]:
    chunks: list[DocChunk] = []
    for sec in sections:
        paras: list[tuple[str, int]] = []
        for text, page in sec.paras:
            paras.extend(_split_oversize(text, page, max_tokens))
        cur: list[tuple[str, int]] = []
        cur_tok = 0.0

        def emit():
            chunks.append(DocChunk(sec.path, "\n\n".join(t for t, _ in cur),
                                   min(p for _, p in cur), max(p for _, p in cur)))

        for para in paras:
            t = _tokens(para[0])
            if cur and cur_tok + t > max_tokens:
                emit()
                last = cur[-1]
                if _tokens(last[0]) + t <= max_tokens:  # one-paragraph overlap
                    cur, cur_tok = [last], _tokens(last[0])
                else:
                    cur, cur_tok = [], 0.0
            cur.append(para)
            cur_tok += t
        if cur:
            emit()
    return chunks


# ---------- indexing ----------

def point_id(source_path: str, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_path}#{chunk_index}"))


def embed_text(doc: str, section: str, text: str) -> str:
    return f"{doc} — {section}\n{text}"


def build_payloads(entry: PdfEntry, chunks: list[DocChunk]) -> list[dict]:
    m = entry.meta
    return [
        {
            "text": c.text, "domain": m["domain"], "doc": m["doc"], "section": c.section,
            "page_start": c.page_start, "page_end": c.page_end, "revision": m["revision"],
            "effective_date": m["effective_date"], "owner": m["owner"],
            "chunk_index": i, "source_path": entry.source_path,
        }
        for i, c in enumerate(chunks)
    ]


def ensure_collection(client, reset: bool = False) -> None:
    from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

    if reset and client.collection_exists(config.COLLECTION):
        client.delete_collection(config.COLLECTION)
    if not client.collection_exists(config.COLLECTION):
        client.create_collection(
            config.COLLECTION,
            vectors_config=VectorParams(size=config.EMBED_DIM, distance=Distance.COSINE),
        )
    with warnings.catch_warnings():  # local mode ignores payload indexes (server mode uses them)
        warnings.simplefilter("ignore", UserWarning)
        for field_name in ("domain", "source_path"):
            client.create_payload_index(config.COLLECTION, field_name, PayloadSchemaType.KEYWORD)


def _upsert_pdf(client, embedder, entry: PdfEntry, payloads: list[dict]) -> None:
    from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

    client.delete(
        config.COLLECTION,
        points_selector=Filter(must=[FieldCondition(key="source_path", match=MatchValue(value=entry.source_path))]),
    )
    texts = [embed_text(p["doc"], p["section"], p["text"]) for p in payloads]
    vectors = list(embedder.embed(texts, batch_size=32))
    client.upsert(
        config.COLLECTION,
        points=[
            PointStruct(id=point_id(p["source_path"], p["chunk_index"]), vector=list(map(float, v)), payload=p)
            for p, v in zip(payloads, vectors)
        ],
    )


def ingest_all(client, embedder, corpus_dir: Path = config.CORPUS_DIR,
               manifest_path: Path = config.MANIFEST_PATH, reset: bool = False) -> list[dict]:
    manifest = load_manifest(manifest_path)
    entries = discover_pdfs(corpus_dir, manifest, config.DOMAINS)
    ensure_collection(client, reset=reset)
    rows = []
    for entry in entries:
        ex = extract_document(entry.path)
        chunks = chunk_sections(ex.sections)
        payloads = build_payloads(entry, chunks)
        if payloads:
            _upsert_pdf(client, embedder, entry, payloads)
        rows.append({"domain": entry.meta["domain"], "doc": entry.meta["doc"],
                     "chunks": len(payloads), "strategy": ex.strategy})
    return rows


def print_summary(rows: list[dict], total_points: int | None = None) -> None:
    print(f"\n{'domain':<12}{'doc':<54}{'chunks':>7}  strategy")
    print("-" * 84)
    for r in rows:
        print(f"{r['domain']:<12}{r['doc'][:52]:<54}{r['chunks']:>7}  {r['strategy']}")
    print("-" * 84)
    per_domain = Counter()
    for r in rows:
        per_domain[r["domain"]] += r["chunks"]
    for d in config.DOMAINS:
        print(f"{d:<12}{'(total chunks)':<54}{per_domain[d]:>7}")
    if total_points is not None:
        print(f"\npoints in collection '{config.COLLECTION}': {total_points}")


def inspect_pdf(pdf: Path) -> None:
    ex = extract_document(pdf)
    chunks = chunk_sections(ex.sections)
    print(f"{pdf}  strategy={ex.strategy}  sections={len(ex.sections)}  chunks={len(chunks)}")
    for i, c in enumerate(chunks):
        pages = f"p.{c.page_start}" if c.page_start == c.page_end else f"pp.{c.page_start}-{c.page_end}"
        print(f"\n=== [{i}] {c.section}  ({pages}, ~{int(_tokens(c.text))} tok) ===")
        print(c.text)


# ---------- CLI ----------

def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="python -m app.ingest")
    ap.add_argument("--reset", action="store_true", help="drop and recreate the collection")
    ap.add_argument("--inspect", metavar="PDF", help="print extracted sections for one PDF and exit")
    args = ap.parse_args(argv)

    if args.inspect:
        inspect_pdf(Path(args.inspect))
        return 0

    from app.clients import get_embedder, get_qdrant

    client = get_qdrant()
    try:
        rows = ingest_all(client, get_embedder(), reset=args.reset)
        total = client.count(config.COLLECTION, exact=True).count
    except IngestError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    finally:
        client.close()
    print_summary(rows, total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
