"""Ingestion tests. PDFs are generated on the fly with reportlab; embeddings are faked."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

import config
from app import ingest
from app.ingest import IngestError

BODY = ("This is ordinary body text for the fixture document and it contains enough "
        "words to count as a real section of content for chunking purposes.")


def make_pdf(path: Path, bookmarks: bool) -> Path:
    """2-page PDF: header/footer on each page, a title, three headings, and a section
    (Alpha) that continues onto page 2."""
    w, h = letter
    c = canvas.Canvas(str(path), pagesize=letter)

    def chrome(n: int):
        c.setFont("Helvetica", 8)
        c.drawString(50, h - 30, "Fixture Corp | Fixture Manual | CONTROLLED")
        c.drawString(50, 25, f"Rev A | Page {n} of 2")

    def heading(text: str, y: int, size: int, level: int, key: str):
        if bookmarks:
            c.bookmarkPage(key)
            c.addOutlineEntry(text, key, level=level)
        c.setFont("Helvetica-Bold", size)
        c.drawString(60, y, text)

    def body(y: int, n: int = 6):
        c.setFont("Helvetica", 10)
        for i in range(n):
            c.drawString(60, y - 14 * i, BODY[:100] if i % 2 else BODY[60:])

    chrome(1)
    c.setFont("Helvetica-Bold", 26)
    c.drawString(60, 700, "Fixture Manual")
    heading("Alpha Section", 660, 16, 0, "a")
    body(640)
    heading("Sub A", 520, 13, 1, "sa")
    body(500)
    c.showPage()
    chrome(2)
    body(700, 3)  # Alpha > Sub A continues here
    heading("Beta Section", 620, 16, 0, "b")
    body(600)
    c.save()
    return path


def make_table_pdf(path: Path, rows: int = 3) -> Path:
    styles = getSampleStyleSheet()
    data = [["Symptom", "Cause", "Fix"]] + [[f"Symptom {i}", f"Cause {i}", f"Fix {i}"] for i in range(rows)]
    t = Table(data)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black)]))
    SimpleDocTemplate(str(path), pagesize=letter).build(
        [Paragraph(" ".join(["Intro text for the table section."] * 3), styles["BodyText"]), t])
    return path


class FakeEmbedder:
    """Deterministic 384-dim vectors; no model download."""

    def embed(self, texts, batch_size=32):
        for t in texts:
            seed = int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).normal(size=config.EMBED_DIM)
            yield v / np.linalg.norm(v)


# ---------- manifest ----------

def write_corpus(root: Path, manifest: dict, files: dict[str, str]) -> Path:
    """files: {"safety/a.pdf": "bookmarks"|"plain"}"""
    corpus = root / "corpus"
    for rel, kind in files.items():
        (corpus / Path(rel).parent).mkdir(parents=True, exist_ok=True)
        make_pdf(corpus / rel, bookmarks=(kind == "bookmarks"))
    (corpus / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return corpus


META = {"doc": "Fixture Manual", "domain": "safety", "revision": "A",
        "effective_date": "2025-01-01", "owner": "EHS Lead"}


def test_manifest_lookup(tmp_path):
    corpus = write_corpus(tmp_path, {"a.pdf": META}, {"safety/a.pdf": "bookmarks"})
    entries = ingest.discover_pdfs(corpus, ingest.load_manifest(corpus / "manifest.json"), config.DOMAINS)
    assert [e.source_path for e in entries] == ["safety/a.pdf"]
    assert entries[0].meta["owner"] == "EHS Lead"


def test_pdf_without_manifest_entry_fails(tmp_path):
    corpus = write_corpus(tmp_path, {}, {"safety/a.pdf": "bookmarks"})
    with pytest.raises(IngestError, match="no entry in the manifest"):
        ingest.discover_pdfs(corpus, {}, config.DOMAINS)


def test_unknown_domain_fails(tmp_path):
    bad = {**META, "domain": "hr"}
    corpus = write_corpus(tmp_path, {"a.pdf": bad}, {"hr/a.pdf": "bookmarks"})
    with pytest.raises(IngestError, match="not a configured domain"):
        ingest.discover_pdfs(corpus, {"a.pdf": bad}, config.DOMAINS)


def test_domain_folder_mismatch_fails(tmp_path):
    corpus = write_corpus(tmp_path, {"a.pdf": {**META, "domain": "quality"}}, {"safety/a.pdf": "bookmarks"})
    with pytest.raises(IngestError, match="manifest says"):
        ingest.discover_pdfs(corpus, {"a.pdf": {**META, "domain": "quality"}}, config.DOMAINS)


def test_manifest_entry_without_pdf_warns(tmp_path, capsys):
    corpus = write_corpus(tmp_path, {"a.pdf": META, "ghost.pdf": META}, {"safety/a.pdf": "bookmarks"})
    ingest.discover_pdfs(corpus, ingest.load_manifest(corpus / "manifest.json"), config.DOMAINS)
    assert "ghost.pdf" in capsys.readouterr().err


# ---------- extraction ----------

def paths(ex):
    return [s.path for s in ex.sections]


def test_toc_headings(tmp_path):
    ex = ingest.extract_document(make_pdf(tmp_path / "t.pdf", bookmarks=True))
    assert ex.strategy == "toc"
    assert paths(ex) == ["Alpha Section", "Alpha Section > Sub A", "Beta Section"]


def test_font_headings_without_toc(tmp_path):
    ex = ingest.extract_document(make_pdf(tmp_path / "f.pdf", bookmarks=False))
    assert ex.strategy == "font"
    # the 26pt title is the document title, not a section
    assert paths(ex) == ["Alpha Section", "Alpha Section > Sub A", "Beta Section"]
    assert "Fixture Manual" not in paths(ex)


@pytest.mark.parametrize("bookmarks", [True, False])
def test_headers_and_footers_stripped(tmp_path, bookmarks):
    ex = ingest.extract_document(make_pdf(tmp_path / "h.pdf", bookmarks=bookmarks))
    text = " ".join(t for s in ex.sections for t, _ in s.paras)
    assert "Fixture Corp" not in text and "CONTROLLED" not in text and "Page 1 of 2" not in text
    assert "ordinary body text" in text


def test_page_tracking(tmp_path):
    chunks = ingest.chunk_sections(ingest.extract_document(make_pdf(tmp_path / "p.pdf", True)).sections)
    by = {c.section: c for c in chunks}
    assert (by["Alpha Section"].page_start, by["Alpha Section"].page_end) == (1, 1)
    assert (by["Alpha Section > Sub A"].page_start, by["Alpha Section > Sub A"].page_end) == (1, 2)
    assert (by["Beta Section"].page_start, by["Beta Section"].page_end) == (2, 2)


def test_table_rendered_as_pipe_rows(tmp_path):
    ex = ingest.extract_document(make_table_pdf(tmp_path / "tbl.pdf"))
    text = "\n".join(t for s in ex.sections for t, _ in s.paras)
    assert "Symptom | Cause | Fix" in text
    assert "Symptom 1 | Cause 1 | Fix 1" in text


# ---------- chunking ----------

def sec(paras, page=1):
    return ingest.Section("S", 1, [(p, page) for p in paras])


def test_small_section_is_one_chunk():
    chunks = ingest.chunk_sections([sec(["word " * 50, "more " * 50])], max_tokens=400)
    assert len(chunks) == 1


def test_oversize_section_splits_with_one_paragraph_overlap():
    paras = [f"p{i} " + "word " * 99 for i in range(6)]  # ~130 tokens each
    chunks = ingest.chunk_sections([sec(paras)], max_tokens=400)
    assert len(chunks) > 1
    for c in chunks:
        assert ingest._tokens(c.text) <= 400
    for a, b in zip(chunks, chunks[1:]):
        assert a.text.split("\n\n")[-1] == b.text.split("\n\n")[0]  # overlap
    joined = "\n\n".join(c.text for c in chunks)
    assert all(f"p{i} " in joined for i in range(6))


def test_oversize_table_splits_with_header_repeated():
    rows = ["Symptom | Cause | Fix"] + [f"row{i} | " + "x " * 40 + "| fix" for i in range(30)]
    chunks = ingest.chunk_sections([sec(["\n".join(rows)])], max_tokens=200)
    assert len(chunks) > 1
    assert all(c.text.startswith("Symptom | Cause | Fix") for c in chunks)
    assert all(ingest._tokens(c.text) <= 200 for c in chunks)


# ---------- ids & end-to-end ----------

def test_point_ids_deterministic():
    assert ingest.point_id("safety/a.pdf", 0) == ingest.point_id("safety/a.pdf", 0)
    assert ingest.point_id("safety/a.pdf", 0) != ingest.point_id("safety/a.pdf", 1)
    assert ingest.point_id("safety/a.pdf", 0) != ingest.point_id("quality/a.pdf", 0)


def test_embed_text_prefix_and_raw_payload():
    assert ingest.embed_text("Doc", "Sec > Sub", "body") == "Doc — Sec > Sub\nbody"
    entry = ingest.PdfEntry(Path("x.pdf"), "safety/x.pdf", META)
    p = ingest.build_payloads(entry, [ingest.DocChunk("Sec", "raw text", 2, 3)])[0]
    assert p["text"] == "raw text"  # raw, without the prefix
    assert {"domain", "doc", "section", "page_start", "page_end", "revision",
            "effective_date", "owner", "chunk_index", "source_path"} <= set(p)


def test_reingest_is_idempotent(tmp_path):
    from qdrant_client import QdrantClient

    corpus = write_corpus(tmp_path, {"a.pdf": META, "b.pdf": META},
                          {"safety/a.pdf": "bookmarks", "safety/b.pdf": "plain"})
    client = QdrantClient(path=str(tmp_path / "q"))
    try:
        ingest.ingest_all(client, FakeEmbedder(), corpus, corpus / "manifest.json", reset=True)
        n1 = client.count(config.COLLECTION, exact=True).count
        ingest.ingest_all(client, FakeEmbedder(), corpus, corpus / "manifest.json")
        n2 = client.count(config.COLLECTION, exact=True).count
        assert n1 > 0 and n1 == n2
        ingest.ingest_all(client, FakeEmbedder(), corpus, corpus / "manifest.json", reset=True)
        assert client.count(config.COLLECTION, exact=True).count == n1
        pts, _ = client.scroll(config.COLLECTION, limit=1, with_payload=True)
        assert pts[0].payload["domain"] == "safety" and pts[0].payload["source_path"].startswith("safety/")
    finally:
        client.close()
