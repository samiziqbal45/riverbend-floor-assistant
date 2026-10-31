"""Render corpus/source/<domain>/*.md to corpus/<domain>/<slug>.pdf with bookmarks.

Idempotent: re-running overwrites the PDFs. Usage: python scripts/build_corpus.py
"""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle,
    Flowable, ListFlowable, ListItem,
)

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
SOURCE = CORPUS / "source"

SS = getSampleStyleSheet()
BODY = ParagraphStyle("Body", parent=SS["BodyText"], fontSize=10, leading=13.5, spaceAfter=5)
H1 = ParagraphStyle("H1", parent=SS["Title"], fontSize=22, leading=26, spaceAfter=12)
H2 = ParagraphStyle("H2", parent=SS["Heading1"], fontSize=16, leading=20, spaceBefore=12, spaceAfter=6)
H3 = ParagraphStyle("H3", parent=SS["Heading2"], fontSize=12.5, leading=16, spaceBefore=8, spaceAfter=4)
CELL = ParagraphStyle("Cell", parent=BODY, fontSize=8.5, leading=10.5, spaceAfter=0)
CELL_H = ParagraphStyle("CellH", parent=CELL, fontName="Helvetica-Bold")
CALLOUT = ParagraphStyle("Callout", parent=BODY, backColor=colors.HexColor("#FFF4D6"),
                         borderColor=colors.HexColor("#D9A400"), borderWidth=0.8,
                         borderPadding=5, leftIndent=8, rightIndent=8, spaceBefore=6, spaceAfter=10)


def inline(text: str) -> str:
    """Convert the small inline markdown subset to reportlab markup."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", text)
    return text


class HeadingParagraph(Paragraph):
    """Heading that registers its own PDF bookmark + outline entry when drawn,
    so the bookmark always lands on the page the heading is actually on."""

    def __init__(self, text: str, style, title: str, level: int, key: str):
        super().__init__(text, style)
        self.bm_title, self.bm_level, self.bm_key = title, level, key

    def draw(self):
        c = self.canv
        c.bookmarkPage(self.bm_key)
        c.addOutlineEntry(self.bm_title, self.bm_key, level=self.bm_level, closed=False)
        super().draw()


def draw_decorations(c, info: dict, total: int) -> None:
    """Header/footer for controlled documents (drawn per page via onPage)."""
    w, h = letter
    c.saveState()
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#444444"))
    c.drawString(0.75 * inch, h - 0.5 * inch, f"Riverbend Packaging  |  {info['doc']}")
    c.drawRightString(w - 0.75 * inch, h - 0.5 * inch, "CONTROLLED DOCUMENT")
    c.setStrokeColor(colors.HexColor("#999999"))
    c.line(0.75 * inch, h - 0.55 * inch, w - 0.75 * inch, h - 0.55 * inch)
    c.line(0.75 * inch, 0.6 * inch, w - 0.75 * inch, 0.6 * inch)
    c.drawString(0.75 * inch, 0.42 * inch,
                 f"Rev {info['revision']}  |  Effective {info['effective_date']}  |  Owner: {info['owner']}")
    c.drawRightString(w - 0.75 * inch, 0.42 * inch, f"Page {c.getPageNumber()} of {total}")
    c.restoreState()


def make_table(rows: list[list[str]]) -> Table:
    data = []
    for i, row in enumerate(rows):
        style = CELL_H if i == 0 else CELL
        data.append([Paragraph(inline(c), style) for c in row])
    ncols = len(rows[0])
    t = Table(data, colWidths=[(7.0 * inch) / ncols] * ncols, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F1")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#888888")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def parse_table(lines: list[str]) -> list[list[str]]:
    rows = []
    for ln in lines:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
            continue
        rows.append(cells)
    return rows


def md_to_story(md: str) -> list:
    story: list = []
    lines = md.splitlines()
    i, n_bm = 0, 0
    para: list[str] = []

    def flush_para():
        if para:
            story.append(Paragraph(inline(" ".join(para)), BODY))
            para.clear()

    while i < len(lines):
        ln = lines[i]
        if ln.startswith("# "):
            flush_para()
            story.append(Paragraph(inline(ln[2:]), H1))
        elif ln.startswith("## ") or ln.startswith("### "):
            flush_para()
            level = 0 if ln.startswith("## ") else 1
            title = ln.lstrip("#").strip()
            n_bm += 1
            story.append(HeadingParagraph(inline(title), H2 if level == 0 else H3,
                                          title, level, f"bm{n_bm}"))
        elif ln.startswith("|"):
            flush_para()
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            story.append(make_table(parse_table(block)))
            story.append(Spacer(1, 8))
            continue
        elif ln.startswith(">"):
            flush_para()
            block = []
            while i < len(lines) and lines[i].startswith(">"):
                block.append(lines[i].lstrip("> ").strip())
                i += 1
            story.append(Paragraph(inline(" ".join(block)), CALLOUT))
            continue
        elif re.match(r"^\d+\.\s", ln) or ln.startswith("- "):
            flush_para()
            ordered = bool(re.match(r"^\d+\.\s", ln))
            items = []
            while i < len(lines) and (re.match(r"^\d+\.\s", lines[i]) or lines[i].startswith("- ")):
                txt = re.sub(r"^(\d+\.|-)\s+", "", lines[i])
                items.append(ListItem(Paragraph(inline(txt), BODY), leftIndent=18))
                i += 1
            story.append(ListFlowable(items, bulletType="1" if ordered else "bullet",
                                      leftIndent=18, bulletFontSize=9))
            continue
        elif not ln.strip():
            flush_para()
        else:
            para.append(ln.strip())
        i += 1
    flush_para()
    return story


def _build_once(md_path: Path, out, meta: dict, total: int) -> int:
    doc = BaseDocTemplate(out, pagesize=letter, title=meta["doc"], author="Riverbend Packaging",
                          leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                          topMargin=0.85 * inch, bottomMargin=0.85 * inch)
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([PageTemplate(
        id="p", frames=[frame], onPage=lambda c, d: draw_decorations(c, meta, total))])
    doc.build(md_to_story(md_path.read_text(encoding="utf-8")))
    return doc.page


def build_pdf(md_path: Path, out_path: Path, meta: dict) -> None:
    """Two passes: the first counts pages so the footer can say 'Page X of Y'
    without deferring showPage (which would break bookmark page references)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = _build_once(md_path, io.BytesIO(), meta, 0)
    _build_once(md_path, str(out_path), meta, total)


def main() -> int:
    manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    built = 0
    for pdf_name, meta in manifest.items():
        md_path = SOURCE / meta["domain"] / (Path(pdf_name).stem + ".md")
        if not md_path.exists():
            print(f"MISSING source for {pdf_name}: {md_path}", file=sys.stderr)
            return 1
        build_pdf(md_path, CORPUS / meta["domain"] / pdf_name, meta)
        built += 1
        print(f"built {meta['domain']}/{pdf_name}")
    print(f"{built} PDFs built")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
