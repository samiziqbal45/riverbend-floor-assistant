"""Riverbend Floor Assistant - Streamlit demo UI. Run: streamlit run ui/streamlit_app.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

EXAMPLES = [
    ("Clean", "What PPE do I need for sanitizer changeover?"),
    ("Clean", "How often should the RP-200 seal be replaced?"),
    ("Ambiguous", "The RP-200 is leaking sanitizer at the shaft. What do I do?"),
    ("Ambiguous", "Cap torque is reading 16 in-lb, is that ok?"),
    ("Conflict", "A bottle fell over on conveyor C2. Can I clear it with the conveyor in jog mode?"),
    ("Not covered", "How do I operate a forklift?"),
]
SINGLE_DOMAIN_ON = 0.85


@st.cache_resource
def load_backend():
    """Returns (ask_fn, is_stub). Falls back to the canned stub until app.pipeline exists."""
    try:
        from app.pipeline import ask
        return ask, False
    except ImportError:
        from ui.stub import ask
        return ask, True


def log_dir() -> Path:
    try:
        import config
        return Path(config.LOG_DIR)
    except Exception:  # noqa: BLE001
        return ROOT / "logs"


def pages(c) -> str:
    if c.page_start == c.page_end:
        return f"p. {c.page_start}"
    return f"pp. {c.page_start}–{c.page_end}"


@st.cache_data(show_spinner=False)
def page_image(pdf_path: str, page_no: int) -> bytes:
    import pymupdf
    with pymupdf.open(pdf_path) as doc:
        return doc[page_no - 1].get_pixmap(dpi=110).tobytes("png")


def find_pdf(doc_title: str, domain: str) -> Path | None:
    """Locate the PDF for a doc via corpus/manifest.json (title -> filename)."""
    try:
        manifest = json.loads((ROOT / "corpus" / "manifest.json").read_text(encoding="utf-8"))
    except OSError:
        return None
    for fname, meta in manifest.items():
        if meta.get("doc") == doc_title and meta.get("domain") == domain:
            p = ROOT / "corpus" / domain / fname
            return p if p.exists() else None
    return None


def render_route(res) -> None:
    r = res.route
    st.markdown(f"**Route:** {r.primary.title()} · {r.confidence:.2f}"
                f"  —  Searched: {' + '.join(res.searched_domains)}")
    cols = st.columns(len(r.scores))
    for col, (d, s) in zip(cols, r.scores.items()):
        col.caption(f"{d} {s:.2f}")
        col.progress(min(max(float(s), 0.0), 1.0))


def render_conflict(cf, chunks) -> None:
    a, b = chunks[cf.cite_a - 1], chunks[cf.cite_b - 1]
    lines = [
        f"**Documents disagree: {cf.topic}** ({cf.kind})",
        f"- [{cf.cite_a}] {a.doc}, {a.section}, {pages(a)}, rev {a.revision}: {cf.claim_a}",
        f"- [{cf.cite_b}] {b.doc}, {b.section}, {pages(b)}, rev {b.revision}: {cf.claim_b}",
    ]
    if cf.resolution:
        lines.append(f"**Resolved by precedence rule** ({cf.rule}): {cf.resolution}")
    else:
        lines.append("**Escalate to:** " + ", ".join(cf.escalate_to))
    st.warning("\n\n".join(lines), icon="⚠️")


def render_sources(res, key: str) -> None:
    with st.expander(f"Sources ({len(res.chunks)})"):
        for i, c in enumerate(res.chunks, 1):
            st.markdown(f"**[{i}] {c.doc}** — {c.section}  \n"
                        f"{c.domain} · {pages(c)} · rev {c.revision} · score {c.score:.3f}")
            st.text(c.text)
            if st.checkbox("Show PDF page", key=f"{key}-pdf-{i}"):
                pdf = find_pdf(c.doc, c.domain)
                if pdf:
                    st.image(page_image(str(pdf), c.page_start))
                else:
                    st.caption("PDF not found.")


def render_result(res, key: str) -> None:
    render_route(res)
    if res.not_covered:
        st.info(res.answer)
    else:
        st.markdown(res.answer)
    for cf in res.conflicts:
        render_conflict(cf, res.chunks)
    render_sources(res, key)
    with st.expander("Router reasoning & latency"):
        st.write(res.route.reasoning)
        st.json(res.latency_ms)


def conflicts_page() -> None:
    import pandas as pd
    path = log_dir() / "conflicts.jsonl"
    st.subheader("Conflicts found")
    if not path.exists():
        st.caption("No conflicts logged yet.")
        return
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def sidebar(is_stub: bool) -> tuple[float | None, bool, str | None]:
    st.sidebar.header("Examples")
    picked = None
    for kind, q in EXAMPLES:
        if st.sidebar.button(f"{kind}: {q}", key=f"ex-{q}", use_container_width=True):
            picked = q
    single = st.sidebar.toggle(f"Single-domain search (≥ {SINGLE_DOMAIN_ON})")
    show_conflicts = st.sidebar.toggle("Show conflicts log")
    if is_stub:
        st.sidebar.warning("STUB MODE: app.pipeline not found; canned answers.")
    return (SINGLE_DOMAIN_ON if single else None), show_conflicts, picked


@st.cache_resource(show_spinner="Preparing the document index...")
def startup_errors() -> list[str]:
    """Build the index if missing (no-Docker hosts), then fail fast on config problems."""
    from app.clients import get_embedder, get_qdrant
    from ui import startup

    client = get_qdrant()
    try:
        startup.ensure_index(client, get_embedder())
    except Exception as e:  # noqa: BLE001
        return [f"Could not build the document index: {e}"]
    return startup.preflight_errors(client)


def password_gate() -> bool:
    """Single shared-password field; off when APP_PASSWORD is unset."""
    from ui import startup

    try:
        import config
        expected = config.APP_PASSWORD
    except Exception:  # noqa: BLE001
        expected = None
    if not expected or st.session_state.get("authed"):
        return True
    entered = st.text_input("Password", type="password")
    if entered:
        if startup.password_ok(entered, expected):
            st.session_state["authed"] = True
            st.rerun()
        st.error("Incorrect password.")
    return False


def main() -> None:
    st.set_page_config(page_title="Riverbend Floor Assistant", layout="wide")
    st.title("Riverbend Floor Assistant")
    if not password_gate():
        st.stop()
    ask, is_stub = load_backend()
    if not is_stub:
        errors = startup_errors()
        if errors:
            for e in errors:
                st.error(e)
            st.stop()
    threshold, show_conflicts, picked = sidebar(is_stub)
    if show_conflicts:
        conflicts_page()
        return

    history = st.session_state.setdefault("history", [])
    for n, (q, res) in enumerate(history):
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            render_result(res, f"h{n}")

    question = st.chat_input("Ask about safety, maintenance, or quality...") or picked
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                try:
                    kwargs = {} if threshold is None else {"single_domain_threshold": threshold}
                    res = ask(question, **kwargs)
                except Exception as e:  # noqa: BLE001
                    st.error(f"Request failed: {e}")
                    return
            history.append((question, res))
            render_result(res, f"h{len(history) - 1}")


main()
