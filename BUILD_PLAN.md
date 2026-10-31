# Plant Floor Doc Assistant — Build Plan

Prototype for the AI Prototype Challenge (manufacturing scenario). Floor supervisors ask questions; the system routes to the right documentation (safety, maintenance, quality), retrieves from the top two likely domains, and returns one grounded, cited answer that surfaces conflicts between documents instead of hiding them.

**How to use this file:** put it in the repo root along with `pipeline.py` (the reference implementation). Give Claude Code the "Shared context" section once, then run each component prompt in order. Each component has acceptance criteria — don't move on until they pass.

---

## Shared context (paste first)

```
We are building a Python prototype: a documentation Q&A assistant for manufacturing
floor supervisors. Read BUILD_PLAN.md and pipeline.py in the repo root before starting.

Architecture:
- Offline: PDF docs -> extract with PyMuPDF (pymupdf) -> chunk by heading (from the PDF
  outline/TOC, falling back to font-size detection) -> embed (fastembed,
  BAAI/bge-small-en-v1.5, 384-dim) -> one Qdrant collection in local mode
  (QdrantClient(path="./qdrant_data")), no server. Document metadata (domain,
  revision, effective_date, owner) comes from corpus/manifest.json, not the PDF
  itself. Every chunk carries payload: domain, doc, section, page_start, page_end,
  revision, effective_date, owner, text.
- Online: question -> LLM router scores all 3 domains (safety, maintenance, quality),
  normalized to sum to 1 -> two filtered Qdrant searches (top-2 domains) -> merge with
  a per-domain floor -> ONE generation call over all chunks -> structured output
  {answer, conflicts[], not_covered} with inline [n] citations.
- Router scores by relevance only. There is NO safety bias in routing.
- Conflicts between documents are never silently resolved: both claims are shown with
  citations and an escalation owner. Precedence rules are client-configured and default
  to "none".
- LLM: Anthropic SDK, model "claude-sonnet-5-5", structured outputs via forced tool use.
- UI: Streamlit. Tests: pytest, with the LLM mocked.

Conventions: Python 3.11, type hints, dataclasses, all tunables in config.py, no
global side effects at import time (lazy-init clients), small functions, docstrings
only where non-obvious. Keep it a prototype: no user accounts (a single shared APP_PASSWORD gate only),
no DB beyond Qdrant. Docker is used only for packaging and deployment (Component 10).
```

---

## Repo layout (target)

```
plant-assistant/
├── BUILD_PLAN.md
├── pipeline.py              # reference only; logic moves into app/
├── requirements.txt
├── README.md
├── config.py
├── corpus/
│   ├── manifest.json        # per-PDF metadata: domain, revision, date, owner
│   ├── source/<domain>/*.md # authoring source for the synthetic docs
│   ├── safety/*.pdf
│   ├── maintenance/*.pdf
│   └── quality/*.pdf
├── scripts/build_corpus.py  # md -> pdf with bookmarks
├── app/
│   ├── __init__.py
│   ├── clients.py           # lazy Qdrant, embedder, Anthropic
│   ├── ingest.py
│   ├── router.py
│   ├── retrieve.py
│   ├── generate.py
│   ├── precedence.py
│   └── pipeline.py          # ask()
├── ui/streamlit_app.py
├── eval/
│   ├── golden.jsonl
│   ├── fewshot.jsonl        # router examples, disjoint from golden
│   ├── run_eval.py
│   └── results/
└── tests/
```

## Build order

| # | Component | Depends on | Est. time |
|---|-----------|-----------|-----------|
| 0 | Scaffold + config | — | 15 min |
| 1 | Sample corpus | 0 | 45 min |
| 2 | Ingestion | 0, 1 | 45 min |
| 3 | Router | 0 | 45 min |
| 4 | Dual retrieval | 2 | 30 min |
| 5 | Generation + conflict detection | 4 | 45 min |
| 6 | Orchestration + precedence | 3, 4, 5 | 30 min |
| 7 | Streamlit UI | 6 | 45 min |
| 8 | Golden set + eval harness | 6 | 60 min |
| 9 | README + demo script | all | 20 min |
| 10 | Deployment (local Docker + cloud) | 7 | 45–60 min |

Critical path for a demo: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7. Eval (8) is what you defend the design with — don't skip it. Deploy (10) as soon as 7 works, even before 8, so that any hosting problems turn up early and not the night before the interview. After that, redeploying is one command.

---

## Component 0 — Scaffold + config

**Prompt:**
```
Create the repo skeleton from the "Repo layout" in BUILD_PLAN.md.

1. requirements.txt: qdrant-client, fastembed, anthropic, streamlit, pydantic,
   python-dotenv, pytest, pandas, matplotlib, pymupdf, reportlab, markdown.
2. config.py with these settings (env-overridable where sensible):
   DOMAINS = ["safety", "maintenance", "quality"]
   COLLECTION = "plant_docs"; QDRANT_PATH = "./qdrant_data"
   QDRANT_URL = None; QDRANT_API_KEY = None   # if QDRANT_URL is set, use a
                                               # Qdrant server instead of local mode
   FASTEMBED_CACHE = "./models"               # so the model can be baked into an image
   LOG_DIR = "./logs"; APP_PASSWORD = None    # password gate is off when unset
   EMBED_MODEL = "BAAI/bge-small-en-v1.5"; EMBED_DIM = 384
   LLM_MODEL = "claude-sonnet-5-5"
   K_PER_DOMAIN = 5; FINAL_K = 6; MIN_PER_DOMAIN = 2
   SINGLE_DOMAIN_THRESHOLD = None   # if set (e.g. 0.85), search only top domain
                                     # when its score >= threshold
   MAX_CHUNK_TOKENS = 400 (approximate with words*1.3)
   PRECEDENCE_RULES = []             # client-configured; empty = no auto-resolution
   DOMAIN_OWNERS = {"safety": "EHS Lead", "maintenance": "Maintenance Supervisor",
                    "quality": "Quality Engineer"}
3. app/clients.py: lazy singletons get_qdrant(), get_embedder(), get_llm(),
   loading ANTHROPIC_API_KEY from .env. get_qdrant() returns
   QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY) when QDRANT_URL is set,
   otherwise QdrantClient(path=QDRANT_PATH). get_embedder() passes
   cache_dir=FASTEMBED_CACHE. Every setting reads from env vars first, so the same
   code runs locally, in Docker, and in the cloud with no code changes.
4. .env.example, .gitignore (qdrant_data/, models/, logs/, .env, eval/results/).

Acceptance: `python -c "import config, app.clients"` succeeds without creating
any clients or files.
```

---

## Component 1 — Sample corpus

**Prompt:**
```
Write a realistic synthetic document corpus in corpus/ for a fictional plant
"Riverbend Packaging" that runs a filling/capping line with conveyors, a
centrifugal pump (model "RP-200"), a capper (model "CX-4"), and a hydraulic press.

Step A — author the content as markdown in corpus/source/<domain>/<slug>.md.
Use # for the doc title, ## and ### for sections. Each doc 600–1500 words, with
realistic touches: numbered steps, warning callouts, and at least one table per
manual (troubleshooting tables, spec tables).

Step B — write corpus/manifest.json, one entry per document:
{"<slug>.pdf": {"doc": "<title>", "domain": "safety|maintenance|quality",
  "revision": "C", "effective_date": "YYYY-MM-DD", "owner": "<role>"}}

Step C — write scripts/build_corpus.py, which renders each .md to
corpus/<domain>/<slug>.pdf with reportlab:
- Headings get larger bold fonts AND PDF outline bookmarks (bookmarkPage +
  addOutlineEntry, level 0 for ##, level 1 for ###), so ingestion can read the
  structure from the TOC.
- Render tables as reportlab Tables, and add a header/footer with the doc title,
  revision, and "Page X of Y" (like real controlled documents).
- Make it idempotent and run it at the end.

Write 4 docs per domain:
- safety: Lockout/Tagout procedure; PPE matrix by task; Conveyor jam clearing safety
  procedure; Chemical spill / sanitizer handling response.
- maintenance: RP-200 pump manual (operation, troubleshooting table, PM schedule);
  CX-4 capper maintenance manual (torque head calibration, changeover); Conveyor
  maintenance & jam troubleshooting guide; Hydraulic press PM checklist.
- quality: Cap torque specification & inspection method; Fill weight/volume
  specification with SPC limits; Nonconformance & hold procedure; Incoming
  material inspection standard.

Deliberately plant these test cases and list them in corpus/PLANTED_CASES.md:
1. CONFLICT (cross-domain): the conveyor maintenance guide says small jams may be
   cleared "with the conveyor in jog mode at reduced speed"; the safety jam-clearing
   procedure requires full LOTO before any hand enters the guarded zone.
2. CONFLICT (numeric): the quality cap torque spec says 18–22 in-lb; the CX-4
   capper manual's calibration section says set heads to 15–20 in-lb.
3. AMBIGUOUS questions that legitimately span domains (e.g. torque, jam clearing,
   pump leaking sanitizer).
4. NOT COVERED: topics deliberately absent (e.g. forklift operation, payroll) so
   the system can be tested for refusal.

Acceptance: 12 PDFs build, each opens with visible bookmarks, manifest.json has
an entry for every PDF, and PLANTED_CASES.md lists each case with the exact
file, section, and page.
```

---

## Component 2 — Ingestion

**Prompt:**
```
Implement app/ingest.py and a CLI entry `python -m app.ingest [--reset]`.

1. Load corpus/manifest.json. For every corpus/<domain>/*.pdf, look up its metadata
   (fail loudly if a PDF has no manifest entry or its domain is not in
   config.DOMAINS; warn about manifest entries with no PDF).
2. Extract with PyMuPDF (import pymupdf):
   - Strip repeating headers/footers: drop text blocks in the top/bottom ~6% of
     the page, or lines that repeat on more than 50% of pages.
   - Section boundaries: use doc.get_toc() if it is non-empty, locating each
     heading's text on its page to split exactly at it. Fallback when there is no
     TOC: page.get_text("dict"), where spans whose font size is larger than the
     body median, or bold short lines, count as headings (level by size rank).
   - Keep tables readable: try page.find_tables() and render each table as
     pipe-delimited rows inside its chunk instead of jumbled text.
3. Chunk per section. `section` is the heading path (e.g. "Troubleshooting > Seal
   leak"). Track page_start/page_end for each chunk. If a section exceeds
   MAX_CHUNK_TOKENS, split on paragraph boundaries with ~1 paragraph of overlap.
   Prepend "<doc> — <section>\n" to the embedded text, but store the raw text
   separately.
4. Payload per chunk: text, domain, doc, section, page_start, page_end, revision,
   effective_date, owner, chunk_index, source_path.
5. Embed in batches with fastembed; upsert to Qdrant with deterministic IDs
   (uuid5 of source_path + chunk_index) so re-ingest is idempotent.
6. Create payload index on `domain` (keyword).
7. --reset drops and recreates the collection.
8. Print a summary table: chunks per domain, per doc, and which heading
   strategy (toc / font) each PDF used.
9. Add `python -m app.ingest --inspect <pdf>` to print the extracted sections
   with their pages, for debugging extraction.

Tests (tests/test_ingest.py): generate a tiny 2-page PDF in a fixture with
reportlab (with and without bookmarks); verify manifest lookup, TOC-based and
font-based heading detection, header/footer stripping, page tracking, oversize
split, and deterministic IDs. Use a temp Qdrant path.

Acceptance: ingest runs end-to-end on corpus/, running twice yields the same
point count, --inspect on the pump manual shows the troubleshooting table as
readable rows, and tests pass.
```

---

## Component 3 — Router

**Prompt:**
```
Implement app/router.py, starting from route() in the root pipeline.py.

- Forced tool use with a `route` tool returning a score 0–1 per domain plus
  `reasoning` (one sentence). Normalize scores to sum to 1; if all are 0, fall back
  to uniform.
- Router prompt: describe each domain concretely (safety = hazard controls, LOTO,
  PPE, emergency/incident response; maintenance = operation, repair,
  troubleshooting, PM, parts; quality = specs, tolerances, inspection, SPC,
  nonconformance/holds). State explicitly: "Score by relevance only. Do not favor
  any domain by default. If a question spans domains, split probability between
  them."
- Load few-shot examples from eval/fewshot.jsonl (create 6 examples: 2 clean per
  domain, include 2 cross-domain with split scores). These MUST NOT overlap with
  eval/golden.jsonl.
- Return a Route dataclass: scores, reasoning, ranked, primary, margin (top1 - top2).
- Handle API errors with one retry, then raise a clear exception.

Tests: mock the Anthropic client; verify normalization, ranking, margin, retry.

Acceptance: route("What PPE do I need for sanitizer changeover?") ranks safety
first; route("Cap torque is reading 16 in-lb, is that ok?") splits quality and
maintenance.
```

---

## Component 4 — Dual retrieval

**Prompt:**
```
Implement app/retrieve.py, starting from retrieve() in the root pipeline.py.

- retrieve(question, domains) -> list[Chunk]: embed the query once, run one
  Qdrant query_points per domain with a `domain` match filter, limit K_PER_DOMAIN.
- Merge: take the top MIN_PER_DOMAIN from each domain first (the floor), then fill
  to FINAL_K by raw cosine score across both, dedupe by id, sort by score desc.
- If a domain returns nothing, don't fail — fill from the other.
- Chunk is a dataclass with all payload fields + id + score.
- Domain selection lives in a separate pure function
  choose_domains(route, threshold=config.SINGLE_DOMAIN_THRESHOLD) -> list[str]:
  top-2 by default; top-1 only if threshold is set and top score >= threshold.

Tests with a temp Qdrant and hand-made vectors: the floor guarantees the
secondary domain appears, dedupe works, the empty domain is handled,
choose_domains respects the threshold.

Acceptance: tests pass; on the real corpus, the jam-clearing question returns
chunks from both safety and maintenance.
```

---

## Component 5 — Generation + conflict detection

**Prompt:**
```
Implement app/generate.py: ONE LLM call over all retrieved chunks, structured output
via a forced `answer` tool:

{
  "answer": str,                 # concise, actionable, inline [n] citations
  "citations_used": [int],
  "not_covered": bool,           # true if excerpts don't answer the question
  "conflicts": [
    {"topic": str,
     "claim_a": str, "cite_a": int,
     "claim_b": str, "cite_b": int,
     "kind": "contradiction" | "scope_difference" | "revision_mismatch"}
  ]
}

Context formatting: number the chunks [1]..[n], each header shows
"(domain | doc | section | p. N | rev X, effective YYYY-MM-DD)". Page ranges
render as "pp. N–M".

System prompt requirements:
- Use ONLY the excerpts. No outside knowledge, no guessing.
- If excerpts don't answer: not_covered=true, say what document would be needed.
- If two excerpts disagree, NEVER merge, average, or pick one. Report both in
  `conflicts` and in the answer write: "The documents disagree on <topic>:" then
  both positions with citations, then "Confirm with <owner> before proceeding."
- If they only appear to disagree because they cover different conditions, explain
  the distinction and use kind=scope_difference.
- Audience: supervisors on the floor. Short, step-oriented, no preamble.

Post-processing: validate that cite indices are in range, and drop invalid
citations with a logged warning.

Tests: mock the LLM; verify context formatting, schema validation, invalid
citation handling.

Acceptance: on the real corpus, both planted conflicts from PLANTED_CASES.md are
reported in `conflicts`; "How do I operate a forklift?" returns not_covered=true.
```

---

## Component 6 — Orchestration + precedence

**Prompt:**
```
Implement app/precedence.py and app/pipeline.py.

precedence.py:
- apply_precedence(conflicts, chunks, rules=config.PRECEDENCE_RULES) -> conflicts
  annotated with `resolution` and `escalate_to`.
- Supported rule types (all opt-in; default list is empty):
  {"type": "latest_revision"}  -> if both claims come from the SAME doc at
      different revisions/dates, the later effective_date wins.
  {"type": "domain_order", "order": [...]} -> earlier domain wins.
- If no rule resolves it: resolution=None, escalate_to = DOMAIN_OWNERS of both
  domains (or the chunk `owner` fields if present).

pipeline.py:
- ask(question) -> AskResult dataclass: question, route (scores, reasoning,
  primary, confidence, margin), searched_domains, chunks, answer, conflicts
  (post-precedence), not_covered, latency_ms per stage (route, retrieve, generate).
- Log each call as one JSON line to logs/queries.jsonl (question, route, searched,
  conflicts, not_covered, latencies). Conflicts are also appended to
  logs/conflicts.jsonl, which serves as the documentation-quality report.

Acceptance: `python -c "from app.pipeline import ask; print(ask('Cap torque is reading
16 in-lb, is that ok?'))"` returns a result with both domains searched and the
torque conflict escalated to Quality Engineer + Maintenance Supervisor.
```

---

## Component 7 — Streamlit UI

**Prompt:**
```
Build ui/streamlit_app.py, a clean demo UI for floor supervisors.

Layout:
- Title "Riverbend Floor Assistant", a chat input, and a chat history in session_state.
- For each answer:
  1. Route badge: primary domain + confidence (e.g. "Maintenance · 0.62"), plus a
     small horizontal bar of all 3 domain scores. Show "Searched: maintenance +
     quality".
  2. Answer text with [n] citations.
  3. If conflicts: a prominent amber warning box per conflict showing both claims,
     their sources, kind, and "Escalate to: …". If resolved by precedence, show
     the rule that resolved it.
  4. If not_covered: a neutral info box.
  5. Expander "Sources": each chunk with [n], domain, doc, section, page(s),
     revision, score, and text, plus a button to open the PDF at that page
     (render the page as an image with pymupdf page.get_pixmap() and show it inline).
  6. Expander "Router reasoning" + per-stage latency.
- Sidebar: 6 example question buttons (2 clean, 2 ambiguous, 1 conflict, 1 not
  covered), pulled from PLANTED_CASES.md; a toggle for SINGLE_DOMAIN_THRESHOLD
  (off / 0.85); a link to "Conflicts found" that renders logs/conflicts.jsonl as
  a table.
- Cache the clients with st.cache_resource.
- If config.APP_PASSWORD is set, show a single password field before the app
  loads (compare with hmac.compare_digest, store the result in session_state). A
  public demo URL must not leave the Anthropic key open to anyone.

Acceptance: `streamlit run ui/streamlit_app.py` works; every example button
produces a correct-looking result; the conflict example renders the warning box.
```

---

## Component 8 — Golden set + eval harness

**Prompt:**
```
Create eval/golden.jsonl (25–30 items) and eval/run_eval.py.

Golden item schema:
{"id", "question",
 "acceptable_domains": [...],      # 1 for clean, 2 for ambiguous
 "expected_docs": [...],           # docs that must appear in retrieved chunks
 "expect_conflict": bool,
 "expect_not_covered": bool,
 "category": "clean" | "ambiguous" | "conflict" | "not_covered"}

Mix: ~12 clean (4/domain), ~8 ambiguous, 3 conflict (incl. both planted), 3 not
covered. No overlap with eval/fewshot.jsonl — assert this in run_eval.py.

run_eval.py metrics, overall and per category:
- Router top-1 accuracy (top-1 in acceptable_domains)
- Router top-2 coverage (all acceptable_domains within the searched domains)
- Retrieval recall@FINAL_K on expected_docs
- Conflict detection: precision/recall on expect_conflict
- Refusal accuracy on expect_not_covered
- p50/p95 latency per stage

Threshold sweep (route once per question, cache it, then re-run retrieval only):
for SINGLE_DOMAIN_THRESHOLD in [None, 0.6, 0.7, 0.8, 0.9], report top-2
coverage and the % of queries that used single-domain search. Save
eval/results/threshold_sweep.png (coverage vs % single-domain) and
eval/results/report.md with a summary table and a per-question failure list.

CLI flags: --skip-generation (route + retrieve only, cheap), --limit N.

Acceptance: the full run completes, report.md and the chart are written, and
failures are listed with question, expected vs. actual.
```

---

## Component 9 — README + demo script

**Prompt:**
```
Write README.md: one-paragraph problem/solution, architecture diagram (mermaid),
setup (venv, pip install, .env, `python -m app.ingest --reset`, streamlit run),
eval command, key design decisions (relevance-only routing; dual filtered retrieval
with per-domain floor; one generation call; conflicts surfaced not resolved;
client-configured precedence), known limitations (synthetic PDFs have clean
bookmarks; real scanned manuals will need OCR, and messy layouts need the
font-size fallback or a layout model), and "path to production" (Qdrant
server, real PDF ingestion from the client's DMS with OCR for scans, access control, feedback loop from
logs, conflict report to doc owners).

Also write DEMO.md: a 5-minute demo script — 1 clean question, 1 ambiguous
(show the split route scores), the jam-clearing conflict (show the warning +
escalation), 1 not-covered refusal, then the eval report + threshold chart — with
talking points for each.
```

---

## Component 10 — Deployment (local + cloud)

Design: **one Docker image runs everywhere.** The embedding model and the built
Qdrant index are baked into the image at build time. The container starts in
seconds, needs no volume, and only needs ANTHROPIC_API_KEY (+ APP_PASSWORD) at
runtime. That's the right fit here: the corpus is small and changes rarely, and
Qdrant local mode is single-process, which suits one container well.

**Prompt:**
```
Add deployment support so the app runs locally and on a cloud host from the same
image.

1. Dockerfile (python:3.11-slim, multi-stage if helpful):
   - pip install -r requirements.txt
   - COPY the repo; RUN python scripts/build_corpus.py && python -m app.ingest --reset
     so the PDFs, the fastembed model (in ./models) and ./qdrant_data are all baked
     into the image. Ingest needs no API key (embeddings are local), so this works
     at build time.
   - Run as a non-root user; EXPOSE 8080; HEALTHCHECK on Streamlit's
     /_stcore/health.
   - CMD: streamlit run ui/streamlit_app.py --server.port=${PORT:-8080}
     --server.address=0.0.0.0 --server.headless=true
   - .dockerignore: .env, logs/, eval/results/, .git, __pycache__, qdrant_data/
     (rebuilt in the image).

2. Local:
   - Makefile targets: setup (venv + install), corpus, ingest, run (streamlit
     locally, no Docker), test, eval, docker-build, docker-run (passes --env-file .env,
     maps 8080).
   - docker-compose.yml with two profiles:
     default: just the app (embedded Qdrant, matches the cloud setup)
     server: app + qdrant/qdrant container, with QDRANT_URL=http://qdrant:6333 and
     an ingest step run against it. This demonstrates the production storage path
     with no code changes.

3. Cloud — target Google Cloud Run (scales to zero, HTTPS, supports Streamlit
   websockets):
   - scripts/deploy_cloudrun.sh: gcloud builds submit → gcloud run deploy with
     --port 8080, --memory 2Gi, --cpu 1, --max-instances 1 (embedded Qdrant is
     single-writer, and this also caps cost), --min-instances 0, --timeout 3600,
     --session-affinity, --allow-unauthenticated (the app password is the gate).
     Pass ANTHROPIC_API_KEY and APP_PASSWORD from Secret Manager (--set-secrets),
     never as plain env vars. Parameterize PROJECT_ID and REGION.
   - Alternatives in README with exact steps: Render / Fly.io (same Dockerfile), and
     Streamlit Community Cloud (no Docker; it builds the index on first start in
     st.cache_resource if ./qdrant_data is missing — add that fallback to the UI
     so this path works too).

4. Runtime concerns:
   - Logs: in containers, also write query/conflict logs to stdout as JSON
     (Cloud Run captures them); file logs under LOG_DIR are best-effort, because
     the cloud filesystem is ephemeral. Note this in the README.
   - The "Conflicts found" page reads the file log, so in cloud it only shows the
     current instance's history. That's acceptable for a demo; document it.
   - Fail fast at startup with a clear message if ANTHROPIC_API_KEY is missing
     or the collection is empty.

5. scripts/smoke_test.py <base_url>: hits /_stcore/health, then calls ask() in-process
   against the built image (`docker run ... python scripts/smoke_test.py --local`)
   with one clean question and one conflict question, and asserts the expected
   domains and that a conflict is present.

Acceptance:
- `make run` works with no Docker.
- `make docker-build && make docker-run` → app at http://localhost:8080, password
  gate works, and the first query answers in <10s with no model download at runtime.
- `docker compose --profile server up` works against a Qdrant server.
- deploy_cloudrun.sh deploys successfully and smoke_test passes against the
  public URL.
```

---

## Definition of done

- [ ] `python -m app.ingest --reset` builds the index from scratch
- [ ] `pytest` passes
- [ ] Streamlit demo handles all 6 example questions correctly
- [ ] Both planted conflicts are detected and escalated
- [ ] Not-covered questions are refused, not answered from general knowledge
- [ ] `eval/results/report.md` + threshold chart exist with real numbers
- [ ] README + DEMO.md written
- [ ] Runs locally via `make run` and via Docker on :8080
- [ ] Deployed to a public URL behind the app password, and the smoke test passes

## Interview talking points this plan sets up

- **Why no safety bias:** routing by relevance; unequal costs are handled through conflict surfacing and client-configured precedence, not by skewing the router.
- **Why two filtered searches instead of one unfiltered search:** a guaranteed slot for each routed domain, comparable scores (same model, same collection), and a clear audit trail of which domains were searched.
- **Why one generation call:** the model sees both domains together, so it can notice conflicts. That's cheaper and more coherent than generating per domain and merging.
- **Threshold chosen by data:** the golden-set sweep shows the trade-off between accuracy and the number of chunks sent to the model.
- **Conflict log as a by-product:** the system shows where the client's documentation contradicts itself.