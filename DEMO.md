# Demo script (5 minutes)

**The story in one line:** a supervisor asks a question; the assistant finds the right
documents across three owners, answers with citations, and, when the documents contradict
each other, says so and says who to call instead of quietly picking one.

## Before you start (10 minutes, the day before and again 15 minutes before)

1. `.env` has a working `ANTHROPIC_API_KEY` (and `APP_PASSWORD` if the demo is hosted).
2. The index exists: `python -m app.ingest --reset` (or use the deployed image, which has it baked in).
3. Start the app: `streamlit run ui/streamlit_app.py`, then **ask one throwaway question** so
   the embedding model and the LLM connection are warm. The first request is the slowest.
4. Run `python -m eval.run_eval` once, so `eval/results/report.md` and `threshold_sweep.png`
   hold real numbers from the current code. Read the report before the demo and note the
   three or four numbers you want to quote (see step 5). Do not quote numbers you have not seen.
5. Open in tabs: the app, `eval/results/report.md`, `eval/results/threshold_sweep.png`.
6. Know the fallback: the LLM is not deterministic. If the conflict box does not appear, ask
   the same question again (it has appeared consistently in testing), and say so honestly.

The sidebar's example buttons are the six questions below, so you do not have to type.

---

## 0:00 - Setup (30 s)

> "Plant floors run on three families of documents: safety procedures, maintenance manuals,
> quality specs. Different owners, different revision dates. Supervisors don't know which one
> to open, and nobody knows when two of them disagree. This is a prototype of the layer in
> between. The corpus is 12 synthetic PDFs for a fictional packaging plant."

Show the UI title and the sidebar. Mention the three domains.

## 0:30 - A clean question (45 s)

**Click:** *Clean: How often should the RP-200 seal be replaced?*

(Not the PPE example: "What PPE do I need for sanitizer changeover?" currently flags a real
face-shield disagreement between the PPE matrix and the spill procedure, so it is not a clean case.)

**What you'll see:** route badge **Maintenance, about 0.94**, with the three-bar score chart
(safety and quality near 0.03). "Searched: maintenance + safety". A short cited answer:
replace the mechanical seal annually as part of the 3-hour annual PM, and also when it leaks,
cited to the RP-200 pump manual. No warning boxes.

**Talking points**
- "The router scores all three domains by relevance only. There is deliberately no safety
  bias; I'll show you where safety gets protected in a minute."
- Open **Sources**: every claim is numbered and traces to a document, section, page and
  revision. Tick "Show PDF page" on one source to show the real page.
- "Even though routing was confident, we still searched the top *two* domains. That costs
  little and is what makes the next two examples work."

## 1:15 - An ambiguous question: the router splits (60 s)

**Click:** *Ambiguous: The RP-200 is leaking sanitizer at the shaft. What do I do?*

**What you'll see:** route roughly **Maintenance 0.65 / Safety 0.30**, searched maintenance +
safety. Sources come from the pump manual and the chemical-spill procedure. The answer opens
with "The documents disagree on whether to stop the pump", and a warning box shows both sides:
the pump manual says a few drops per minute of weeping can continue for a limited time, the
spill procedure says to stop the equipment with no exception for minor weeping. Escalate to
Maintenance Supervisor and EHS Lead.

**Talking points**
- "This question genuinely lives in two documents. The router doesn't pretend otherwise: it
  splits its probability, and both domains get searched with a guaranteed minimum of two chunks
  each, so the less likely domain can't be crowded out."
- "And it found something I did **not** plant. That contradiction between the pump manual and
  the spill procedure is a real inconsistency in my own synthetic documents. That's the pitch
  for a real client: the first thing this does is audit their documentation."
- Open **Router reasoning**: "one sentence of why, and the latency per stage, so we can see
  where the time goes."
- Optional: toggle **Single-domain search** and re-ask the clean question to show it narrows to
  one domain. "That's a cost/coverage trade-off. I set it from data, which you'll see in the
  eval chart."

For a more even route split with no drama, type *Fill weights are drifting low, what should I
check?* (measured: maintenance 0.55 / quality 0.43).

## 2:15 - The conflict (75 s)

**Click:** *Conflict: A bottle fell over on conveyor C2. Can I clear it with the conveyor in jog mode?*

**What you'll see:** route **Safety about 0.6 / Maintenance about 0.35**. The answer opens with
"The documents disagree on...", and an amber warning box shows both sides:

- Maintenance guide (rev A, 2022): a single fallen bottle can be cleared in jog mode at reduced speed.
- Safety procedure (rev B, 2025): clearing jams with the conveyor running or in jog mode is prohibited; full lockout required.
- **Escalate to: Maintenance Supervisor, EHS Lead**, and "Confirm with ... before proceeding."

There may be two warning boxes for the same disagreement (the maintenance guide conflicts with two
different safety chunks). Say so rather than hide it.

**Talking points**
- "This is the point of the whole thing. A normal RAG bot retrieves both passages and writes a
  smooth answer that picks one. Here the model is forbidden from merging or choosing."
- "I know the maintenance guide is older and probably stale. The system doesn't decide that
  for the client. Precedence is a client decision, configured in `PRECEDENCE_RULES`, empty by
  default. With a rule like 'safety outranks maintenance', this box would show the rule that
  resolved it. Without one, it escalates to the document owners."
- "That's where safety is protected: not by biasing the router, but by never hiding a
  contradiction. The cost of the two errors isn't equal, and this handles that explicitly."
- If asked how reliable it is: "Conflict detection is its own LLM pass, tuned for recall.
  Early on a single combined call missed the structured conflict about half the time, which
  is why I split it. Precision is the number I watch in the eval."

Then flip on **Show conflicts log** in the sidebar:

- "Every conflict is logged with both sources, and the owners to contact. That's a
  documentation-quality report: the seal-weep contradiction from the last question is already
  in it. The owners can fix these at the source."

(Spare conflict if there is time: *Cap torque is reading 16 in-lb, is that ok?* Routes 0.87 to
quality, but the floor still pulls in the CX-4 manual, which sets heads to 15-20 in-lb against the
quality spec's 18-22. Escalated to Quality Engineer and Maintenance Supervisor.)

## 3:30 - A question the documents don't answer (30 s)

**Click:** *Not covered: How do I operate a forklift?*

**What you'll see:** a neutral blue info box: "Not covered", with what is in the excerpts and
the kind of document needed (a forklift operating procedure or operator training/certification
requirements). No citations, no invented steps, no conflicts.

**Talking points**
- "It would be easy for a model to give generic forklift advice. On a plant floor, a confident
  wrong answer is worse than no answer. It refuses and tells you what document is missing."
- "Refusal is measured in the eval: how often it refuses when it should, and how often it
  refuses when it shouldn't."

## 4:00 - How we know it works (60 s)

Switch to `eval/results/report.md`.

1. **Summary table:** point at router top-1 and top-2 coverage, retrieval recall@6, conflict
   precision/recall, refusal recall and false-refusal rate, by category (clean / ambiguous /
   conflict / not covered). Quote the numbers you saw when you ran it.
2. **Threshold sweep:** show `threshold_sweep.png`. "Searching only the top domain is cheaper.
   This chart is the price in coverage for each threshold, measured on the golden set. That's
   how I pick the setting instead of guessing." Each question is routed once and the cached
   route is reused, so the sweep is cheap and repeatable.
3. **Failures list:** "The report lists every miss with expected versus actual. For example,
   conflict precision is the weak spot: the detector favours recall, so it sometimes flags a
   borderline case. I'd rather see that in a table than discover it on the floor."

Close with the gaps you'd tackle next: OCR and real document ingestion, a Qdrant server,
per-role access control, the feedback loop from the logs, and a conflict report to document owners
(`README.md`, "Path to production").

---

## Likely questions

| Question | Short answer |
|---|---|
| Why not just one big search across all documents? | Two filtered searches with a floor guarantee the second domain a slot, keep scores comparable, and leave an audit trail of what was searched. |
| Why isn't safety weighted higher? | Routing by relevance keeps the router honest. Unequal costs are handled by surfacing conflicts and by client-set precedence. |
| Why two LLM calls to answer? | A dedicated conflict pass is far more reliable than asking one call to do prose and structure at once; the answer pass is then told what was found. |
| Is every flagged conflict real? | Not always. Detection is tuned for recall, so borderline `scope_difference` flags happen; the eval reports conflict precision. Two of the conflicts it found (seal weeping, PPE face shield) are real inconsistencies in the corpus that were not planted. |
| What stops hallucination? | The model may only use the numbered excerpts, must cite them, must refuse if they don't answer, and invalid citations are dropped in code. |
| What if the documents are scans? | The prototype assumes text PDFs with bookmarks. Production needs OCR and, for messy layouts, a layout model. |
| What happens when a document is revised? | Chunks carry revision and effective date; re-ingest replaces that document's chunks; a `latest_revision` rule can auto-resolve same-document conflicts. |
| How fast is it? | About 2-5 s to route and 7-14 s to answer when there's a conflict (about 3 s for a refusal). Running the two domains' work concurrently and streaming are the next latency wins. |

## If something goes wrong

- **"Connection error" from the SDK:** `pip install -U "brotli>=1.2"` (see README, Troubleshooting).
- **Every answer says "Not covered":** the index is empty; re-run `python -m app.ingest --reset`.
- **Conflict box missing:** re-ask once; the model is not deterministic on borderline wording.
- **Slow first answer:** that's model warm-up; the throwaway question in the setup step avoids it.
- **API outage:** show `eval/results/report.md` and the saved conflict log (`logs/conflicts.jsonl`) and walk through the architecture diagram in the README.
