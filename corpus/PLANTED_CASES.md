# Planted Test Cases

Page numbers refer to the PDFs built by `scripts/build_corpus.py`. Re-verify them if the source markdown changes.

## 1. CONFLICT (cross-domain): jam clearing with the conveyor energized

| Side | File | Section | Page | Claim |
|---|---|---|---|---|
| A | `corpus/maintenance/conveyor-maintenance-guide.pdf` (Rev A, 2022-04-05) | Jam Troubleshooting > Clearing Small Jams | 2 | Small jams "can be cleared quickly with the conveyor in jog mode at reduced speed." |
| B | `corpus/safety/conveyor-jam-clearing.pdf` (Rev B, 2025-06-10) | Core Rule | 1 | No body part may enter the guarded zone until full lockout, for ALL jams; "Jog mode, reduced speed, E-stop, and pause buttons are not acceptable substitutes for lockout." |

Expected: conflict reported (kind `contradiction`), both claims cited, escalation to EHS Lead + Maintenance Supervisor. Note the maintenance guide is older (Rev A, 2022), so it is probably stale, but with no precedence rule configured the system must not resolve it.

## 2. CONFLICT (numeric): cap torque

| Side | File | Section | Page | Claim |
|---|---|---|---|---|
| A | `corpus/quality/cap-torque-spec.pdf` (Rev E, 2025-07-01) | Torque Requirements | 1 | 38 mm / 500 mL cap: **18 to 22 in-lb** |
| B | `corpus/maintenance/cx4-capper-manual.pdf` (Rev B, 2023-08-14) | Torque Head Calibration > Calibration Targets | 1 | Set heads to **15 to 20 in-lb** for 38 mm / 500 mL |

Expected: conflict reported, both ranges cited (they overlap only at 18 to 20; 15 to 17.9 would fail the quality spec), escalation to Quality Engineer + Maintenance Supervisor. The quality spec's "Relationship to Capper Settings" section explains a legitimate difference (clutch setting vs finished-package torque), so `scope_difference` is an arguable alternative; the golden set accepts either kind as long as a conflict is flagged.

## 3. AMBIGUOUS questions (legitimately span domains)

| Question | Domains | Expected sources |
|---|---|---|
| Cap torque is reading 16 in-lb, is that ok? | quality + maintenance | cap-torque-spec, cx4-capper-manual |
| A bottle jammed at the capper infeed, how do I clear it? | safety + maintenance | conveyor-jam-clearing, conveyor-maintenance-guide |
| The RP-200 is leaking sanitizer at the shaft. What do I do? | maintenance + safety | rp200-pump-manual (Seal Leak, p. 2), chemical-spill-response (Spills on or Near Equipment, p. 2) |
| Fill weights are drifting low, what should I check? | quality + maintenance | fill-weight-spec, rp200-pump-manual |
| What do I do with product that got chemical on it? | safety + quality | chemical-spill-response, nonconformance-hold |

## 4. NOT COVERED (deliberately absent)

No document mentions these topics; the system must return `not_covered=true` instead of answering from general knowledge.

- "How do I operate a forklift?" (no forklift content anywhere in the corpus)
- "How is overtime pay calculated?" (payroll / HR)
- "What is the vacation request process?" (HR)
- "How do I program the PLC on the labeler?" (no labeler or PLC documentation)

## Example questions for the UI sidebar (Component 7)

1. Clean: "What PPE do I need for sanitizer changeover?"
2. Clean: "How often should the RP-200 seal be replaced?"
3. Ambiguous: "The RP-200 is leaking sanitizer at the shaft. What do I do?"
4. Ambiguous: "Cap torque is reading 16 in-lb, is that ok?"
5. Conflict: "A bottle fell over on conveyor C2. Can I clear it with the conveyor in jog mode?"
6. Not covered: "How do I operate a forklift?"

## 5. UNPLANTED inconsistencies (found by the conflict detector, not on purpose)

These were discovered while testing; they are genuine disagreements between the synthetic documents. Do not label questions about them `expect_conflict: false` in the golden set (or fix the documents and re-run `scripts/build_corpus.py` and `python -m app.ingest --reset`).

| Topic | Side A | Side B | Questions that surface it |
|---|---|---|---|
| Pump with a small seal leak | `rp200-pump-manual.pdf`, Troubleshooting > Seal Leak (p. 2): weeping of a few drops per minute "can continue for a limited time"; only a steady stream needs immediate shutdown | `chemical-spill-response.pdf`, Spills on or Near Equipment (p. 2): if a chemical leaks from the RP-200 pump seal, stop the equipment, with no exception for minor weeping | "The RP-200 is leaking sanitizer at the shaft. What do I do?" (conflict in 2 of 2 runs) |
| Face shield for sanitizer work | `ppe-matrix.pdf`, PPE Matrix (p. 1): sanitizer changeover / CIP requires splash goggles and a face shield | `chemical-spill-response.pdf`, Safe Handling (p. 1): face shield for concentrate only | "What PPE do I need for sanitizer changeover?" (flagged in 2 of 2 runs, once as `scope_difference`) |
