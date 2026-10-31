"""Canned ask() used only when app.pipeline is not importable yet (UI development)."""
from __future__ import annotations

from types import SimpleNamespace as NS


def _chunk(domain, doc, section, ps, pe, rev, date, owner, text, score):
    return NS(id=f"{doc}-{section}", domain=domain, doc=doc, section=section,
              page_start=ps, page_end=pe, revision=rev, effective_date=date,
              owner=owner, text=text, score=score)


def ask(question: str, *, single_domain_threshold: float | None = None) -> NS:
    q = question.lower()
    a = _chunk("maintenance", "Conveyor Maintenance Guide", "Jam Troubleshooting > Clearing Small Jams",
               2, 2, "A", "2022-04-05", "Maintenance Supervisor",
               "Small jams can be cleared quickly with the conveyor in jog mode at reduced speed.", 0.71)
    b = _chunk("safety", "Conveyor Jam Clearing", "Core Rule", 1, 1, "B", "2025-06-10", "EHS Lead",
               "No body part may enter the guarded zone until full lockout. Jog mode is not a substitute.", 0.69)
    base = dict(question=question, searched_domains=["safety", "maintenance"], citations_used=[1, 2],
                latency_ms={"route": 1, "retrieve": 1, "generate": 1})
    if "forklift" in q:
        return NS(**{**base, "citations_used": []}, chunks=[a], conflicts=[], not_covered=True,
                  answer="Not covered. A forklift operation procedure would be needed.",
                  route=NS(**dict(scores={"safety": .5, "maintenance": .4, "quality": .1},
                             reasoning="[stub]", primary="safety", confidence=.5, margin=.1)))
    conflicts = []
    if "jog" in q or "jam" in q:
        conflicts = [NS(topic="Clearing jams with conveyor energized", claim_a=a.text, cite_a=1,
                        claim_b=b.text, cite_b=2, kind="contradiction", resolution=None, rule=None,
                        escalate_to=["EHS Lead", "Maintenance Supervisor"])]
    answer = ("[stub] The documents disagree on jam clearing: [1] vs [2]. Confirm with EHS Lead."
              if conflicts else "[stub] Example answer citing [1] and [2].")
    return NS(**base, chunks=[a, b], conflicts=conflicts, not_covered=False, answer=answer,
              route=NS(**dict(scores={"safety": .45, "maintenance": .45, "quality": .1},
                         reasoning="[stub] spans safety and maintenance", primary="safety",
                         confidence=.45, margin=0.0)))
