"""Client-configured conflict precedence. Default (no rules): never resolve, always escalate."""
from __future__ import annotations

from dataclasses import replace

import config
from app.generate import Conflict
from app.retrieve import Chunk


def _escalation(a: Chunk, b: Chunk) -> list[str]:
    """Owners of both documents (chunk owner field, else the domain's configured owner), deduped."""
    owners: list[str] = []
    for c in (a, b):
        o = c.owner or config.DOMAIN_OWNERS.get(c.domain, "")
        if o and o not in owners:
            owners.append(o)
    return owners


def _latest_revision(a: Chunk, b: Chunk, ca: int, cb: int, rule: dict):
    same_doc = (a.source_path and a.source_path == b.source_path) or a.doc == b.doc
    if not same_doc or a.effective_date == b.effective_date:
        return None
    win, lose, wc = (a, b, ca) if a.effective_date > b.effective_date else (b, a, cb)
    text = (f"{win.doc} rev {win.revision} (effective {win.effective_date}) supersedes "
            f"rev {lose.revision} (effective {lose.effective_date})")
    return wc, text


def _domain_order(a: Chunk, b: Chunk, ca: int, cb: int, rule: dict):
    order = rule.get("order", [])
    if a.domain == b.domain or a.domain not in order or b.domain not in order:
        return None
    win, lose, wc = (a, b, ca) if order.index(a.domain) < order.index(b.domain) else (b, a, cb)
    return wc, f"{win.domain} takes precedence over {lose.domain} ({win.doc} governs)"


RULES = {"latest_revision": _latest_revision, "domain_order": _domain_order}


def apply_precedence(conflicts: list[Conflict], chunks: list[Chunk], rules: list[dict] | None = None) -> list[Conflict]:
    """Return annotated copies. First matching rule wins; otherwise resolution=None and escalate_to is set."""
    rules = config.PRECEDENCE_RULES if rules is None else rules
    out: list[Conflict] = []
    for c in conflicts:
        a, b = chunks[c.cite_a - 1], chunks[c.cite_b - 1]
        resolved = None
        for rule in rules:
            fn = RULES.get(rule.get("type"))
            if fn is None:
                raise ValueError(f"unknown precedence rule type: {rule.get('type')!r}")
            hit = fn(a, b, c.cite_a, c.cite_b, rule)
            if hit:
                resolved = (rule["type"], *hit)
                break
        if resolved:
            rtype, winner, text = resolved
            out.append(replace(c, resolution=text, rule=rtype, winner_cite=winner, escalate_to=[]))
        else:
            out.append(replace(c, resolution=None, rule=None, winner_cite=None, escalate_to=_escalation(a, b)))
    return out
