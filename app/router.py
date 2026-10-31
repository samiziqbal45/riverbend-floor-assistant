"""LLM router: scores every domain by relevance only (no safety bias)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import config
from app import clients


class RouterError(RuntimeError):
    pass


ROUTE_TOOL = {
    "name": "route",
    "description": "Score how relevant each documentation domain is to the question.",
    "input_schema": {
        "type": "object",
        "properties": {
            **{d: {"type": "number", "minimum": 0, "maximum": 1} for d in config.DOMAINS},
            "reasoning": {"type": "string", "description": "One sentence."},
        },
        "required": [*config.DOMAINS, "reasoning"],
    },
}

SYSTEM_PROMPT = """You route manufacturing floor questions to documentation domains.
Domains:
- safety: hazard controls, lockout/tagout, PPE, emergency and incident response
- maintenance: equipment operation, repair, troubleshooting, preventive maintenance, parts
- quality: specifications, tolerances, inspection, SPC, nonconformance and holds

Score by relevance only. Do not favor any domain by default. If a question spans
domains, split probability between them. Give each domain a score from 0 to 1
(roughly a probability; they will be normalized) and one sentence of reasoning.

You MUST respond by calling the `route` tool exactly once. Do not reply with plain text.

Examples:
{examples}"""


@dataclass
class Route:
    scores: dict[str, float]
    reasoning: str

    @property
    def ranked(self) -> list[str]:
        return sorted(self.scores, key=lambda d: self.scores[d], reverse=True)

    @property
    def primary(self) -> str:
        return self.ranked[0]

    @property
    def margin(self) -> float:
        s = sorted(self.scores.values(), reverse=True)
        return s[0] - s[1]


def normalize(raw: dict) -> dict[str, float]:
    vals = {d: max(0.0, float(raw.get(d, 0.0))) for d in config.DOMAINS}
    total = sum(vals.values())
    if total <= 0:
        return {d: 1 / len(config.DOMAINS) for d in config.DOMAINS}
    return {d: v / total for d, v in vals.items()}


def load_fewshot(path=None) -> list[dict]:
    with open(path or config.FEWSHOT_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@lru_cache(maxsize=1)
def _system_prompt() -> str:
    lines = []
    for ex in load_fewshot():
        scores = ", ".join(f"{d}={ex['scores'][d]}" for d in config.DOMAINS)
        lines.append(f'Q: {ex["question"]}\n-> {scores}; {ex["reasoning"]}')
    return SYSTEM_PROMPT.format(examples="\n\n".join(lines))


def _call(question: str) -> dict:
    last: Exception | None = None
    for _ in range(config.LLM_RETRIES + 1):
        try:
            resp = clients.get_llm().messages.create(
                model=config.LLM_MODEL, max_tokens=config.ROUTER_MAX_TOKENS, system=_system_prompt(),
                tools=[ROUTE_TOOL], tool_choice={"type": "auto"},  # forced tool use isn't supported by the model
                messages=[{"role": "user", "content": f"Question: {question}"}],
            )
            block = next((b for b in resp.content if b.type == "tool_use"), None)
            if block is None:
                raise ValueError("model replied without calling the `route` tool")
            return dict(block.input)
        except Exception as e:  # noqa: BLE001 - SDK raises many types
            last = e
    raise RouterError(f"routing failed after {config.LLM_RETRIES + 1} attempts: {last}") from last


def route(question: str) -> Route:
    out = _call(question)
    return Route(normalize(out), str(out.get("reasoning", "")))
