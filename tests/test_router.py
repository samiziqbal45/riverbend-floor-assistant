from types import SimpleNamespace as NS

import pytest

from app import clients, router


class FakeLLM:
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), 0
        self.messages = self

    def create(self, **kw):
        self.calls += 1
        self.kwargs = kw
        o = self.outputs.pop(0)
        if isinstance(o, Exception):
            raise o
        if o == "TEXT":
            return NS(content=[NS(type="text", text="plain reply")])
        return NS(content=[NS(type="tool_use", input=o)])


@pytest.fixture
def llm(monkeypatch):
    def install(outputs):
        fake = FakeLLM(outputs)
        monkeypatch.setattr(clients, "get_llm", lambda: fake)
        return fake
    return install


def test_normalizes_and_ranks(llm):
    llm([{"safety": 2, "maintenance": 6, "quality": 2, "reasoning": "r"}])
    r = router.route("q")
    assert sum(r.scores.values()) == pytest.approx(1)
    assert r.ranked == ["maintenance", "safety", "quality"] or r.ranked[0] == "maintenance"
    assert r.primary == "maintenance"
    assert r.margin == pytest.approx(0.4)


def test_all_zero_is_uniform(llm):
    llm([{"safety": 0, "maintenance": 0, "quality": 0, "reasoning": ""}])
    r = router.route("q")
    assert all(v == pytest.approx(1 / 3) for v in r.scores.values())
    assert r.margin == pytest.approx(0)


def test_retry_then_success(llm):
    fake = llm([RuntimeError("boom"), {"safety": 1, "maintenance": 0, "quality": 0, "reasoning": ""}])
    assert router.route("q").primary == "safety"
    assert fake.calls == 2


def test_retry_exhausted_raises(llm):
    fake = llm([RuntimeError("a"), RuntimeError("b")])
    with pytest.raises(router.RouterError):
        router.route("q")
    assert fake.calls == 2


def test_fewshot_loads_and_prompt_has_no_bias_instruction():
    assert len(router.load_fewshot()) == 6 + 2
    assert "Do not favor any domain" in router.SYSTEM_PROMPT


def test_uses_auto_tool_choice_and_prompt_demands_tool_call(llm):
    fake = llm([{"safety": 1, "maintenance": 0, "quality": 0, "reasoning": ""}])
    router.route("q")
    assert fake.kwargs["tool_choice"] == {"type": "auto"}  # forced tool use is rejected by the model
    assert fake.kwargs["tools"] == [router.ROUTE_TOOL]
    assert "MUST respond by calling the `route` tool" in fake.kwargs["system"]


def test_text_only_reply_is_retried_then_raises(llm):
    fake = llm(["TEXT", "TEXT"])
    with pytest.raises(router.RouterError, match="without calling"):
        router.route("q")
    assert fake.calls == 2
