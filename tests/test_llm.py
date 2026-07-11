from __future__ import annotations

import json
import os

import pytest

from llm4osc.llm import (
    _normalize_parsed,
    apply_retrieval_gate,
    extract_json_object,
    load_few_shot_examples,
    resolve_nl_llm,
)
from llm4osc.models import RefusalIntent, SuccessIntent
from llm4osc.profile import find_committed_profile
from llm4osc.prompt import build_messages, format_pattern_block


PROFILE = find_committed_profile("max-msp")


class MockLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls += 1
        return self.response


def test_extract_json_from_markdown_fence() -> None:
    raw = '```json\n{"kind": "intent", "pattern_id": "gain_set"}\n```'
    data = extract_json_object(raw)
    assert data["pattern_id"] == "gain_set"


def test_build_messages_includes_patterns() -> None:
    messages = build_messages("set gain to 50%", PROFILE, PROFILE.patterns[:2])
    assert messages[0]["role"] == "system"
    system = messages[0]["content"]
    assert '"kind": "intent"' in system
    user = messages[1]["content"]
    assert "gain_set" in user
    assert "set gain to 50%" in user


def test_format_pattern_block_json_lines() -> None:
    block = format_pattern_block(PROFILE.patterns[:1])
    row = json.loads(block.splitlines()[0])
    assert row["pattern_id"] == "gain_set"


def test_infer_kind_from_pattern_id() -> None:
    data = _normalize_parsed(
        {"pattern_id": "gain_set", "args": [0.5]},
        PROFILE,
    )
    assert data["kind"] == "intent"
    assert data["address"] == "/gain"
    assert data["type_tags"] == "f"


def test_infer_kind_refusal() -> None:
    data = _normalize_parsed(
        {"reason": "unknown_pattern", "message": "nope"},
        PROFILE,
    )
    assert data["kind"] == "refusal"


def test_normalize_refusal_reason_alias() -> None:
    data = _normalize_parsed(
        {
            "kind": "refusal",
            "reason": "no_matching_pattern",
            "message": "No matching pattern.",
        },
        PROFILE,
    )
    result = RefusalIntent.model_validate(data)
    assert result.reason.value == "unknown_pattern"


def test_resolve_nl_llm_accepts_refusal_alias() -> None:
    raw = json.dumps(
        {
            "kind": "refusal",
            "reason": "no_matching_pattern",
            "message": "No matching pattern.",
        }
    )
    llm = MockLLM(raw)
    result = resolve_nl_llm("flux capacitor", PROFILE, llm=llm)
    assert isinstance(result, RefusalIntent)
    assert result.reason.value == "unknown_pattern"
    assert llm.calls == 1


def test_resolve_nl_llm_without_kind_field() -> None:
    partial = json.dumps(
        {
            "pattern_id": "gain_set",
            "args": [0.5],
        }
    )
    llm = MockLLM(partial)
    result = resolve_nl_llm("set gain to half", PROFILE, llm=llm)
    assert isinstance(result, SuccessIntent)
    assert result.pattern_id == "gain_set"
    assert result.address == "/gain"
    assert result.args == [0.5]


def test_resolve_nl_llm_success() -> None:
    intent_json = json.dumps(
        {
            "schema_version": "1.0",
            "kind": "intent",
            "device_id": "max-msp",
            "profile_version": PROFILE.profile_version,
            "pattern_id": "gain_set",
            "address": "/gain",
            "type_tags": "f",
            "args": [0.5],
        }
    )
    llm = MockLLM(intent_json)
    result = resolve_nl_llm("set gain to half", PROFILE, llm=llm)
    assert isinstance(result, SuccessIntent)
    assert result.pattern_id == "gain_set"
    assert result.args == [0.5]
    assert llm.calls == 1


def test_resolve_nl_llm_retries_on_bad_json() -> None:
    good = json.dumps(
        {
            "schema_version": "1.0",
            "kind": "intent",
            "device_id": "max-msp",
            "profile_version": PROFILE.profile_version,
            "pattern_id": "transport_start",
            "address": "/transport/start",
            "type_tags": "",
            "args": [],
        }
    )

    class FlakyLLM:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages: list[dict[str, str]]) -> str:
            self.calls += 1
            return "not json" if self.calls == 1 else good

    llm = FlakyLLM()
    result = resolve_nl_llm("play", PROFILE, llm=llm)
    assert isinstance(result, SuccessIntent)
    assert llm.calls == 2


def test_few_shot_examples_load() -> None:
    examples = load_few_shot_examples("max-msp")
    assert len(examples) >= 2


def test_retrieval_gate_oov_refuses_hallucinated_intent() -> None:
    raw = json.dumps(
        {
            "kind": "intent",
            "pattern_id": "gain_set",
            "address": "/gain",
            "type_tags": "f",
            "args": [0.03],
        }
    )
    llm = MockLLM(raw)
    result = resolve_nl_llm("boost the bass band by 3db", PROFILE, llm=llm)
    assert isinstance(result, RefusalIntent)
    assert result.reason.value == "unknown_pattern"


def test_retrieval_gate_ambiguous_start() -> None:
    raw = json.dumps(
        {
            "kind": "intent",
            "pattern_id": "transport_start",
            "address": "/transport/start",
            "type_tags": "",
            "args": [],
        }
    )
    llm = MockLLM(raw)
    result = resolve_nl_llm("start", PROFILE, llm=llm)
    assert isinstance(result, RefusalIntent)
    assert result.reason.value == "ambiguous_pattern"


def test_retrieval_gate_missing_slot() -> None:
    raw = json.dumps(
        {
            "kind": "intent",
            "pattern_id": "gain_set",
            "address": "/gain",
            "type_tags": "f",
            "args": [0.5],
        }
    )
    llm = MockLLM(raw)
    result = resolve_nl_llm("set gain", PROFILE, llm=llm)
    assert isinstance(result, RefusalIntent)
    assert result.reason.value == "missing_slot"


def test_apply_retrieval_gate_unknown_pattern_dict() -> None:
    data = apply_retrieval_gate(
        {
            "kind": "intent",
            "pattern_id": "gain_set",
            "args": [0.5],
        },
        PROFILE,
        "flux capacitor to 88 mph",
    )
    assert data["kind"] == "refusal"
    assert data["reason"] == "unknown_pattern"


@pytest.mark.llm
@pytest.mark.skipif(
    os.environ.get("LLM4OSC_RUN_LLM") != "1",
    reason="Set LLM4OSC_RUN_LLM=1 to run live Qwen tests",
)
def test_live_qwen_gain() -> None:
    from llm4osc.resolver import resolve_nl

    result = resolve_nl("set gain to 40%", PROFILE, backend="b1")
    assert isinstance(result, SuccessIntent)
    assert result.pattern_id == "gain_set"
    assert result.args == pytest.approx([0.4])
