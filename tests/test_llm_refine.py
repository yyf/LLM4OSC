from __future__ import annotations

from llm4osc.llm import _normalize_parsed
from llm4osc.profile import find_committed_profile

PROFILE = find_committed_profile("max-msp")


def test_refine_corrects_pattern_and_slots() -> None:
    data = _normalize_parsed(
        {
            "kind": "intent",
            "pattern_id": "gain_set",
            "args": [0.5],
        },
        PROFILE,
        "kick off playback",
    )
    assert data["pattern_id"] == "transport_start"
    assert data["args"] == []


def test_refine_word_percent_volume() -> None:
    data = _normalize_parsed(
        {
            "kind": "intent",
            "pattern_id": "gain_set",
            "args": [30],
        },
        PROFILE,
        "attenuate output to thirty percent",
    )
    assert data["pattern_id"] == "master_volume"
    assert data["args"] == [0.3]


def test_refine_normalizes_literal_percent() -> None:
    data = _normalize_parsed(
        {
            "kind": "intent",
            "pattern_id": "master_volume",
            "args": [30],
        },
        PROFILE,
        "set master volume to 30%",
    )
    assert data["pattern_id"] == "master_volume"
    assert data["args"] == [0.3]
