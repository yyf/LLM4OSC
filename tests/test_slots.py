from __future__ import annotations

from llm4osc.profile import find_committed_profile
from llm4osc.slots import (
    fill_pattern_args,
    normalize_unit_float_args,
    parse_word_percent,
)

PROFILE = find_committed_profile("max-msp")


def _pattern(pattern_id: str):
    return next(p for p in PROFILE.patterns if p.pattern_id == pattern_id)


def test_parse_word_percent() -> None:
    assert parse_word_percent("attenuate output to thirty percent") == 0.3
    assert parse_word_percent("make the level half") == 0.5


def test_fill_pattern_args_volume_words() -> None:
    args = fill_pattern_args(_pattern("master_volume"), "attenuate output to thirty percent")
    assert args == [0.3]


def test_fill_pattern_args_pan_center() -> None:
    args = fill_pattern_args(_pattern("pan_set"), "center the stereo image")
    assert args == [0.0]


def test_fill_pattern_args_transport() -> None:
    args = fill_pattern_args(_pattern("transport_start"), "kick off playback")
    assert args == []


def test_normalize_unit_float_args() -> None:
    out = normalize_unit_float_args([30], _pattern("master_volume"))
    assert out == [0.3]
