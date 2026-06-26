from __future__ import annotations

import pytest

from llm4osc.models import RefusalIntent, SuccessIntent
from llm4osc.profile import find_committed_profile
from llm4osc.resolver import resolve_nl
from tier3.encode import decode_packet, encode_intent


PROFILE = find_committed_profile("max-msp")


def test_encode_gain_float() -> None:
    intent = SuccessIntent(
        device_id="max-msp",
        profile_version=PROFILE.profile_version,
        pattern_id="gain_set",
        address="/gain",
        type_tags="f",
        args=[0.5],
    )
    address, args = decode_packet(encode_intent(intent))
    assert address == "/gain"
    assert args == [0.5]


def test_encode_transport_no_args() -> None:
    intent = SuccessIntent(
        device_id="max-msp",
        profile_version=PROFILE.profile_version,
        pattern_id="transport_start",
        address="/transport/start",
        type_tags="",
        args=[],
    )
    address, args = decode_packet(encode_intent(intent))
    assert address == "/transport/start"
    assert args == []


def test_resolver_volume_percent() -> None:
    result = resolve_nl("set master volume to 30%", PROFILE)
    assert isinstance(result, SuccessIntent)
    assert result.pattern_id == "master_volume"
    assert result.args == [0.3]


def test_resolver_reproducible() -> None:
    a = resolve_nl("set gain to 50%", PROFILE)
    b = resolve_nl("set gain to 50%", PROFILE)
    assert a == b


def test_resolver_ambiguous() -> None:
    result = resolve_nl("start", PROFILE)
    assert isinstance(result, RefusalIntent)
    assert result.reason.value == "ambiguous_pattern"


def test_resolver_unknown() -> None:
    result = resolve_nl("flux capacitor", PROFILE)
    assert isinstance(result, RefusalIntent)
    assert result.reason.value == "unknown_pattern"
