from __future__ import annotations

import pytest

from llm4osc.models import SuccessIntent
from llm4osc.profile import find_committed_profile
from tier3.validate import ValidationError, validate_intent


PROFILE = find_committed_profile("max-msp")


def test_validate_ok() -> None:
    intent = SuccessIntent(
        device_id="max-msp",
        profile_version=PROFILE.profile_version,
        pattern_id="gain_set",
        address="/gain",
        type_tags="f",
        args=[0.25],
    )
    validate_intent(intent, PROFILE)


def test_validate_wrong_address() -> None:
    intent = SuccessIntent(
        device_id="max-msp",
        profile_version=PROFILE.profile_version,
        pattern_id="gain_set",
        address="/wrong",
        type_tags="f",
        args=[0.25],
    )
    with pytest.raises(ValidationError):
        validate_intent(intent, PROFILE)


def test_clamp_out_of_range() -> None:
    from tier3.clamp import clamp_intent

    intent = SuccessIntent(
        device_id="max-msp",
        profile_version=PROFILE.profile_version,
        pattern_id="gain_set",
        address="/gain",
        type_tags="f",
        args=[2.0],
    )
    clamped = clamp_intent(intent, PROFILE)
    assert clamped.args == [1.0]
