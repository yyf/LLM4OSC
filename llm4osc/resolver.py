from __future__ import annotations

from typing import Literal

from llm4osc.llm import IntentLLM, resolve_nl_llm
from llm4osc.models import (
    DeviceProfile,
    RefusalIntent,
    RefusalReason,
    SuccessIntent,
)
from llm4osc.retrieval import rank_patterns
from llm4osc.slots import parse_float, parse_percent

Backend = Literal["b0", "b1", "b2"]


def _fill_args(pattern, text: str) -> list | None:
    if not pattern.type_tags:
        return []
    if len(pattern.type_tags) == 1 and pattern.type_tags == "f":
        pct = parse_percent(text)
        if pct is not None:
            return [pct]
        val = parse_float(text)
        if val is not None:
            slot_name = pattern.slots[0].name if pattern.slots else None
            range_spec = pattern.ranges.get(slot_name) if slot_name else None
            if val > 1.0 and "%" not in text and (range_spec is None or range_spec.max <= 1.0):
                return [val / 100.0]
            return [val]
        return None
    if len(pattern.type_tags) == 1 and pattern.type_tags == "i":
        val = parse_float(text)
        if val is not None:
            return [int(val)]
        return None
    return None


def resolve_nl_b0(text: str, profile: DeviceProfile) -> SuccessIntent | RefusalIntent:
    ranked = rank_patterns(text, profile.patterns)
    if not ranked or ranked[0][1] == 0:
        return RefusalIntent(
            device_id=profile.device_id,
            profile_version=profile.profile_version,
            reason=RefusalReason.UNKNOWN_PATTERN,
            message="No matching control pattern found.",
        )

    top_score = ranked[0][1]
    winners = [p for p, s in ranked if s == top_score]
    if len(winners) > 1:
        return RefusalIntent(
            device_id=profile.device_id,
            profile_version=profile.profile_version,
            reason=RefusalReason.AMBIGUOUS_PATTERN,
            message="Multiple patterns match. Be more specific.",
            candidates=[f"pattern_id:{p.pattern_id}" for p in winners],
        )

    pattern = winners[0]
    args = _fill_args(pattern, text)
    if args is None and pattern.type_tags:
        return RefusalIntent(
            device_id=profile.device_id,
            profile_version=profile.profile_version,
            reason=RefusalReason.MISSING_SLOT,
            message="Could not extract required parameter values from text.",
        )

    return SuccessIntent(
        device_id=profile.device_id,
        profile_version=profile.profile_version,
        pattern_id=pattern.pattern_id,
        address=pattern.address,
        type_tags=pattern.type_tags,
        args=args or [],
    )


def resolve_nl(
    text: str,
    profile: DeviceProfile,
    *,
    backend: Backend = "b0",
    llm: IntentLLM | None = None,
    model_id: str | None = None,
) -> SuccessIntent | RefusalIntent:
    if backend == "b0":
        return resolve_nl_b0(text, profile)
    return resolve_nl_llm(
        text,
        profile,
        few_shot=(backend == "b2"),
        llm=llm,
        model_id=model_id,
    )
