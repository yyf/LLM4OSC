from __future__ import annotations

from typing import Literal

from llm4osc.llm import IntentLLM, default_adapter_path, resolve_nl_llm
from llm4osc.models import (
    DeviceProfile,
    RefusalIntent,
    RefusalReason,
    SuccessIntent,
)
from llm4osc.retrieval import rank_patterns
from llm4osc.slots import fill_pattern_args

Backend = Literal["b0", "b1", "b2", "b3"]


def _fill_args(pattern, text: str) -> list | None:
    return fill_pattern_args(pattern, text)


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
    adapter_path: str | None = None,
    serve_url: str | None = None,
) -> SuccessIntent | RefusalIntent:
    if backend == "b0":
        return resolve_nl_b0(text, profile)

    from llm4osc.serve import resolve_remote, serve_url as _serve_url

    url = _serve_url(serve_url)
    if url and llm is None:
        return resolve_remote(
            url,
            text,
            profile.device_id,
            backend=backend,
            model_id=model_id,
            adapter_path=adapter_path,
        )

    resolved_adapter = adapter_path
    if backend == "b3" and resolved_adapter is None:
        default = default_adapter_path()
        if default.is_dir():
            resolved_adapter = str(default)

    return resolve_nl_llm(
        text,
        profile,
        few_shot=(backend == "b2"),
        llm=llm,
        model_id=model_id,
        adapter_path=resolved_adapter,
    )
