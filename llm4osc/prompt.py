from __future__ import annotations

import json
from typing import Any

from llm4osc.models import DeviceProfile, PatternRecord

REFUSAL_EXAMPLE = {
    "schema_version": "1.0",
    "kind": "refusal",
    "reason": "unknown_pattern",
    "message": "No matching pattern.",
}


def system_prompt(profile: DeviceProfile) -> str:
    intent_example = {
        "schema_version": "1.0",
        "kind": "intent",
        "device_id": profile.device_id,
        "profile_version": profile.profile_version,
        "pattern_id": "gain_set",
        "address": "/gain",
        "type_tags": "f",
        "args": [0.5],
    }
    refusal_example = {
        **REFUSAL_EXAMPLE,
        "device_id": profile.device_id,
        "profile_version": profile.profile_version,
    }
    return f"""You map natural language to OSC intent JSON for one device.
Use ONLY patterns listed in CONTEXT. Output a single JSON object only — no prose.
Every response MUST include "kind": "intent" or "kind": "refusal".
If unsure, output kind=refusal with a reason. Never invent addresses.
Refusal "reason" MUST be exactly one of: unknown_device, unknown_pattern,
ambiguous_pattern, out_of_range, missing_slot, unsupported_command, invalid_input.

SUCCESS EXAMPLE:
{json.dumps(intent_example, ensure_ascii=False)}

REFUSAL EXAMPLE:
{json.dumps(refusal_example, ensure_ascii=False)}"""

USER_TEMPLATE = """DEVICE: {device_id}
PROFILE: {profile_version}

CONTEXT (retrieved patterns):
{pattern_block}
{few_shot_block}
USER REQUEST:
{nl}"""


def format_pattern_block(patterns: list[PatternRecord]) -> str:
    lines: list[str] = []
    for p in patterns:
        ranges = {
            name: {"min": spec.min, "max": spec.max}
            for name, spec in p.ranges.items()
        }
        lines.append(
            json.dumps(
                {
                    "pattern_id": p.pattern_id,
                    "address": p.address,
                    "type_tags": p.type_tags,
                    "description": p.description,
                    "ranges": ranges or None,
                    "tags": p.tags,
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines)


def format_few_shot_block(examples: list[dict[str, Any]]) -> str:
    if not examples:
        return ""
    lines = ["FEW-SHOT EXAMPLES:"]
    for ex in examples:
        lines.append(f"USER: {ex['nl']}")
        lines.append(f"ASSISTANT: {json.dumps(ex['intent'], ensure_ascii=False)}")
    return "\n".join(lines) + "\n\n"


def build_messages(
    nl: str,
    profile: DeviceProfile,
    patterns: list[PatternRecord],
    *,
    few_shot_examples: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    user_content = USER_TEMPLATE.format(
        device_id=profile.device_id,
        profile_version=profile.profile_version,
        pattern_block=format_pattern_block(patterns),
        few_shot_block=format_few_shot_block(few_shot_examples or []),
        nl=nl,
    )
    return [
        {"role": "system", "content": system_prompt(profile)},
        {"role": "user", "content": user_content},
    ]
