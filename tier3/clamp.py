from __future__ import annotations

from copy import deepcopy

from llm4osc.models import DeviceProfile, SuccessIntent


def clamp_intent(intent: SuccessIntent, profile: DeviceProfile) -> SuccessIntent:
    pattern = next(p for p in profile.patterns if p.pattern_id == intent.pattern_id)
    clamped_args = list(intent.args)

    for slot in pattern.slots:
        range_spec = pattern.ranges.get(slot.name)
        if range_spec is None:
            continue
        idx = slot.arg_index
        value = float(clamped_args[idx])
        clamped = max(range_spec.min, min(range_spec.max, value))
        if slot.type == "int":
            clamped_args[idx] = int(round(clamped))
        else:
            clamped_args[idx] = clamped

    result = deepcopy(intent)
    result.args = clamped_args
    return result
