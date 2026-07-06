from __future__ import annotations

import re

_WORD_NUMBERS: dict[str, float] = {
    "ten": 10,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "quarter": 25,
    "half": 50,
}


def parse_percent(text: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if match:
        return float(match.group(1)) / 100.0
    return None


def parse_word_percent(text: str) -> float | None:
    lower = text.lower()
    if "half" in lower and "percent" not in lower and "%" not in lower:
        if any(w in lower for w in ("level", "gain", "volume", "half")):
            return 0.5
    match = re.search(
        r"\b(" + "|".join(_WORD_NUMBERS) + r")\s+percent\b",
        lower,
    )
    if match:
        return _WORD_NUMBERS[match.group(1)] / 100.0
    return None


def parse_float(text: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))
    return None


def parse_int(text: str) -> int | None:
    match = re.search(r"\b(\d+)\b", text)
    if match:
        return int(match.group(1))
    return None


def parse_channel(text: str) -> int | None:
    ordinals = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
    }
    lower = text.lower()
    for word, num in ordinals.items():
        if f"channel {word}" in lower or f"track {word}" in lower:
            return num
    match = re.search(r"(?:channel|track)\s+(\d+)", lower)
    if match:
        return int(match.group(1))
    return None


def fill_pattern_args(pattern, text: str) -> list | None:
    """Deterministic slot fill from NL for a known pattern (profile-driven)."""
    if not pattern.type_tags:
        return []

    lower = text.lower()

    if pattern.pattern_id == "pan_set" and "center" in lower:
        if "left" not in lower and "right" not in lower:
            return [0.0]

    if len(pattern.type_tags) == 1 and pattern.type_tags == "f":
        for parser in (parse_percent, parse_word_percent):
            pct = parser(text)
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


def normalize_unit_float_args(args: list, pattern) -> list:
    """Scale whole-number percents (e.g. 30) into 0–1 for unit-float patterns."""
    if not pattern.type_tags or pattern.type_tags != "f" or not args:
        return args
    out = list(args)
    for i, slot in enumerate(pattern.slots):
        if slot.type != "float" or i >= len(out):
            continue
        spec = pattern.ranges.get(slot.name)
        if spec is None or spec.max > 1.0:
            continue
        try:
            v = float(out[i])
        except (TypeError, ValueError):
            continue
        if v > 1.0:
            out[i] = v / 100.0
    return out
