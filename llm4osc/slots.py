from __future__ import annotations

import re


def parse_percent(text: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if match:
        return float(match.group(1)) / 100.0
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
