from __future__ import annotations

import re

_STOPWORDS = frozenset(
    {"set", "to", "the", "a", "an", "at", "on", "in", "for", "and", "or", "of"}
)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def score_pattern(text: str, description: str, tags: list[str]) -> int:
    tokens = _tokenize(text) - _STOPWORDS
    if not tokens:
        return 0
    tag_tokens = _tokenize(" ".join(tags))
    desc_tokens = _tokenize(description) - _STOPWORDS
    tag_hits = len(tokens & tag_tokens)
    desc_hits = len(tokens & desc_tokens)
    return tag_hits * 3 + desc_hits


def rank_patterns(
    text: str,
    patterns: list,
) -> list[tuple[object, int]]:
    scored = [
        (p, score_pattern(text, p.description, p.tags))
        for p in patterns
    ]
    return sorted(scored, key=lambda x: x[1], reverse=True)


def patterns_for_context(text: str, patterns: list, k: int = 8) -> list:
    ranked = rank_patterns(text, patterns)
    positive = [p for p, score in ranked if score > 0]
    if positive:
        return positive[:k]
    return [p for p, _ in ranked[:k]]
