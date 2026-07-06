from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm4osc.scorecard import compare_track_c, score

GOLDEN_PARAPHRASE_DIR = Path(__file__).parent / "golden_nl_paraphrase"


def _load_cases(directory: Path):
    for path in sorted(directory.glob("*.json")):
        yield path.stem, json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case_id,payload", list(_load_cases(GOLDEN_PARAPHRASE_DIR)))
def test_paraphrase_fixture_shape(case_id: str, payload: dict) -> None:
    assert payload["id"] == case_id
    assert "literal_id" in payload
    assert payload["expect"]["kind"] == "intent"
    assert "pattern_id" in payload["expect"]


def test_b0_paraphrase_passes_hero_suite() -> None:
    """B0 passes the hero paraphrase suite with profile tags + slot parsers."""
    report = score("max-msp", backend="b0", suite="paraphrase")
    assert report["counts"]["nl_cases"] == 8
    assert report["metrics"]["semantic_accuracy"] >= 0.9
    assert report["metrics"]["wrong_send_rate"] == 0.0


def test_track_c_compare_structure() -> None:
    report = compare_track_c("max-msp", backends=("b0",))
    assert "gaps" in report
    assert "b0" in report["gaps"]
    assert report["backends"]["b0"]["literal"]["gates"]["passed"] is True
    assert "recommendation" in report
