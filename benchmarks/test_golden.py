from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm4osc.models import SuccessIntent
from llm4osc.profile import find_committed_profile, load_profile
from tier3.encode import decode_packet, encode_intent
from tier3.pipeline import run_pipeline

GOLDEN_DIR = Path(__file__).parent / "golden"
PROFILE = find_committed_profile("max-msp")


def _cases():
    for path in sorted(GOLDEN_DIR.glob("*.json")):
        yield path.stem, json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case_id,payload", list(_cases()))
def test_golden_intent_to_osc(case_id: str, payload: dict) -> None:
    if payload.get("pipeline_only"):
        pytest.skip("clamp case — pipeline only")

    intent_data = payload["intent"]
    expect = payload["expect"]

    intent = SuccessIntent.model_validate(intent_data)
    dgram = encode_intent(intent)
    address, args = decode_packet(dgram)

    assert address == expect["address"]
    if any(isinstance(a, float) for a in expect["args"]):
        assert args == pytest.approx(expect["args"])
    else:
        assert args == expect["args"]


@pytest.mark.parametrize("case_id,payload", list(_cases()))
def test_golden_tier3_pipeline_dry_run(case_id: str, payload: dict) -> None:
    result = run_pipeline(payload["intent"], PROFILE, dry_run=True)
    assert result.preview.address == payload["expect"]["address"]
    assert result.preview.args == pytest.approx(payload["expect"]["args"])
