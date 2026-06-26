from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llm4osc.models import RefusalIntent, SuccessIntent
from llm4osc.profile import find_committed_profile
from llm4osc.resolver import resolve_nl
from tier3.pipeline import run_pipeline
from tier3.validate import ValidationError

GOLDEN_NL_DIR = Path(__file__).parent / "golden_nl"
GOLDEN_REFUSAL_DIR = Path(__file__).parent / "golden_refusal"
PROFILE = find_committed_profile("max-msp")


def _load_cases(directory: Path):
    for path in sorted(directory.glob("*.json")):
        yield path.stem, json.loads(path.read_text(encoding="utf-8"))


def _args_match(got: list[Any], expect: list[Any]) -> bool:
    if any(isinstance(v, float) for v in expect):
        return got == pytest.approx(expect)
    return got == expect


def _intent_matches(result: SuccessIntent, expect: dict) -> bool:
    if result.pattern_id != expect["pattern_id"]:
        return False
    if "address" in expect and result.address != expect["address"]:
        return False
    if "type_tags" in expect and result.type_tags != expect["type_tags"]:
        return False
    if "args" in expect and not _args_match(result.args, expect["args"]):
        return False
    return True


@pytest.mark.parametrize("case_id,payload", list(_load_cases(GOLDEN_NL_DIR)))
def test_golden_nl_resolver(case_id: str, payload: dict) -> None:
    result = resolve_nl(payload["nl"], PROFILE)
    expect = payload["expect"]
    assert expect["kind"] == "intent"
    assert isinstance(result, SuccessIntent)
    assert _intent_matches(result, expect)


@pytest.mark.parametrize("case_id,payload", list(_load_cases(GOLDEN_REFUSAL_DIR)))
def test_golden_refusal(case_id: str, payload: dict) -> None:
    result = resolve_nl(payload["nl"], PROFILE)
    expect = payload["expect"]
    assert isinstance(result, RefusalIntent)
    assert result.reason.value == expect["reason"]


@pytest.mark.parametrize("case_id,payload", list(_load_cases(GOLDEN_NL_DIR)))
def test_golden_nl_tier3_dry_run(case_id: str, payload: dict) -> None:
    result = resolve_nl(payload["nl"], PROFILE)
    assert isinstance(result, SuccessIntent)
    pipeline = run_pipeline(result.model_dump(mode="json"), PROFILE, dry_run=True)
    expect = payload.get("expect_osc") or {
        "address": payload["expect"].get("address", result.address),
        "args": payload["expect"]["args"],
    }
    assert pipeline.preview.address == expect["address"]
    assert pipeline.preview.args == pytest.approx(expect["args"])


def test_refusal_never_sends() -> None:
    for _, payload in _load_cases(GOLDEN_REFUSAL_DIR):
        result = resolve_nl(payload["nl"], PROFILE)
        assert isinstance(result, RefusalIntent)
        with pytest.raises(ValidationError):
            run_pipeline(result.model_dump(mode="json"), PROFILE, dry_run=True)
