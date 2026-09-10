from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm4osc.acceptance import (
    AcceptanceError,
    case_counts,
    ensure_suite,
    pin_case,
    pin_from_result,
    run_acceptance,
)
from llm4osc.models import PatternRecord, RefusalIntent, RefusalReason, SuccessIntent
from llm4osc.profile import (
    commit_draft,
    find_committed_profile,
    init_draft,
    save_profile,
)


def test_ensure_seeds_max(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from llm4osc import acceptance as acc
    from llm4osc import profile as profile_mod

    monkeypatch.setattr(profile_mod, "profiles_dir", lambda: tmp_path / "profiles")
    monkeypatch.setattr(acc, "profiles_dir", lambda: tmp_path / "profiles")
    # Point find_committed at real repo profile via monkeypatch of committed only —
    # use real find by copying max profile into tmp committed.
    src = profile_mod.repo_root() / "profiles" / "committed"
    dest = tmp_path / "profiles" / "committed"
    dest.mkdir(parents=True)
    for path in src.glob("max-msp*.json"):
        (dest / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    meta = ensure_suite("max-msp")
    assert meta["device_id"] == "max-msp"
    counts = case_counts("max-msp")
    assert counts["nl_cases"] >= 8
    assert counts["refusal_cases"] >= 4


def test_pin_and_run_acceptance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from llm4osc import acceptance as acc
    from llm4osc import profile as profile_mod

    monkeypatch.setattr(profile_mod, "profiles_dir", lambda: tmp_path / "profiles")
    monkeypatch.setattr(acc, "profiles_dir", lambda: tmp_path / "profiles")
    src = profile_mod.repo_root() / "profiles" / "committed"
    dest = tmp_path / "profiles" / "committed"
    dest.mkdir(parents=True)
    for path in src.glob("max-msp*.json"):
        (dest / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    profile = find_committed_profile("max-msp")
    ensure_suite("max-msp", profile=profile, seed_from_benchmarks=True)
    path = pin_case(
        "max-msp",
        "set gain to 50%",
        {
            "kind": "intent",
            "pattern_id": "gain_set",
            "address": "/gain",
            "args": [0.5],
        },
        profile=profile,
    )
    assert path.exists()
    report = run_acceptance("max-msp", profile=profile, backend="b0")
    assert report["gates"]["passed"] is True
    assert report["acceptance"] is True


def test_commit_skips_gate_without_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm4osc import profile as profile_mod

    draft_root = tmp_path / "profiles"
    monkeypatch.setattr(profile_mod, "drafts_dir", lambda: draft_root / "drafts")
    monkeypatch.setattr(profile_mod, "committed_dir", lambda: draft_root / "committed")
    monkeypatch.setattr(profile_mod, "profiles_dir", lambda: draft_root)

    profile = init_draft("gadget")
    profile.patterns.append(
        PatternRecord(
            pattern_id="on",
            address="/on",
            type_tags="",
            description="Turn on",
            tags=["on"],
        )
    )
    draft = draft_root / "drafts" / "gadget.json"
    save_profile(draft, profile)
    out = commit_draft(draft)
    assert out.exists()


def test_pin_refuse_from_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from llm4osc import acceptance as acc
    from llm4osc import profile as profile_mod

    monkeypatch.setattr(profile_mod, "profiles_dir", lambda: tmp_path / "profiles")
    monkeypatch.setattr(acc, "profiles_dir", lambda: tmp_path / "profiles")
    src = profile_mod.repo_root() / "profiles" / "committed"
    dest = tmp_path / "profiles" / "committed"
    dest.mkdir(parents=True)
    for path in src.glob("max-msp*.json"):
        (dest / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    profile = find_committed_profile("max-msp")
    result = RefusalIntent(
        device_id="max-msp",
        profile_version=profile.profile_version,
        reason=RefusalReason.UNKNOWN_PATTERN,
        message="nope",
    )
    path = pin_from_result(
        "max-msp",
        "flux capacitor to 88 mph",
        result,
        profile=profile,
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["expect"]["kind"] == "refusal"
