from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm4osc.profile import (
    commit_draft,
    import_text_proposals,
    init_draft,
    load_profile,
    save_profile,
    validate_profile,
    drafts_dir,
)
from llm4osc.models import PatternRecord


def test_init_and_validate_draft() -> None:
    profile = init_draft("test-device")
    profile.patterns.append(
        PatternRecord(
            pattern_id="test_ping",
            address="/ping",
            type_tags="",
            description="Ping test pattern",
            tags=["ping"],
        )
    )
    path = drafts_dir() / "test-device.json"
    save_profile(path, profile)
    errors = validate_profile(load_profile(path))
    assert errors == []


def test_import_text_proposals() -> None:
    text = "# excerpt\n/gain float 0.0-1.0\n/mute int"
    proposals = import_text_proposals(text)
    assert len(proposals) == 2
    assert proposals[0]["address"] == "/gain"


def test_commit_draft(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from llm4osc import profile as profile_mod

    draft_root = tmp_path / "profiles"
    monkeypatch.setattr(profile_mod, "drafts_dir", lambda: draft_root / "drafts")
    monkeypatch.setattr(profile_mod, "committed_dir", lambda: draft_root / "committed")

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
    committed = load_profile(out)
    assert committed.device_id == "gadget"
    assert committed.profile_version.startswith("prof_")
