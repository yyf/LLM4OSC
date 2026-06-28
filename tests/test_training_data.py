from __future__ import annotations

import json

import pytest

from llm4osc.profile import find_committed_profile
from llm4osc.training_data import (
    _load_held_out_nl,
    build_training_rows,
    generate_dataset,
)


def test_held_out_excludes_golden_nl() -> None:
    held = _load_held_out_nl()
    assert len(held) >= 16
    rows = build_training_rows(find_committed_profile("max-msp"))
    train_nl = {r["nl"].strip().lower() for r in rows}
    overlap = train_nl & held
    assert overlap == set(), f"golden NL leaked into train: {overlap}"


def test_all_rows_tier3_validated() -> None:
    rows = build_training_rows(find_committed_profile("max-msp"))
    assert len(rows) >= 100
    assert all(r["tier3_validated"] for r in rows)
    assert all(r["messages"][-1]["role"] == "assistant" for r in rows)


def test_messages_match_prompt_format() -> None:
    rows = build_training_rows(find_committed_profile("max-msp"))
    row = rows[0]
    assert row["messages"][0]["role"] == "system"
    assert row["messages"][1]["role"] == "user"
    label = json.loads(row["messages"][-1]["content"])
    assert label["kind"] in ("intent", "refusal")


def test_generate_dataset_writes_manifest(tmp_path) -> None:
    manifest = generate_dataset("max-msp", output_dir=tmp_path)
    assert manifest["train_count"] > 0
    assert manifest["val_count"] > 0
    assert (tmp_path / "train.jsonl").is_file()
    assert (tmp_path / "val.jsonl").is_file()
    assert (tmp_path / "manifest.json").is_file()
