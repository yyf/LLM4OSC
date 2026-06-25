from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from llm4osc.models import DeviceProfile, PatternRecord


class ProfileError(ValueError):
    pass


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def profiles_dir() -> Path:
    return repo_root() / "profiles"


def drafts_dir() -> Path:
    return profiles_dir() / "drafts"


def committed_dir() -> Path:
    return profiles_dir() / "committed"


def load_profile(path: Path | str) -> DeviceProfile:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return DeviceProfile.model_validate(data)


def save_profile(path: Path | str, profile: DeviceProfile) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(profile.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )


def find_committed_profile(device_id: str) -> DeviceProfile:
    committed = committed_dir()
    matches = sorted(committed.glob(f"{device_id}*.json"))
    if not matches:
        raise ProfileError(f"No committed profile for device {device_id!r}")
    return load_profile(matches[-1])


def validate_profile(profile: DeviceProfile) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_addresses: set[str] = set()
    for pattern in profile.patterns:
        if pattern.pattern_id in seen_ids:
            errors.append(f"Duplicate pattern_id: {pattern.pattern_id}")
        seen_ids.add(pattern.pattern_id)
        if pattern.address in seen_addresses:
            errors.append(f"Duplicate address: {pattern.address}")
        seen_addresses.add(pattern.address)
        if len(pattern.type_tags) != len(pattern.slots) and pattern.slots:
            # slots optional when single implicit arg
            pass
    return errors


def new_profile_version() -> str:
    now = datetime.now(timezone.utc)
    suffix = hashlib.sha256(now.isoformat().encode()).hexdigest()[:4]
    return f"prof_{now.strftime('%Y%m%d')}_{suffix}"


def init_draft(device_id: str) -> DeviceProfile:
    drafts_dir().mkdir(parents=True, exist_ok=True)
    return DeviceProfile(
        device_id=device_id,
        profile_version="draft",
        patterns=[],
        manual_sources=[],
    )


def commit_draft(draft_path: Path) -> Path:
    profile = load_profile(draft_path)
    errors = validate_profile(profile)
    if errors:
        raise ProfileError("; ".join(errors))
    if not profile.patterns:
        raise ProfileError("Profile has no patterns")

    profile.profile_version = new_profile_version()
    profile.reviewed_at = datetime.now(timezone.utc).isoformat()

    if profile.manual_sources:
        joined = "|".join(
            f"{s.ref}:{s.hash or ''}" for s in profile.manual_sources
        )
        manual_hash = hashlib.sha256(joined.encode()).hexdigest()[:12]
        for source in profile.manual_sources:
            if source.hash is None:
                source.hash = manual_hash

    out = committed_dir() / f"{profile.device_id}-{profile.profile_version}.json"
    committed_dir().mkdir(parents=True, exist_ok=True)
    save_profile(out, profile)
    return out


def add_pattern(profile: DeviceProfile, pattern: PatternRecord) -> None:
    errors = validate_profile(
        DeviceProfile(
            device_id=profile.device_id,
            profile_version=profile.profile_version,
            patterns=[*profile.patterns, pattern],
        )
    )
    if errors:
        raise ProfileError("; ".join(errors))
    profile.patterns.append(pattern)


def import_patterns_yaml(path: Path) -> list[PatternRecord]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else data.get("patterns", [])
    return [PatternRecord.model_validate(item) for item in items]


_ADDRESS_RE = re.compile(r"(/[\w/.\-]+)")


def import_text_proposals(text: str) -> list[dict[str, Any]]:
    """Rule-based OSC excerpt parser — proposals only, never auto-commit."""
    proposals: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _ADDRESS_RE.search(line)
        if not match:
            continue
        address = match.group(1)
        pattern_id = address.strip("/").replace("/", "_") or "root"
        type_tags = ""
        slots: list[dict[str, Any]] = []
        ranges: dict[str, dict[str, float]] = {}

        lower = line.lower()
        if "float" in lower or "f " in lower or "0." in lower:
            type_tags = "f"
            slots = [{"name": "value", "type": "float", "arg_index": 0}]
            ranges = {"value": {"min": 0.0, "max": 1.0}}
        elif "int" in lower:
            type_tags = "i"
            slots = [{"name": "value", "type": "int", "arg_index": 0}]
            ranges = {"value": {"min": 0, "max": 127}}

        proposals.append(
            {
                "pattern_id": pattern_id,
                "address": address,
                "type_tags": type_tags,
                "description": f"Proposed from excerpt: {line[:80]}",
                "slots": slots,
                "ranges": ranges,
                "tags": [],
                "manual_ref": "import-text",
                "_proposed": True,
            }
        )
    return proposals


def diff_profiles(a: DeviceProfile, b: DeviceProfile) -> dict[str, list[str]]:
    a_ids = {p.pattern_id for p in a.patterns}
    b_ids = {p.pattern_id for p in b.patterns}
    return {
        "added": sorted(b_ids - a_ids),
        "removed": sorted(a_ids - b_ids),
        "unchanged": sorted(a_ids & b_ids),
    }
