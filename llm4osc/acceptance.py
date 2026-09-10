"""Per-device Profile Acceptance suites (show-local goldens tied to a profile)."""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm4osc.models import DeviceProfile, RefusalIntent, SuccessIntent
from llm4osc.profile import ProfileError, find_committed_profile, profiles_dir, repo_root

BENCHMARKS_DIR = repo_root() / "benchmarks"


class AcceptanceError(ValueError):
    pass


def acceptance_dir(device_id: str) -> Path:
    return profiles_dir() / "acceptance" / device_id


def meta_path(device_id: str) -> Path:
    return acceptance_dir(device_id) / "meta.json"


def last_report_path(device_id: str) -> Path:
    return acceptance_dir(device_id) / "last_report.json"


def golden_nl_dir(device_id: str) -> Path:
    return acceptance_dir(device_id) / "golden_nl"


def golden_refusal_dir(device_id: str) -> Path:
    return acceptance_dir(device_id) / "golden_refusal"


def suite_exists(device_id: str) -> bool:
    return meta_path(device_id).is_file()


def load_meta(device_id: str) -> dict[str, Any]:
    path = meta_path(device_id)
    if not path.is_file():
        raise AcceptanceError(f"No acceptance suite for device {device_id!r}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_meta(
    device_id: str,
    *,
    profile_version: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = acceptance_dir(device_id)
    root.mkdir(parents=True, exist_ok=True)
    golden_nl_dir(device_id).mkdir(parents=True, exist_ok=True)
    golden_refusal_dir(device_id).mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "schema_version": "1.0",
        "device_id": device_id,
        "profile_version": profile_version,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        meta.update(extra)
    meta_path(device_id).write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def _copy_json_dir(src: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    if not src.is_dir():
        return 0
    for path in sorted(src.glob("*.json")):
        target = dest / path.name
        if not target.exists():
            shutil.copy2(path, target)
            n += 1
    return n


def ensure_suite(
    device_id: str,
    *,
    profile: DeviceProfile | None = None,
    seed_from_benchmarks: bool = True,
) -> dict[str, Any]:
    """Create acceptance dirs; optionally seed Max hero cases from benchmarks/."""
    if profile is None:
        profile = find_committed_profile(device_id)
    if suite_exists(device_id):
        meta = load_meta(device_id)
        return meta

    write_meta(device_id, profile_version=profile.profile_version)
    copied_nl = copied_refuse = 0
    if seed_from_benchmarks and device_id == "max-msp":
        copied_nl = _copy_json_dir(
            BENCHMARKS_DIR / "golden_nl", golden_nl_dir(device_id)
        )
        copied_refuse = _copy_json_dir(
            BENCHMARKS_DIR / "golden_refusal", golden_refusal_dir(device_id)
        )
    return write_meta(
        device_id,
        profile_version=profile.profile_version,
        extra={
            "seeded_from_benchmarks": bool(copied_nl or copied_refuse),
            "seeded_nl": copied_nl,
            "seeded_refusal": copied_refuse,
        },
    )


def bind_profile_version(device_id: str, profile_version: str) -> dict[str, Any]:
    ensure_suite(device_id, seed_from_benchmarks=False)
    meta = load_meta(device_id)
    meta["profile_version"] = profile_version
    meta["updated_at"] = datetime.now(timezone.utc).isoformat()
    meta_path(device_id).write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def case_counts(device_id: str) -> dict[str, int]:
    if not suite_exists(device_id):
        return {"nl_cases": 0, "refusal_cases": 0, "total": 0}
    nl = len(list(golden_nl_dir(device_id).glob("*.json")))
    refuse = len(list(golden_refusal_dir(device_id).glob("*.json")))
    return {"nl_cases": nl, "refusal_cases": refuse, "total": nl + refuse}


def _slug(nl: str, limit: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", nl.lower()).strip("_")
    return (slug or "case")[:limit]


def _unique_path(directory: Path, stem: str) -> Path:
    candidate = directory / f"{stem}.json"
    if not candidate.exists():
        return candidate
    for i in range(2, 1000):
        candidate = directory / f"{stem}_{i}.json"
        if not candidate.exists():
            return candidate
    raise AcceptanceError(f"Too many cases with stem {stem!r}")


def expect_from_result(result: SuccessIntent | RefusalIntent) -> dict[str, Any]:
    if isinstance(result, SuccessIntent):
        return {
            "kind": "intent",
            "pattern_id": result.pattern_id,
            "address": result.address,
            "args": result.args,
        }
    return {
        "kind": "refusal",
        "reason": result.reason.value,
    }


def pin_case(
    device_id: str,
    nl: str,
    expect: dict[str, Any],
    *,
    profile: DeviceProfile | None = None,
) -> Path:
    """Pin an NL case into the device acceptance suite (must-work or must-refuse)."""
    nl = nl.strip()
    if not nl:
        raise AcceptanceError("Cannot pin empty NL")
    if profile is None:
        profile = find_committed_profile(device_id)
    ensure_suite(device_id, profile=profile, seed_from_benchmarks=True)

    kind = expect.get("kind")
    if kind == "intent":
        directory = golden_nl_dir(device_id)
        prefix = "pin"
        case_id = f"pin_{_slug(nl)}"
    elif kind == "refusal":
        directory = golden_refusal_dir(device_id)
        prefix = "refuse"
        reason = expect.get("reason", "unknown_pattern")
        case_id = f"refuse_{_slug(str(reason))}_{_slug(nl, 24)}"
    else:
        raise AcceptanceError(f"expect.kind must be intent or refusal, got {kind!r}")

    path = _unique_path(directory, case_id)
    payload = {
        "id": path.stem,
        "nl": nl,
        "expect": expect,
        "pinned_at": datetime.now(timezone.utc).isoformat(),
        "source": "acceptance_pin",
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_meta(device_id, profile_version=profile.profile_version)
    return path


def pin_from_result(
    device_id: str,
    nl: str,
    result: SuccessIntent | RefusalIntent,
    *,
    as_refusal: bool | None = None,
    profile: DeviceProfile | None = None,
) -> Path:
    """Pin from a resolve result. as_refusal=True forces a refusal expect."""
    if as_refusal is True:
        if isinstance(result, RefusalIntent):
            expect = expect_from_result(result)
        else:
            expect = {"kind": "refusal", "reason": "unknown_pattern"}
    elif as_refusal is False:
        if not isinstance(result, SuccessIntent):
            raise AcceptanceError("Cannot pin must-work from a refusal result")
        expect = expect_from_result(result)
    else:
        expect = expect_from_result(result)
    return pin_case(device_id, nl, expect, profile=profile)


def write_last_report(device_id: str, report: dict[str, Any]) -> Path:
    acceptance_dir(device_id).mkdir(parents=True, exist_ok=True)
    path = last_report_path(device_id)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def load_last_report(device_id: str) -> dict[str, Any] | None:
    path = last_report_path(device_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_report(report: dict[str, Any]) -> str:
    gates = report.get("gates", {})
    metrics = report.get("metrics", {})
    counts = report.get("counts", {})
    passed = gates.get("passed")
    status = "PASS" if passed else "FAIL"
    return (
        f"{status} · acc {metrics.get('semantic_accuracy', 0):.0%} · "
        f"wrong-send {metrics.get('wrong_send_rate', 0):.0%} · "
        f"{counts.get('nl_cases', 0)} must-work · "
        f"{counts.get('refusal_cases', 0)} must-refuse · "
        f"{report.get('profile_version', '?')}"
    )


def run_acceptance(
    device_id: str,
    *,
    profile: DeviceProfile | None = None,
    backend: str = "b0",
    bind_meta: bool = False,
) -> dict[str, Any]:
    """Score the device acceptance suite; persist last_report.json."""
    from llm4osc.scorecard import score

    if profile is None:
        profile = find_committed_profile(device_id)
    ensure_suite(device_id, profile=profile, seed_from_benchmarks=True)
    meta = load_meta(device_id)
    report = score(
        device_id,
        backend=backend,  # type: ignore[arg-type]
        suite="full",
        profile=profile,
        suite_root=acceptance_dir(device_id),
    )
    report["acceptance"] = True
    report["suite_root"] = str(acceptance_dir(device_id))
    report["meta_profile_version"] = meta.get("profile_version")
    report["profile_version_match"] = meta.get("profile_version") == profile.profile_version
    write_last_report(device_id, report)
    if bind_meta and report.get("gates", {}).get("passed"):
        bind_profile_version(device_id, profile.profile_version)
    return report


def check_commit_gate(
    device_id: str,
    profile: DeviceProfile,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Run acceptance for a draft/committed profile. Raises if gate fails."""
    if not suite_exists(device_id):
        ensure_suite(device_id, profile=profile, seed_from_benchmarks=True)
    report = run_acceptance(device_id, profile=profile, backend="b0")
    if report.get("gates", {}).get("passed"):
        return report
    if force:
        report["forced"] = True
        write_last_report(device_id, report)
        return report
    raise ProfileError(
        "Acceptance gate failed — "
        f"{summarize_report(report)}. "
        "Fix cases or pin updates, then retry (or commit --force)."
    )


def status_panel(device_id: str) -> dict[str, Any]:
    """UI-friendly status blob for the demo."""
    counts = case_counts(device_id)
    meta = load_meta(device_id) if suite_exists(device_id) else None
    report = load_last_report(device_id)
    return {
        "device_id": device_id,
        "suite_exists": suite_exists(device_id),
        "counts": counts,
        "meta": meta,
        "last_report": report,
        "summary": summarize_report(report) if report else "No acceptance run yet",
    }
