from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from llm4osc.models import DeviceProfile, RefusalIntent, SuccessIntent
from llm4osc.profile import find_committed_profile, repo_root
from llm4osc.prompt import build_messages
from llm4osc.retrieval import patterns_for_context
from tier3.pipeline import run_pipeline

BENCHMARKS = repo_root() / "benchmarks"
HELD_OUT_NL: frozenset[str] | None = None


def _load_held_out_nl() -> frozenset[str]:
    global HELD_OUT_NL
    if HELD_OUT_NL is not None:
        return HELD_OUT_NL
    texts: set[str] = set()
    for sub in ("golden_nl", "golden_nl_paraphrase", "golden_refusal"):
        directory = BENCHMARKS / sub
        if not directory.is_dir():
            continue
        for path in directory.glob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            texts.add(data["nl"].strip().lower())
    HELD_OUT_NL = frozenset(texts)
    return HELD_OUT_NL


def intent_for_pattern(
    profile: DeviceProfile,
    pattern_id: str,
    args: list[Any],
) -> dict[str, Any]:
    pattern = next(p for p in profile.patterns if p.pattern_id == pattern_id)
    intent = SuccessIntent(
        device_id=profile.device_id,
        profile_version=profile.profile_version,
        pattern_id=pattern_id,
        address=pattern.address,
        type_tags=pattern.type_tags,
        args=args,
    )
    return intent.model_dump(mode="json")


def refusal_for(
    profile: DeviceProfile,
    reason: str,
    message: str,
    *,
    candidates: list[str] | None = None,
) -> dict[str, Any]:
    intent = RefusalIntent(
        device_id=profile.device_id,
        profile_version=profile.profile_version,
        reason=reason,  # type: ignore[arg-type]
        message=message,
        candidates=candidates or [],
    )
    return intent.model_dump(mode="json")


def _pct_templates(pattern_id: str, pct: int) -> list[tuple[str, list[Any]]]:
    f = pct / 100.0
    tag = "gain" if pattern_id == "gain_set" else "volume"
    return [
        (f"set {tag} to {pct}%", [f]),
        (f"adjust {tag} to {pct} percent", [f]),
        (f"bring {tag} to {pct}%", [f]),
        (f"set {tag} level to {pct}%", [f]),
        (f"change {tag} to {pct} percent", [f]),
    ]


def _float_templates(pattern_id: str, value: float, word: str) -> list[tuple[str, list[Any]]]:
    if pattern_id == "frequency_set":
        return [
            (f"set frequency to {value:g}", [value]),
            (f"tune oscillator to {value:g} hz", [value]),
            (f"set pitch to {value:g}", [value]),
            (f"frequency {value:g} hertz", [value]),
            (f"set freq to {value:g}", [value]),
        ]
    if pattern_id == "tempo_set":
        v = float(value)
        return [
            (f"set tempo to {v:g} bpm", [v]),
            (f"run metronome at {v:g} beats per minute", [v]),
            (f"metro tempo {v:g}", [v]),
            (f"set bpm to {v:g}", [v]),
            (f"change tempo to {v:g}", [v]),
        ]
    if pattern_id == "pan_set":
        return [
            (f"set pan to {value:g}", [value]),
            (f"pan position {value:g}", [value]),
            (f"set stereo pan to {value:g}", [value]),
            (f"balance to {value:g}", [value]),
            (f"move pan to {value:g}", [value]),
        ]
    return [(f"set {word} to {value:g}", [value])]


def _no_arg_templates(pattern_id: str) -> list[str]:
    mapping: dict[str, list[str]] = {
        "transport_start": [
            "start transport playback",
            "begin transport",
            "press play on transport",
            "start playback now",
            "go transport",
            "hit play",
            "start the transport",
        ],
        "transport_stop": [
            "stop transport",
            "halt transport playback",
            "stop playback",
            "end transport",
            "transport stop now",
            "pause playback session",
        ],
        "record_start": [
            "start recording",
            "begin capture",
            "start audio capture",
            "roll recording",
            "commence capture",
            "start tape",
        ],
        "record_stop": [
            "stop recording",
            "end capture",
            "stop audio capture",
            "halt recording",
            "finish recording",
        ],
        "mute_on": [
            "mute output",
            "turn mute on",
            "silence the output",
            "enable mute",
            "mute audio",
        ],
        "mute_off": [
            "unmute output",
            "turn mute off",
            "disable mute",
            "unmute audio",
            "restore audio",
        ],
        "bypass_on": [
            "enable bypass",
            "turn bypass on",
            "bypass the effect",
            "activate bypass",
        ],
    }
    return mapping.get(pattern_id, [])


def _refusal_specs(profile: DeviceProfile) -> list[tuple[str, dict[str, Any]]]:
    return [
        (
            "activate the flux capacitor",
            refusal_for(profile, "unknown_pattern", "No matching pattern."),
        ),
        (
            "warp drive to ludicrous speed",
            refusal_for(profile, "unknown_pattern", "Unknown control."),
        ),
        (
            "set gain",
            refusal_for(profile, "missing_slot", "Missing gain value."),
        ),
        (
            "adjust volume",
            refusal_for(profile, "missing_slot", "Missing volume value."),
        ),
        (
            "set frequency",
            refusal_for(profile, "missing_slot", "Missing frequency."),
        ),
        (
            "begin",
            refusal_for(
                profile,
                "ambiguous_pattern",
                "Ambiguous start command.",
                candidates=["pattern_id:transport_start", "pattern_id:record_start"],
            ),
        ),
        (
            "stop",
            refusal_for(
                profile,
                "ambiguous_pattern",
                "Ambiguous stop command.",
                candidates=["pattern_id:transport_stop", "pattern_id:record_stop"],
            ),
        ),
    ]


def _pattern_examples(profile: DeviceProfile) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []

    for pct in (10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90):
        for nl, args in _pct_templates("gain_set", pct):
            rows.append((nl, intent_for_pattern(profile, "gain_set", args)))
        for nl, args in _pct_templates("master_volume", pct):
            rows.append((nl, intent_for_pattern(profile, "master_volume", args)))

    for freq in (220, 330, 440, 880, 1000, 200, 500):
        for nl, args in _float_templates("frequency_set", float(freq), "frequency"):
            rows.append((nl, intent_for_pattern(profile, "frequency_set", args)))

    for tempo in (60, 90, 100, 120, 128, 140, 160, 180):
        for nl, args in _float_templates("tempo_set", float(tempo), "tempo"):
            rows.append((nl, intent_for_pattern(profile, "tempo_set", args)))

    for pan in (-1.0, -0.5, 0.0, 0.5, 1.0):
        for nl, args in _float_templates("pan_set", pan, "pan"):
            rows.append((nl, intent_for_pattern(profile, "pan_set", args)))

    for pattern_id in (
        "transport_start",
        "transport_stop",
        "record_start",
        "record_stop",
        "mute_on",
        "mute_off",
        "bypass_on",
    ):
        pattern = next(p for p in profile.patterns if p.pattern_id == pattern_id)
        args: list[Any] = [1] if pattern.type_tags == "i" else []
        for nl in _no_arg_templates(pattern_id):
            rows.append((nl, intent_for_pattern(profile, pattern_id, args)))

    rows.extend(_refusal_specs(profile))
    return rows


def _validate_intent(profile: DeviceProfile, intent_data: dict[str, Any]) -> bool:
    if intent_data.get("kind") == "refusal":
        return True
    try:
        run_pipeline(intent_data, profile, dry_run=True)
        return True
    except Exception:
        return False


def _split_for(nl: str) -> str:
    digest = hashlib.sha256(nl.encode()).hexdigest()
    return "val" if int(digest[:8], 16) % 5 == 0 else "train"


def build_training_rows(
    profile: DeviceProfile,
    *,
    exclude_held_out: bool = True,
) -> list[dict[str, Any]]:
    held_out = _load_held_out_nl() if exclude_held_out else frozenset()
    rows: list[dict[str, Any]] = []

    for nl, intent_data in _pattern_examples(profile):
        key = nl.strip().lower()
        if key in held_out:
            continue
        if not _validate_intent(profile, intent_data):
            continue

        patterns = patterns_for_context(nl, profile.patterns, k=8)
        messages = build_messages(nl, profile, patterns, few_shot_examples=[])
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(intent_data, ensure_ascii=False),
            }
        )
        rows.append(
            {
                "id": hashlib.sha256(f"{nl}|{intent_data['kind']}".encode()).hexdigest()[:16],
                "split": _split_for(nl),
                "device_id": profile.device_id,
                "profile_version": profile.profile_version,
                "nl": nl,
                "messages": messages,
                "label_source": "synthetic",
                "tier3_validated": True,
            }
        )

    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_dataset(
    device_id: str = "max-msp",
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    profile = find_committed_profile(device_id)
    rows = build_training_rows(profile)
    out = output_dir or (repo_root() / "data")
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    write_jsonl(out / "train.jsonl", train_rows)
    write_jsonl(out / "val.jsonl", val_rows)

    payload = json.dumps(train_rows + val_rows, sort_keys=True).encode()
    data_hash = hashlib.sha256(payload).hexdigest()[:12]
    manifest = {
        "device_id": device_id,
        "profile_version": profile.profile_version,
        "data_hash": data_hash,
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "held_out_nl_count": len(_load_held_out_nl()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
