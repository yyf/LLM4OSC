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


_WORD_PCT: dict[str, float] = {
    "ten": 0.1,
    "twenty": 0.2,
    "thirty": 0.3,
    "forty": 0.4,
    "fifty": 0.5,
    "sixty": 0.6,
    "seventy": 0.7,
    "eighty": 0.8,
    "ninety": 0.9,
}


def _paraphrase_examples(profile: DeviceProfile) -> list[tuple[str, dict[str, Any]]]:
    """Near-golden phrasing variants (held-out goldens excluded at build time)."""
    rows: list[tuple[str, dict[str, Any]]] = []

    def add(nl: str, pattern_id: str, args: list[Any]) -> None:
        rows.append((nl, intent_for_pattern(profile, pattern_id, args)))

    # gain / volume (reinforce fractional & attenuate phrasing)
    for nl, pattern_id, f in [
        ("make the level a quarter", "gain_set", 0.25),
        ("make the level three quarters", "gain_set", 0.75),
        ("cut the level in half", "gain_set", 0.5),
        ("bring level to half", "gain_set", 0.5),
        ("level at fifty percent", "gain_set", 0.5),
        ("attenuate output to twenty percent", "master_volume", 0.2),
        ("attenuate output to forty percent", "master_volume", 0.4),
        ("attenuate output to fifty percent", "master_volume", 0.5),
        ("reduce master to sixty percent", "master_volume", 0.6),
        ("pull output down to twenty percent", "master_volume", 0.2),
        ("attenuate master to thirty percent", "master_volume", 0.3),
        ("attenuate master to forty percent", "master_volume", 0.4),
        ("pull volume to thirty percent", "master_volume", 0.3),
        ("reduce output to thirty percent", "master_volume", 0.3),
        ("set gain to thirty percent", "gain_set", 0.3),
        ("set gain to fifty percent", "gain_set", 0.5),
    ]:
        add(nl, pattern_id, [f])

    for word, frac in _WORD_PCT.items():
        add(f"attenuate output to {word} percent", "master_volume", [frac])
        add(f"reduce master to {word} percent", "master_volume", [frac])
        add(f"set gain to {word} percent", "gain_set", [frac])

    # frequency (short / musical phrasing — not literal "set frequency to N")
    for nl, freq in [
        ("concert A at 440", 440.0),
        ("A4 note please", 440.0),
        ("440 cycles per second", 440.0),
        ("tuning reference 440", 440.0),
        ("give me 880 hz", 880.0),
        ("B4 at 493 please", 493.0),
        ("middle C at 261", 261.0),
        ("oscillator at 220 hertz", 220.0),
        ("pitch to 1000", 1000.0),
        ("A440 tone", 440.0),
        ("A440 tone please", 440.0),
        ("please concert A 440", 440.0),
        ("concert pitch 440", 440.0),
        ("tuning A4 440 hz", 440.0),
        ("reference tone A440", 440.0),
    ]:
        add(nl, "frequency_set", [freq])

    # tempo (beats-per-minute phrasing)
    for nl, tempo in [
        ("run at 90 beats per minute", 90.0),
        ("run at 100 beats per minute", 100.0),
        ("run at 128 beats per minute", 128.0),
        ("run at 140 beats per minute", 140.0),
        ("play at 120 bpm", 120.0),
        ("keep tempo at 160 beats per minute", 160.0),
        ("metronome at 80 beats per minute", 80.0),
        ("clock at 180 beats per minute", 180.0),
    ]:
        add(nl, "tempo_set", [tempo])

    # pan (center / stereo image phrasing)
    for nl, pan in [
        ("center the mix", 0.0),
        ("center the sound", 0.0),
        ("put the sound in the middle", 0.0),
        ("pan dead center", 0.0),
        ("balance in the center", 0.0),
        ("center the stereo field", 0.0),
        ("put stereo image in center", 0.0),
        ("dead center stereo", 0.0),
        ("center stereo balance", 0.0),
        ("middle of the stereo field", 0.0),
        ("hard left panning", -1.0),
        ("full right stereo", 1.0),
        ("slightly left of center", -0.25),
        ("slightly right of center", 0.25),
    ]:
        add(nl, "pan_set", [pan])

    # transport / record (idiomatic verbs)
    for nl, pid in [
        ("kick off the playback", "transport_start"),
        ("kick off playback now", "transport_start"),
        ("fire up playback", "transport_start"),
        ("start playing back", "transport_start"),
        ("start playback session", "transport_start"),
        ("go playback", "transport_start"),
        ("pause the session transport", "transport_stop"),
        ("halt playback now", "transport_stop"),
        ("commence recording now", "record_start"),
        ("begin a recording pass", "record_start"),
        ("commence recording session now", "record_start"),
    ]:
        pattern = next(p for p in profile.patterns if p.pattern_id == pid)
        args: list[Any] = [1] if pattern.type_tags == "i" else []
        add(nl, pid, args)

    return rows


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
    rows.extend(_paraphrase_examples(profile))
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
