from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm4osc.models import RefusalIntent, SuccessIntent
from llm4osc.profile import find_committed_profile, repo_root
from llm4osc.resolver import Backend, resolve_nl
from tier3.pipeline import run_pipeline

BENCHMARKS_DIR = repo_root() / "benchmarks"
GOLDEN_NL = BENCHMARKS_DIR / "golden_nl"
GOLDEN_REFUSAL = BENCHMARKS_DIR / "golden_refusal"


def _load_json_dir(directory: Path) -> list[dict[str, Any]]:
    cases = []
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_file"] = path.name
        cases.append(data)
    return cases


def _approx_equal(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-5)
    return a == b


def _args_equal(got: list[Any], expect: list[Any]) -> bool:
    if len(got) != len(expect):
        return False
    return all(_approx_equal(g, e) for g, e in zip(got, expect))


def _intent_semantic_match(result: SuccessIntent, expect: dict[str, Any]) -> bool:
    if result.pattern_id != expect["pattern_id"]:
        return False
    if "address" in expect and result.address != expect["address"]:
        return False
    if "args" in expect and not _args_equal(result.args, expect["args"]):
        return False
    return True


def _would_wrong_send(
    result: SuccessIntent | RefusalIntent,
    expect: dict[str, Any],
    profile,
) -> bool:
    if expect["kind"] == "refusal":
        if isinstance(result, RefusalIntent):
            return False
        try:
            run_pipeline(result.model_dump(mode="json"), profile, dry_run=True)
            return True
        except Exception:
            return False

    if not isinstance(result, SuccessIntent):
        return False
    if _intent_semantic_match(result, expect):
        return False
    try:
        run_pipeline(result.model_dump(mode="json"), profile, dry_run=True)
        return True
    except Exception:
        return False


def score(
    device_id: str = "max-msp",
    *,
    backend: Backend = "b0",
    llm=None,
    model_id: str | None = None,
) -> dict[str, Any]:
    profile = find_committed_profile(device_id)
    nl_cases = _load_json_dir(GOLDEN_NL)
    refusal_cases = _load_json_dir(GOLDEN_REFUSAL)

    nl_correct = 0
    wrong_sends = 0
    latencies_ms: list[float] = []

    def _resolve(nl: str):
        return resolve_nl(
            nl,
            profile,
            backend=backend,
            llm=llm,
            model_id=model_id,
        )

    for case in nl_cases:
        t0 = time.perf_counter()
        result = _resolve(case["nl"])
        latencies_ms.append((time.perf_counter() - t0) * 1000)

        expect = case["expect"]
        if expect["kind"] == "intent" and isinstance(result, SuccessIntent):
            if _intent_semantic_match(result, expect):
                nl_correct += 1
        if _would_wrong_send(result, expect, profile):
            wrong_sends += 1

    refusal_tp = refusal_fp = refusal_fn = 0
    for case in refusal_cases:
        result = _resolve(case["nl"])
        expect = case["expect"]
        predicted_refusal = isinstance(result, RefusalIntent)
        gold_refusal = expect["kind"] == "refusal"

        if gold_refusal and predicted_refusal:
            if result.reason.value == expect.get("reason", result.reason.value):
                refusal_tp += 1
            else:
                refusal_fn += 1
        elif gold_refusal and not predicted_refusal:
            refusal_fn += 1
        elif not gold_refusal and predicted_refusal:
            refusal_fp += 1

        if _would_wrong_send(result, expect, profile):
            wrong_sends += 1

    repro_ok = 0
    for case in nl_cases[:3]:
        first = _resolve(case["nl"])
        if all(_resolve(case["nl"]) == first for _ in range(4)):
            repro_ok += 1

    nl_total = len(nl_cases)
    refusal_total = len(refusal_cases)
    semantic_accuracy = nl_correct / nl_total if nl_total else 1.0
    wrong_send_rate = wrong_sends / (nl_total + refusal_total)
    refusal_precision = (
        refusal_tp / (refusal_tp + refusal_fp) if (refusal_tp + refusal_fp) else 1.0
    )
    refusal_recall = refusal_tp / refusal_total if refusal_total else 1.0
    latencies_ms.sort()

    baseline_label = {"b0": "B0", "b1": "B1", "b2": "B2"}[backend]

    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline": baseline_label,
        "device_id": device_id,
        "profile_version": profile.profile_version,
        "counts": {
            "nl_cases": nl_total,
            "refusal_cases": refusal_total,
        },
        "metrics": {
            "semantic_accuracy": round(semantic_accuracy, 4),
            "wrong_send_rate": round(wrong_send_rate, 4),
            "refusal_precision": round(refusal_precision, 4),
            "refusal_recall": round(refusal_recall, 4),
            "reproducibility": round(repro_ok / 3, 4) if nl_cases else 1.0,
            "latency_ms_p50": round(latencies_ms[len(latencies_ms) // 2], 3)
            if latencies_ms
            else 0.0,
            "latency_ms_p95": round(latencies_ms[int(len(latencies_ms) * 0.95)], 3)
            if latencies_ms
            else 0.0,
        },
        "gates": {
            "wrong_send_rate_max": 0.0,
            "semantic_accuracy_min": 0.9,
            "passed": wrong_send_rate == 0.0 and semantic_accuracy >= 0.9,
        },
    }


def score_b0(device_id: str = "max-msp") -> dict[str, Any]:
    return score(device_id, backend="b0")
