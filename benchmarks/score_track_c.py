#!/usr/bin/env python3
"""Track C: compare B0/B1/B2 on literal vs paraphrase NL suites."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llm4osc.scorecard import compare_track_c


def _print_summary(report: dict) -> None:
    print("\nTrack C summary (semantic accuracy):")
    print(f"{'Backend':<8} {'Literal':>10} {'Paraphrase':>12} {'Gap (pts)':>10}")
    print("-" * 44)
    for backend, gap in report["gaps"].items():
        print(
            f"{backend.upper():<8} "
            f"{gap['literal_semantic_accuracy']:>10.1%} "
            f"{gap['paraphrase_semantic_accuracy']:>12.1%} "
            f"{gap['paraphrase_gap_pts']:>10.1f}"
        )
    rec = report["recommendation"]
    print(f"\nLoRA recommended: {rec['lora_recommended']} ({rec['note']})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Track C literal vs paraphrase comparison")
    parser.add_argument("--device", default="max-msp")
    parser.add_argument(
        "--backends",
        default="b0,b1,b2",
        help="Comma-separated backends to score (default: b0,b1,b2)",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--serve-url",
        default=None,
        help="Use llm4osc serve (or set LLM4OSC_SERVE_URL)",
    )
    parser.add_argument(
        "--write",
        type=Path,
        default=None,
        help="Write full report JSON (default: benchmarks/results/track_c.json)",
    )
    args = parser.parse_args()

    backends = tuple(b.strip() for b in args.backends.split(",") if b.strip())  # type: ignore
    valid = {"b0", "b1", "b2"}
    bad = [b for b in backends if b not in valid]
    if bad:
        print(f"ERROR: invalid backends: {bad}", file=sys.stderr)
        return 1

    try:
        report = compare_track_c(
            args.device,
            backends=backends,  # type: ignore
            model_id=args.model,
            serve_url=args.serve_url,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    out_path = args.write or Path("benchmarks/results/track_c.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")

    for backend in backends:
        for suite_name in ("literal", "paraphrase"):
            sub = report["backends"][backend][suite_name]
            write_path = Path(f"benchmarks/results/{backend}_{suite_name}.json")
            write_path.write_text(json.dumps(sub, indent=2) + "\n", encoding="utf-8")
            print(f"Wrote {write_path}")

    _print_summary(report)
    print(json.dumps(report["recommendation"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
