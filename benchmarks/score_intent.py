#!/usr/bin/env python3
"""Score resolver — CLI entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm4osc.scorecard import score


def main() -> int:
    parser = argparse.ArgumentParser(description="Score NL resolver")
    parser.add_argument("--device", default="max-msp")
    parser.add_argument("--backend", choices=["b0", "b1", "b2"], default="b0")
    parser.add_argument(
        "--suite",
        choices=["full", "literal", "paraphrase"],
        default="full",
        help="NL suite: full=literal+refusal (default), literal, paraphrase",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--write",
        type=Path,
        default=None,
        help="Write scorecard JSON (default: print only)",
    )
    args = parser.parse_args()

    try:
        report = score(
            args.device,
            backend=args.backend,
            suite=args.suite,
            model_id=args.model,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    text = json.dumps(report, indent=2)

    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {args.write}")

    print(text)
    return 0 if report["gates"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
