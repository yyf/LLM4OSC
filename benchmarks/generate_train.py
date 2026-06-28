#!/usr/bin/env python3
"""Generate Tier-3-validated LoRA training JSONL from device profile templates."""

from __future__ import annotations

import argparse
import json
import sys

from llm4osc.training_data import generate_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate LoRA training data")
    parser.add_argument("--device", default="max-msp")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory for train.jsonl, val.jsonl, manifest.json",
    )
    args = parser.parse_args(argv)

    manifest = generate_dataset(args.device, output_dir=__import__("pathlib").Path(args.output_dir))
    print(json.dumps(manifest, indent=2))
    if manifest["train_count"] == 0:
        print("ERROR: no training rows generated", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
