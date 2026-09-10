from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llm4osc.models import PatternRecord, RefusalIntent, parse_intent
from llm4osc.profile import (
    ProfileError,
    add_pattern,
    commit_draft,
    diff_profiles,
    drafts_dir,
    find_committed_profile,
    import_patterns_yaml,
    import_text_proposals,
    init_draft,
    load_profile,
    save_profile,
    validate_profile,
)
from llm4osc.resolver import resolve_nl
from tier3.pipeline import run_pipeline
from tier3.validate import ValidationError


def _print_json(data: object) -> None:
    if hasattr(data, "model_dump"):
        print(json.dumps(data.model_dump(mode="json"), indent=2))
    else:
        print(json.dumps(data, indent=2))


def cmd_profile_init(args: argparse.Namespace) -> int:
    profile = init_draft(args.device)
    path = drafts_dir() / f"{args.device}.json"
    save_profile(path, profile)
    print(f"Created draft: {path}")
    return 0


def cmd_profile_validate(args: argparse.Namespace) -> int:
    profile = load_profile(args.path)
    errors = validate_profile(profile)
    try:
        profile.model_validate(profile.model_dump())
    except Exception as exc:
        errors.append(str(exc))
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print(f"OK: {args.path}")
    return 0


def cmd_profile_commit(args: argparse.Namespace) -> int:
    try:
        out = commit_draft(
            Path(args.path),
            force=bool(getattr(args, "force", False)),
            skip_acceptance=bool(getattr(args, "skip_acceptance", False)),
        )
    except ProfileError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Committed: {out}")
    return 0


def cmd_acceptance_ensure(args: argparse.Namespace) -> int:
    from llm4osc.acceptance import ensure_suite, case_counts

    meta = ensure_suite(args.device, seed_from_benchmarks=not args.empty)
    counts = case_counts(args.device)
    _print_json({"meta": meta, "counts": counts})
    return 0


def cmd_acceptance_run(args: argparse.Namespace) -> int:
    from llm4osc.acceptance import run_acceptance, summarize_report

    report = run_acceptance(args.device, backend=args.backend)
    _print_json(report)
    print(summarize_report(report), file=sys.stderr)
    return 0 if report.get("gates", {}).get("passed") else 1


def cmd_acceptance_status(args: argparse.Namespace) -> int:
    from llm4osc.acceptance import status_panel

    _print_json(status_panel(args.device))
    return 0


def cmd_golden_add(args: argparse.Namespace) -> int:
    from llm4osc.acceptance import pin_case, pin_from_result
    from llm4osc.profile import find_committed_profile
    from llm4osc.resolver import resolve_nl

    profile = find_committed_profile(args.device)
    nl = args.nl
    if args.expect_refuse:
        path = pin_case(
            args.device,
            nl,
            {"kind": "refusal", "reason": args.expect_refuse},
            profile=profile,
        )
    elif args.from_resolve:
        result = resolve_nl(nl, profile, backend="b0")
        path = pin_from_result(
            args.device,
            nl,
            result,
            as_refusal=args.as_refuse,
            profile=profile,
        )
    else:
        print(
            "ERROR: pass --from-resolve or --expect-refuse REASON",
            file=sys.stderr,
        )
        return 1
    print(f"Pinned: {path}")
    return 0


def cmd_profile_show(args: argparse.Namespace) -> int:
    if args.path:
        profile = load_profile(args.path)
    else:
        profile = find_committed_profile(args.device)
    _print_json(profile)
    return 0


def cmd_profile_list_patterns(args: argparse.Namespace) -> int:
    profile = (
        load_profile(args.path)
        if args.path
        else find_committed_profile(args.device)
    )
    for p in profile.patterns:
        print(f"{p.pattern_id:20} {p.address:24} {p.type_tags!r:6} {p.description}")
    return 0


def cmd_profile_add_pattern(args: argparse.Namespace) -> int:
    profile = load_profile(args.path)
    pattern = PatternRecord(
        pattern_id=args.pattern_id,
        address=args.address,
        type_tags=args.type_tags,
        description=args.description,
        tags=args.tags.split(",") if args.tags else [],
        manual_ref=args.manual_ref,
    )
    try:
        add_pattern(profile, pattern)
    except ProfileError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    save_profile(args.path, profile)
    print(f"Added pattern {args.pattern_id}")
    return 0


def cmd_profile_import(args: argparse.Namespace) -> int:
    profile = load_profile(args.path)
    patterns = import_patterns_yaml(Path(args.file))
    for pattern in patterns:
        add_pattern(profile, pattern)
    save_profile(args.path, profile)
    print(f"Imported {len(patterns)} patterns into {args.path}")
    return 0


def cmd_profile_import_text(args: argparse.Namespace) -> int:
    text = Path(args.file).read_text(encoding="utf-8")
    proposals = import_text_proposals(text)
    _print_json(proposals)
    print("\n(proposals only — edit draft and commit manually)", file=sys.stderr)
    return 0


def cmd_profile_diff(args: argparse.Namespace) -> int:
    a = load_profile(args.a)
    b = load_profile(args.b)
    _print_json(diff_profiles(a, b))
    return 0


def cmd_validate_intent(args: argparse.Namespace) -> int:
    data = json.loads(Path(args.intent).read_text(encoding="utf-8"))
    intent = parse_intent(data)
    if isinstance(intent, RefusalIntent):
        print(f"Refusal: {intent.reason.value} — {intent.message}")
        return 1
    profile = find_committed_profile(intent.device_id)
    try:
        from tier3.validate import validate_intent

        validate_intent(intent, profile)
    except ValidationError as exc:
        print(f"ERROR: {exc.reason.value} — {exc.message}", file=sys.stderr)
        return 1
    print("OK")
    return 0


def cmd_send_intent(args: argparse.Namespace) -> int:
    data = json.loads(Path(args.intent).read_text(encoding="utf-8"))
    intent = parse_intent(data)
    if isinstance(intent, RefusalIntent):
        print(f"Refusal: {intent.reason.value} — {intent.message}", file=sys.stderr)
        return 1
    profile = find_committed_profile(intent.device_id)
    try:
        result = run_pipeline(
            intent.model_dump(mode="json"),
            profile,
            dry_run=args.dry_run,
        )
    except ValidationError as exc:
        print(f"ERROR: {exc.reason.value} — {exc.message}", file=sys.stderr)
        return 1
    print(f"address: {result.preview.address}")
    print(f"args:    {result.preview.args}")
    print(f"hash:    {result.intent_hash}")
    print(f"sent:    {result.sent}")
    return 0


def cmd_send(args: argparse.Namespace) -> int:
    profile = find_committed_profile(args.device)
    try:
        result = resolve_nl(
            args.nl,
            profile,
            backend=args.backend,
            model_id=args.model,
            adapter_path=getattr(args, "adapter", None),
            serve_url=getattr(args, "serve_url", None),
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if isinstance(result, RefusalIntent):
        print(f"Refusal: {result.reason.value}")
        print(result.message)
        if result.candidates:
            print("Candidates:", ", ".join(result.candidates))
        return 1

    _print_json(result)
    if args.preview and not args.yes:
        try:
            answer = input("Send? [y/N] ").strip().lower()
        except EOFError:
            answer = "n"
        if answer not in ("y", "yes"):
            print("Aborted.")
            return 0

    try:
        pipeline_result = run_pipeline(
            result.model_dump(mode="json"),
            profile,
            dry_run=args.dry_run,
        )
    except ValidationError as exc:
        print(f"ERROR: {exc.reason.value} — {exc.message}", file=sys.stderr)
        return 1

    print(f"OSC: {pipeline_result.preview.address} {pipeline_result.preview.args}")
    print(f"hash: {pipeline_result.intent_hash}")
    print(f"sent: {pipeline_result.sent}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    from llm4osc.scorecard import score

    suite_root = None
    if getattr(args, "acceptance", False):
        from llm4osc.acceptance import acceptance_dir, ensure_suite

        ensure_suite(args.device)
        suite_root = acceptance_dir(args.device)

    try:
        report = score(
            args.device,
            backend=args.backend,
            suite=args.suite,
            model_id=args.model,
            adapter_path=getattr(args, "adapter", None),
            serve_url=getattr(args, "serve_url", None),
            suite_root=suite_root,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    text = json.dumps(report, indent=2)
    if args.write:
        path = Path(args.write)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {path}")
    print(text)
    return 0 if report["gates"]["passed"] else 1


def cmd_score_compare(args: argparse.Namespace) -> int:
    from llm4osc.scorecard import compare_track_c

    backends = tuple(b.strip() for b in args.backends.split(",") if b.strip())  # type: ignore
    try:
        report = compare_track_c(
            args.device,
            backends=backends,  # type: ignore
            model_id=args.model,
            adapter_path=getattr(args, "adapter", None),
            serve_url=getattr(args, "serve_url", None),
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    text = json.dumps(report, indent=2)
    if args.write:
        path = Path(args.write)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {path}")
    print(text)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from llm4osc.serve import run_server

    try:
        run_server(
            args.host,
            args.port,
            model_id=args.model,
            adapter_path=getattr(args, "adapter", None),
            preload=not args.no_preload,
        )
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from demo.app import main as demo_main

    demo_main()
    return 0


def cmd_train_data(args: argparse.Namespace) -> int:
    from llm4osc.training_data import generate_dataset

    manifest = generate_dataset(args.device, output_dir=Path(args.output_dir))
    _print_json(manifest)
    return 0 if manifest["train_count"] > 0 else 1


def _add_adapter_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--adapter",
        default=None,
        help="LoRA adapter path for b3 (or set LLM4OSC_ADAPTER)",
    )


def _add_serve_url_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--serve-url",
        default=None,
        help="Use running llm4osc serve (or set LLM4OSC_SERVE_URL)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm4osc", description="LLM4OSC CLI")
    sub = parser.add_subparsers(dest="command")

    profile_parser = sub.add_parser("profile", help="Device profile authoring")
    p_sub = profile_parser.add_subparsers(dest="profile_cmd")

    s = p_sub.add_parser("init", help="Create draft profile")
    s.add_argument("--device", required=True)
    s.set_defaults(func=cmd_profile_init)

    s = p_sub.add_parser("validate", help="Validate profile file")
    s.add_argument("path")
    s.set_defaults(func=cmd_profile_validate)

    s = p_sub.add_parser("commit", help="Commit draft to profiles/committed/")
    s.add_argument("path")
    s.add_argument(
        "--force",
        action="store_true",
        help="Commit even if Profile Acceptance gate fails",
    )
    s.add_argument(
        "--skip-acceptance",
        action="store_true",
        help="Skip acceptance gate (bootstrap / tests)",
    )
    s.set_defaults(func=cmd_profile_commit)

    s = p_sub.add_parser("show", help="Show profile JSON")
    s.add_argument("--device", default="max-msp")
    s.add_argument("--path")
    s.set_defaults(func=cmd_profile_show)

    s = p_sub.add_parser("list-patterns", help="List patterns in profile")
    s.add_argument("--device", default="max-msp")
    s.add_argument("--path")
    s.set_defaults(func=cmd_profile_list_patterns)

    s = p_sub.add_parser("add-pattern", help="Add pattern to draft")
    s.add_argument("path")
    s.add_argument("--pattern-id", required=True)
    s.add_argument("--address", required=True)
    s.add_argument("--type-tags", default="")
    s.add_argument("--description", required=True)
    s.add_argument("--tags", default="")
    s.add_argument("--manual-ref", default=None)
    s.set_defaults(func=cmd_profile_add_pattern)

    s = p_sub.add_parser("import", help="Import patterns from YAML")
    s.add_argument("path", help="Draft profile path")
    s.add_argument("--file", required=True)
    s.set_defaults(func=cmd_profile_import)

    s = p_sub.add_parser("import-text", help="Propose patterns from OSC excerpt")
    s.add_argument("--file", required=True)
    s.set_defaults(func=cmd_profile_import_text)

    s = p_sub.add_parser("diff", help="Diff two profiles")
    s.add_argument("a")
    s.add_argument("b")
    s.set_defaults(func=cmd_profile_diff)

    s = sub.add_parser("validate", help="Validate intent JSON against profile")
    s.add_argument("intent")
    s.set_defaults(func=cmd_validate_intent)

    s = sub.add_parser("send-intent", help="Send intent JSON through Tier 3")
    s.add_argument("intent")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_send_intent)

    s = sub.add_parser("send", help="NL → intent → Tier 3 → OSC")
    s.add_argument("--device", default="max-msp")
    s.add_argument("--nl", required=True)
    s.add_argument(
        "--backend",
        choices=["b0", "b1", "b2", "b3"],
        default="b0",
        help="b0=rules, b1=Qwen zero-shot, b2=Qwen few-shot, b3=Qwen+LoRA",
    )
    s.add_argument(
        "--model",
        default=None,
        help="Hugging Face model id (default: Qwen/Qwen2-0.5B-Instruct)",
    )
    s.add_argument("--preview", action="store_true", default=True)
    s.add_argument("--yes", "-y", action="store_true", help="Skip send confirmation")
    s.add_argument("--dry-run", action="store_true")
    _add_adapter_arg(s)
    _add_serve_url_arg(s)
    s.set_defaults(func=cmd_send)

    s = sub.add_parser("score", help="Run benchmark scorecard")
    s.add_argument("--device", default="max-msp")
    s.add_argument("--backend", choices=["b0", "b1", "b2", "b3"], default="b0")
    s.add_argument(
        "--suite",
        choices=["full", "literal", "paraphrase"],
        default="full",
    )
    s.add_argument(
        "--acceptance",
        action="store_true",
        help="Score profiles/acceptance/<device> instead of benchmarks/",
    )
    s.add_argument("--model", default=None)
    s.add_argument(
        "--write",
        type=str,
        default=None,
        help="Write JSON to path (e.g. benchmarks/results/baseline.json)",
    )
    _add_adapter_arg(s)
    _add_serve_url_arg(s)
    s.set_defaults(func=cmd_score)

    acc = sub.add_parser(
        "acceptance",
        help="Profile Acceptance: per-rig suite pin / run / status",
    )
    a_sub = acc.add_subparsers(dest="acceptance_cmd")

    s = a_sub.add_parser("ensure", help="Create suite (seed Max from benchmarks)")
    s.add_argument("--device", default="max-msp")
    s.add_argument(
        "--empty",
        action="store_true",
        help="Do not seed from benchmarks/",
    )
    s.set_defaults(func=cmd_acceptance_ensure)

    s = a_sub.add_parser("run", help="Score acceptance suite (B0 default)")
    s.add_argument("--device", default="max-msp")
    s.add_argument("--backend", choices=["b0", "b1", "b2", "b3"], default="b0")
    s.set_defaults(func=cmd_acceptance_run)

    s = a_sub.add_parser("status", help="Show suite counts + last report")
    s.add_argument("--device", default="max-msp")
    s.set_defaults(func=cmd_acceptance_status)

    s = sub.add_parser("golden", help="Pin acceptance goldens")
    g_sub = s.add_subparsers(dest="golden_cmd")
    s = g_sub.add_parser("add", help="Pin NL into device acceptance suite")
    s.add_argument("--device", default="max-msp")
    s.add_argument("--nl", required=True)
    s.add_argument(
        "--from-resolve",
        action="store_true",
        help="Resolve with B0 and pin the result",
    )
    s.add_argument(
        "--as-refuse",
        action="store_true",
        help="With --from-resolve, pin as must-refuse",
    )
    s.add_argument(
        "--expect-refuse",
        default=None,
        help="Pin must-refuse with this reason (e.g. unknown_pattern)",
    )
    s.set_defaults(func=cmd_golden_add)

    s = sub.add_parser(
        "score-compare",
        help="Track C: literal vs paraphrase across backends",
    )
    s.add_argument("--device", default="max-msp")
    s.add_argument("--backends", default="b0,b1,b2,b3")
    s.add_argument("--model", default=None)
    s.add_argument(
        "--write",
        type=str,
        default="benchmarks/results/track_c.json",
    )
    _add_adapter_arg(s)
    _add_serve_url_arg(s)
    s.set_defaults(func=cmd_score_compare)

    s = sub.add_parser("serve", help="Keep Qwen loaded; HTTP resolve API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--model", default=None)
    _add_adapter_arg(s)
    s.add_argument(
        "--no-preload",
        action="store_true",
        help="Skip model load at startup (load on first b1/b2/b3 request)",
    )
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("demo", help="Launch Gradio demo UI (profile / backend / dry-run)")
    s.set_defaults(func=cmd_demo)

    s = sub.add_parser("train-data", help="Generate LoRA training JSONL")
    s.add_argument("--device", default="max-msp")
    s.add_argument("--output-dir", default="data")
    s.set_defaults(func=cmd_train_data)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    if args.command == "profile" and not getattr(args, "profile_cmd", None):
        print("usage: llm4osc profile <init|validate|commit|...> ...", file=sys.stderr)
        return 0
    if args.command == "acceptance" and not getattr(args, "acceptance_cmd", None):
        print(
            "usage: llm4osc acceptance <ensure|run|status> [--device max-msp]",
            file=sys.stderr,
        )
        return 0
    if args.command == "golden" and not getattr(args, "golden_cmd", None):
        print(
            "usage: llm4osc golden add --nl ... (--from-resolve | --expect-refuse REASON)",
            file=sys.stderr,
        )
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
