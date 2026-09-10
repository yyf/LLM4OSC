"""Demo UI — plain Gradio + simple HTML.

Launch:
  pip install -e ".[demo]"
  python demo/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from llm4osc.acceptance import (
    AcceptanceError,
    ensure_suite,
    pin_from_result,
    run_acceptance,
    status_panel,
    summarize_report,
)
from llm4osc.llm import default_adapter_path
from llm4osc.models import RefusalIntent, SuccessIntent, parse_intent
from llm4osc.profile import find_committed_profile, list_committed_profiles
from llm4osc.resolver import Backend, resolve_nl
from llm4osc.retrieval import rank_patterns
from tier3.pipeline import run_pipeline
from tier3.validate import ValidationError

EXAMPLE_PHRASES = [
    "set gain to 50%",
    "make the level half",
    "boost the bass band by 3db",
    "start",
    "set gain",
]

EXAMPLE_LABELS = [
    "Literal → /gain",
    "Paraphrase",
    "OOV EQ",
    "Ambiguous",
    "Missing slot",
]

BACKEND_CHOICES: list[tuple[str, str]] = [
    (
        "B0 · Rules (default) — What: Profile retrieval + slot parsing · When: Live control, demos (default)",
        "b0",
    ),
    (
        "B1 · Qwen zero-shot — What: Qwen2-0.5B proposes intent JSON · When: LLM baseline",
        "b1",
    ),
    (
        "B2 · Qwen + few-shot — What: Qwen + profile examples in prompt · When: LLM baseline comparison",
        "b2",
    ),
    (
        "B3 · Qwen + LoRA — What: Fine-tuned adapter + NL refine · When: Paraphrase experiments",
        "b3",
    ),
]

LAYOUT_CSS = """
.backend-menu {
  flex: 3 1 36rem !important;
  min-width: 36rem !important;
}
.backend-menu .wrap,
.backend-menu input,
.backend-menu .secondary-wrap {
  white-space: nowrap !important;
}
.profile-menu {
  flex: 1.5 1 18rem !important;
  min-width: 18rem !important;
}
.profile-menu input,
.profile-menu .wrap input {
  padding-right: 2.75rem !important;
  white-space: nowrap !important;
  text-overflow: ellipsis !important;
  overflow: hidden !important;
}
.query-row {
  align-items: stretch !important;
  gap: 0.5rem !important;
}
/* Match Gradio Textbox block chrome (same as OSC preview). */
.nl-panel {
  background: var(--block-background-fill) !important;
  border: var(--block-border-width, 1px) solid var(--block-border-color) !important;
  border-radius: var(--block-radius) !important;
  box-shadow: var(--block-shadow, none) !important;
  padding: var(--block-padding) !important;
  overflow: visible !important;
  display: flex !important;
  flex-direction: column !important;
  gap: var(--spacing-sm, 0.5rem) !important;
  flex: 1 1 auto !important;
  min-width: 0 !important;
  height: 100% !important;
  box-sizing: border-box !important;
}
.nl-label-row {
  align-items: center !important;
  gap: 0.4rem !important;
  flex-wrap: wrap !important;
  margin: 0 !important;
  min-height: 0 !important;
}
.nl-label-row .block,
.nl-label-row .form,
.nl-label-row .wrap,
.nl-label-row .html-container {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  outline: none !important;
  margin: 0 !important;
  padding: 0 !important;
}
.nl-label {
  margin: 0 !important;
  font-size: var(--text-md) !important;
  font-weight: var(--block-title-text-weight, 600) !important;
  color: var(--block-title-text-color) !important;
}
.nl-input textarea {
  min-height: 4.5rem !important;
}
.resolve-btn {
  min-width: 6.5rem !important;
  height: auto !important;
  align-self: stretch !important;
}
.flow-ascii-wrap {
  margin: 0 0 0.75rem 0 !important;
}
.flow-ascii {
  margin: 0 !important;
  font-size: 0.85rem !important;
  line-height: 1.35 !important;
  white-space: pre !important;
  overflow-x: auto !important;
}
.acceptance-panel {
  background: var(--block-background-fill) !important;
  border: var(--block-border-width, 1px) solid var(--block-border-color) !important;
  border-radius: var(--block-radius) !important;
  padding: var(--block-padding) !important;
  margin-top: 0.5rem !important;
}
.acceptance-panel h3 {
  margin: 0 0 0.35rem 0 !important;
  font-size: 1rem !important;
}
.acceptance-panel p {
  margin: 0.25rem 0 !important;
}
.acceptance-panel .muted {
  opacity: 0.75;
  font-size: 0.9rem;
}
"""


def _backend_value(label_or_value: str) -> str:
    for label, value in BACKEND_CHOICES:
        if label_or_value in (label, value):
            return value
    key = label_or_value.lower()
    if key.startswith("b") and len(key) == 2:
        return key
    return key


def _profile_choices() -> list[str]:
    rows = list_committed_profiles()
    if not rows:
        return ["max-msp"]
    return [r["label"] for r in rows]


def _label_to_device(label: str) -> str:
    for row in list_committed_profiles():
        if row["label"] == label:
            return row["device_id"]
    return label.split(" (", 1)[0].strip()


def _retrieval_preview(nl: str, profile) -> str:
    ranked = rank_patterns(nl, profile.patterns)[:5]
    if not ranked:
        return ""
    return "\n".join(
        f"{score:>3}  {p.pattern_id:<22} {p.address}" for p, score in ranked
    )


def _result_html(badge: str, title: str, meta: str, body: str) -> str:
    return (
        f"<p><strong>{badge}</strong> — {title}</p>"
        f"<p><small>{meta}</small></p>"
        f"<p><code>{body}</code></p>"
    )


def _acceptance_html(device_id: str, note: str = "") -> str:
    try:
        ensure_suite(device_id)
        panel = status_panel(device_id)
    except Exception as exc:
        return (
            "<div class='acceptance-panel'>"
            "<h3>Profile Acceptance</h3>"
            f"<p><strong>error</strong> — {exc}</p>"
            "</div>"
        )
    counts = panel["counts"]
    meta = panel.get("meta") or {}
    report = panel.get("last_report")
    if report:
        passed = report.get("gates", {}).get("passed")
        badge = "PASS" if passed else "FAIL"
        summary = summarize_report(report)
    else:
        badge = "idle"
        summary = "No acceptance run yet — pin cases, then Run acceptance."
    note_html = f"<p class='muted'>{note}</p>" if note else ""
    return (
        "<div class='acceptance-panel'>"
        "<h3>Profile Acceptance</h3>"
        "<p class='muted'>Pin rehearsal cases to this device · "
        "commit only when wrong-send stays 0%</p>"
        f"<p><strong>{badge}</strong> — {summary}</p>"
        f"<p><small>{device_id} · "
        f"{counts.get('nl_cases', 0)} must-work · "
        f"{counts.get('refusal_cases', 0)} must-refuse · "
        f"suite @ {meta.get('profile_version', '—')}</small></p>"
        f"{note_html}"
        "</div>"
    )


def resolve_demo(
    profile_label: str,
    backend: str,
    nl: str,
    retrieval_gate: bool,
) -> tuple[str, str, str, str, dict[str, Any] | None, str]:
    empty_state: dict[str, Any] | None = None
    nl = (nl or "").strip()
    device_id = _label_to_device(profile_label)
    acceptance = _acceptance_html(device_id)

    if not nl:
        return (
            _result_html(
                "standby",
                "Awaiting input",
                "dry-run",
                "Enter NL or pick an example.",
            ),
            "",
            "",
            "",
            empty_state,
            acceptance,
        )

    try:
        profile = find_committed_profile(device_id)
    except Exception as exc:
        return (
            _result_html("error", "Profile error", device_id, str(exc)),
            "",
            "",
            "",
            empty_state,
            acceptance,
        )

    backend_key: Backend = _backend_value(backend)  # type: ignore[assignment]
    if backend_key not in ("b0", "b1", "b2", "b3"):
        return (
            _result_html("error", "Invalid backend", "", backend),
            "",
            "",
            "",
            empty_state,
            acceptance,
        )

    gate = True if backend_key == "b0" else bool(retrieval_gate)
    meta = (
        f"{backend_key} · gate={'on' if gate else 'off'} · "
        f"{profile.device_id} · dry-run"
    )

    try:
        result = resolve_nl(nl, profile, backend=backend_key, retrieval_gate=gate)
    except Exception as exc:
        return (
            _result_html("error", "Resolve failed", meta, str(exc)),
            "",
            "",
            _retrieval_preview(nl, profile),
            empty_state,
            acceptance,
        )

    intent_json = json.dumps(result.model_dump(mode="json"), indent=2)
    retrieval = _retrieval_preview(nl, profile)
    last_case = {
        "device_id": device_id,
        "nl": nl,
        "result": result.model_dump(mode="json"),
    }

    if isinstance(result, RefusalIntent):
        cand = (" · " + ", ".join(result.candidates)) if result.candidates else ""
        return (
            _result_html(
                "refused",
                result.reason.value.replace("_", " "),
                meta,
                f"{result.message}{cand}",
            ),
            intent_json,
            "",
            retrieval,
            last_case,
            acceptance,
        )

    assert isinstance(result, SuccessIntent)
    try:
        pipeline = run_pipeline(
            result.model_dump(mode="json"), profile, dry_run=True
        )
        osc_line = f"{pipeline.preview.address} {pipeline.preview.args}"
        badge = (
            "would-send · ungated"
            if (backend_key != "b0" and not gate)
            else "would-send"
        )
        return (
            _result_html(
                badge,
                f"{result.pattern_id} → {result.address}",
                meta,
                f"args {result.args}",
            ),
            intent_json,
            osc_line,
            retrieval,
            last_case,
            acceptance,
        )
    except ValidationError as exc:
        return (
            _result_html(
                "tier-3 block",
                exc.reason.value.replace("_", " "),
                meta,
                exc.message,
            ),
            intent_json,
            "",
            retrieval,
            last_case,
            acceptance,
        )


def _pin(
    last_case: dict[str, Any] | None,
    *,
    as_refusal: bool,
) -> tuple[str, str]:
    if not last_case:
        return (
            _acceptance_html("max-msp", "Resolve a phrase first, then pin."),
            "No preview to pin — resolve NL first.",
        )
    device_id = last_case["device_id"]
    nl = last_case["nl"]
    try:
        result = parse_intent(last_case["result"])
        profile = find_committed_profile(device_id)
        path = pin_from_result(
            device_id,
            nl,
            result,
            as_refusal=as_refusal,
            profile=profile,
        )
        kind = (
            "must-refuse"
            if as_refusal or isinstance(result, RefusalIntent)
            else "must-work"
        )
        note = f"Pinned {kind}: {path.name}"
        return _acceptance_html(device_id, note), note
    except (AcceptanceError, Exception) as exc:
        return _acceptance_html(device_id, str(exc)), f"Pin failed: {exc}"


def _run_acceptance(profile_label: str) -> tuple[str, str]:
    device_id = _label_to_device(profile_label)
    try:
        report = run_acceptance(device_id, backend="b0")
        note = summarize_report(report)
        return _acceptance_html(device_id, "Last run saved to last_report.json"), note
    except Exception as exc:
        return _acceptance_html(device_id, str(exc)), f"Acceptance failed: {exc}"


def _check_commit_gate(profile_label: str) -> tuple[str, str]:
    """Simulate commit gate against the current committed profile + suite."""
    device_id = _label_to_device(profile_label)
    try:
        profile = find_committed_profile(device_id)
        report = run_acceptance(device_id, profile=profile, backend="b0")
        if report.get("gates", {}).get("passed"):
            note = (
                f"Commit OK for {device_id} @ {profile.profile_version} — "
                f"{summarize_report(report)}"
            )
        else:
            note = (
                f"Commit blocked for {device_id} @ {profile.profile_version} — "
                f"{summarize_report(report)}"
            )
        return _acceptance_html(device_id, note), note
    except Exception as exc:
        return _acceptance_html(device_id, str(exc)), f"Gate check failed: {exc}"


def _refresh_acceptance(profile_label: str) -> str:
    return _acceptance_html(_label_to_device(profile_label))


def build_ui():
    try:
        import gradio as gr
    except ImportError as exc:
        raise SystemExit(f'Gradio required. pip install -e ".[demo]"\n({exc})') from exc

    choices = _profile_choices()
    default_profile = choices[0]
    default_device = _label_to_device(default_profile)
    ensure_suite(default_device)
    lora_note = "B3 LoRA ready" if default_adapter_path().is_dir() else "B3 needs local LoRA"

    with gr.Blocks(title="LLM4OSC", analytics_enabled=False) as demo:
        gr.HTML(f"<style>{LAYOUT_CSS}</style>")
        gr.HTML(
            "<div class='flow-ascii-wrap'>"
            "<pre class='flow-ascii'>"
            "                ┌──────────┐\n"
            "Prompt (NL) ──▶ │ LLM4OSC  │ ──▶ OSC (UDP)\n"
            "                └──────────┘"
            "</pre>"
            "</div>"
        )

        last_case = gr.State(None)

        with gr.Row():
            profile = gr.Dropdown(
                choices=choices,
                value=default_profile,
                label="Profile",
                scale=2,
                elem_classes=["profile-menu"],
            )
            backend = gr.Dropdown(
                choices=BACKEND_CHOICES,
                value="b0",
                label="Backend",
                scale=4,
                elem_classes=["backend-menu"],
            )
            gate = gr.Checkbox(
                value=True,
                label="Retrieval gate (B1–B3)",
                interactive=False,
                scale=1,
            )

        with gr.Row(elem_classes=["query-row"]):
            with gr.Column(elem_classes=["nl-panel"], scale=4):
                with gr.Row(elem_classes=["nl-label-row"]):
                    gr.HTML("<p class='nl-label'>Natural language</p>")
                    example_buttons = []
                    for lbl, phrase in zip(EXAMPLE_LABELS, EXAMPLE_PHRASES):
                        btn = gr.Button(lbl, size="sm")
                        example_buttons.append((btn, phrase))
                nl = gr.Textbox(
                    show_label=False,
                    placeholder="make the level half",
                    lines=2,
                    autofocus=True,
                    container=True,
                    elem_classes=["nl-input"],
                )
            run = gr.Button(
                "Resolve",
                variant="primary",
                scale=0,
                elem_classes=["resolve-btn"],
            )

        status = gr.HTML(
            _result_html(
                "ready",
                "Idle",
                f"dry-run · {lora_note}",
                "Enter a phrase or pick an example.",
            )
        )

        with gr.Row():
            osc_out = gr.Textbox(label="OSC preview", lines=2)
            retrieval_out = gr.Textbox(label="Retrieval scores", lines=2)

        intent_out = gr.Code(label="Intent JSON", language="json", lines=8)

        acceptance_html = gr.HTML(_acceptance_html(default_device))
        with gr.Row():
            pin_work = gr.Button("Pin as must-work", scale=1)
            pin_refuse = gr.Button("Pin as must-refuse", scale=1)
            run_acc = gr.Button("Run acceptance", variant="primary", scale=1)
            check_gate = gr.Button("Check commit gate", scale=1)
        action_note = gr.Textbox(
            label="Acceptance action",
            lines=2,
            interactive=False,
            value="Preview → pin → run acceptance → check commit gate.",
        )

        def _gate_for_backend(backend_choice: str):
            key = _backend_value(backend_choice)
            if key == "b0":
                return gr.update(interactive=False, value=True)
            return gr.update(interactive=True)

        backend.change(
            fn=_gate_for_backend,
            inputs=[backend],
            outputs=[gate],
        )
        profile.change(
            fn=_refresh_acceptance,
            inputs=[profile],
            outputs=[acceptance_html],
        )

        for _btn, phrase in example_buttons:
            _btn.click(fn=lambda p=phrase: p, outputs=nl)

        outputs = [
            status,
            intent_out,
            osc_out,
            retrieval_out,
            last_case,
            acceptance_html,
        ]
        run.click(
            fn=resolve_demo,
            inputs=[profile, backend, nl, gate],
            outputs=outputs,
        )
        nl.submit(
            fn=resolve_demo,
            inputs=[profile, backend, nl, gate],
            outputs=outputs,
        )

        pin_work.click(
            fn=lambda case: _pin(case, as_refusal=False),
            inputs=[last_case],
            outputs=[acceptance_html, action_note],
        )
        pin_refuse.click(
            fn=lambda case: _pin(case, as_refusal=True),
            inputs=[last_case],
            outputs=[acceptance_html, action_note],
        )
        run_acc.click(
            fn=_run_acceptance,
            inputs=[profile],
            outputs=[acceptance_html, action_note],
        )
        check_gate.click(
            fn=_check_commit_gate,
            inputs=[profile],
            outputs=[acceptance_html, action_note],
        )

    return demo


def main() -> None:
    build_ui().launch(
        server_name="127.0.0.1",
        server_port=7860,
        css=LAYOUT_CSS,
    )


if __name__ == "__main__":
    main()
