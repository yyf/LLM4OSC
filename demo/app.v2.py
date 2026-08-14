"""Demo UI — plain Gradio + simple HTML.

Launch:
  pip install -e ".[demo]"
  python demo/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from llm4osc.llm import default_adapter_path
from llm4osc.models import RefusalIntent, SuccessIntent
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
  margin: 0;
  padding: 0;
  font-size: var(--block-title-text-size, var(--text-md, 0.95rem));
  font-weight: var(--block-title-text-weight, 400);
  line-height: var(--line-sm, 1.4);
  color: var(--block-label-text-color, var(--body-text-color));
  white-space: nowrap;
}
.nl-label-row button {
  min-height: 0 !important;
  padding: 0.15rem 0.4rem !important;
  font-size: 0.75rem !important;
  margin: 0 !important;
}
.nl-panel .nl-input {
  margin: 0 !important;
  padding: 0 !important;
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
}
.nl-panel .nl-input > .label-wrap {
  display: none !important;
}
.query-row .resolve-btn {
  display: flex !important;
  align-items: stretch !important;
  align-self: stretch !important;
  margin: 0 !important;
  min-width: 5.5rem !important;
  height: auto !important;
}
.query-row .resolve-btn > .wrap,
.query-row .resolve-btn button {
  height: 100% !important;
  min-height: 100% !important;
  width: 100% !important;
  margin: 0 !important;
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


def resolve_demo(
    profile_label: str,
    backend: str,
    nl: str,
    retrieval_gate: bool,
) -> tuple[str, str, str, str]:
    nl = (nl or "").strip()
    if not nl:
        return (
            _result_html("standby", "Awaiting input", "dry-run", "Enter NL or pick an example."),
            "", "", "",
        )

    device_id = _label_to_device(profile_label)
    try:
        profile = find_committed_profile(device_id)
    except Exception as exc:
        return (
            _result_html("error", "Profile error", device_id, str(exc)),
            "", "", "",
        )

    backend_key: Backend = _backend_value(backend)  # type: ignore[assignment]
    if backend_key not in ("b0", "b1", "b2", "b3"):
        return (
            _result_html("error", "Invalid backend", "", backend),
            "", "", "",
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
            "", "", _retrieval_preview(nl, profile),
        )

    intent_json = json.dumps(result.model_dump(mode="json"), indent=2)
    retrieval = _retrieval_preview(nl, profile)

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
        )

    assert isinstance(result, SuccessIntent)
    try:
        pipeline = run_pipeline(
            result.model_dump(mode="json"), profile, dry_run=True
        )
        osc_line = f"{pipeline.preview.address} {pipeline.preview.args}"
        badge = "would-send · ungated" if (backend_key != "b0" and not gate) else "would-send"
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
        )


def build_ui():
    try:
        import gradio as gr
    except ImportError as exc:
        raise SystemExit(f'Gradio required. pip install -e ".[demo]"\n({exc})') from exc

    choices = _profile_choices()
    default_profile = choices[0]
    lora_note = "B3 LoRA ready" if default_adapter_path().is_dir() else "B3 needs local LoRA"

    with gr.Blocks(title="LLM4OSC", analytics_enabled=False) as demo:
        gr.HTML(f"<style>{LAYOUT_CSS}</style>")
        gr.HTML(
            "<h1>LLM4OSC</h1>"
            "<p>Natural language → intent JSON → Tier 3 dry-run. UDP is not sent.</p>"
        )

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

        for _btn, phrase in example_buttons:
            _btn.click(fn=lambda p=phrase: p, outputs=nl)

        run.click(
            fn=resolve_demo,
            inputs=[profile, backend, nl, gate],
            outputs=[status, intent_out, osc_out, retrieval_out],
        )
        nl.submit(
            fn=resolve_demo,
            inputs=[profile, backend, nl, gate],
            outputs=[status, intent_out, osc_out, retrieval_out],
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
