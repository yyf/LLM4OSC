from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Protocol

from llm4osc.models import (
    DeviceProfile,
    PatternRecord,
    RefusalIntent,
    RefusalReason,
    SuccessIntent,
    parse_intent,
)
from llm4osc.profile import repo_root
from llm4osc.prompt import build_messages
from llm4osc.retrieval import patterns_for_context, rank_patterns
from llm4osc.slots import fill_pattern_args, normalize_unit_float_args
from tier3.validate import ValidationError, validate_intent

DEFAULT_MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
DEFAULT_ADAPTER_DIR = repo_root() / "models" / "qwen2-0.5b-osc" / "adapter"
FEW_SHOT_PATH = repo_root() / "benchmarks" / "few_shot_examples.json"


class IntentLLM(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str: ...


class LLMNotAvailableError(RuntimeError):
    pass


def _debug(msg: str) -> None:
    if os.environ.get("LLM4OSC_DEBUG", "").lower() in ("1", "true", "yes"):
        print(f"[llm4osc] {msg}", file=sys.stderr)


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found in model output")
    return json.loads(cleaned[start : end + 1])


def load_few_shot_examples(device_id: str | None = None) -> list[dict[str, Any]]:
    if not FEW_SHOT_PATH.is_file():
        return []
    items = json.loads(FEW_SHOT_PATH.read_text(encoding="utf-8"))
    if device_id:
        items = [ex for ex in items if ex.get("device_id", device_id) == device_id]
    return items


_MODEL: "QwenIntentModel | None" = None
_MODEL_KEY: tuple[str, str | None] | None = None


def _resolve_adapter_path(adapter_path: str | Path | None) -> Path | None:
    raw = adapter_path or os.environ.get("LLM4OSC_ADAPTER")
    if raw is None or str(raw).strip() == "":
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = repo_root() / path
    return path if path.is_dir() else None


def default_adapter_path() -> Path:
    return DEFAULT_ADAPTER_DIR


def get_qwen_model(
    model_id: str | None = None,
    *,
    adapter_path: str | Path | None = None,
) -> "QwenIntentModel":
    global _MODEL, _MODEL_KEY
    resolved = model_id or os.environ.get("LLM4OSC_MODEL", DEFAULT_MODEL_ID)
    adapter = _resolve_adapter_path(adapter_path)
    adapter_key = str(adapter) if adapter else None
    key = (resolved, adapter_key)
    if _MODEL is None or _MODEL_KEY != key:
        _MODEL = QwenIntentModel(resolved, adapter_path=adapter)
        _MODEL_KEY = key
    return _MODEL


class QwenIntentModel:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        adapter_path: Path | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise LLMNotAvailableError(
                "LLM extras not installed. Run: python -m pip install -e '.[llm]'"
            ) from exc

        self.model_id = model_id
        self.adapter_path = adapter_path
        self._torch = torch
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
        )
        if adapter_path is not None:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise LLMNotAvailableError(
                    "LoRA adapter requires peft. Run: python -m pip install -e '.[llm]'"
                ) from exc
            self.model = PeftModel.from_pretrained(self.model, str(adapter_path))
        self.model.to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def complete(self, messages: list[dict[str, str]], *, max_new_tokens: int = 256) -> str:
        torch = self._torch
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


VALID_REFUSAL_REASONS = {r.value for r in RefusalReason}

# Common Qwen paraphrases → schema enum
_REFUSAL_REASON_ALIASES: dict[str, str] = {
    "no_matching_pattern": "unknown_pattern",
    "no_match": "unknown_pattern",
    "not_found": "unknown_pattern",
    "no_pattern": "unknown_pattern",
    "unknown": "unknown_pattern",
    "ambiguous": "ambiguous_pattern",
    "multiple_matches": "ambiguous_pattern",
    "ambiguous_pattern_match": "ambiguous_pattern",
    "out_of_range": "out_of_range",
    "out of range": "out_of_range",
    "missing_slot": "missing_slot",
    "missing_param": "missing_slot",
    "missing_value": "missing_slot",
    "missing_parameter": "missing_slot",
    "unsupported": "unsupported_command",
    "unsupported_command": "unsupported_command",
    "unknown_command": "unsupported_command",
    "invalid": "invalid_input",
    "invalid_input": "invalid_input",
    "unknown_device": "unknown_device",
    "unknown_pattern": "unknown_pattern",
    "ambiguous_pattern": "ambiguous_pattern",
}


def _normalize_refusal_reason(data: dict[str, Any]) -> None:
    if data.get("kind") != "refusal":
        return
    reason = data.get("reason")
    if not reason or not isinstance(reason, str):
        return
    key = reason.strip().lower().replace(" ", "_").replace("-", "_")
    if key in VALID_REFUSAL_REASONS:
        data["reason"] = key
        return
    mapped = _REFUSAL_REASON_ALIASES.get(key)
    if mapped:
        data["reason"] = mapped
        return
    data["reason"] = "invalid_input"


def _infer_kind(data: dict[str, Any]) -> None:
    kind = data.get("kind")
    if kind in ("intent", "refusal"):
        return
    if data.get("reason"):
        data["kind"] = "refusal"
    elif data.get("pattern_id"):
        data["kind"] = "intent"


def _enrich_from_profile(data: dict[str, Any], profile: DeviceProfile) -> None:
    if data.get("kind") != "intent":
        return
    pattern_id = data.get("pattern_id")
    if not pattern_id:
        return
    pattern = next(
        (p for p in profile.patterns if p.pattern_id == pattern_id),
        None,
    )
    if pattern is None:
        return
    data.setdefault("address", pattern.address)
    data.setdefault("type_tags", pattern.type_tags)
    data.setdefault("args", [])


def _refine_intent_with_nl(
    data: dict[str, Any],
    profile: DeviceProfile,
    nl: str,
) -> None:
    """Profile-driven correction: retrieval + deterministic slot fill."""
    if data.get("kind") != "intent":
        return

    ranked = rank_patterns(nl, profile.patterns)
    if not ranked:
        return

    top_pattern, top_score = ranked[0]
    llm_pid = data.get("pattern_id")
    llm_score = next((s for p, s in ranked if p.pattern_id == llm_pid), 0)

    pattern = next(
        (p for p in profile.patterns if p.pattern_id == llm_pid),
        None,
    )
    if (
        top_score >= 3
        and top_pattern.pattern_id != llm_pid
        and (llm_score == 0 or top_score >= llm_score + 3)
    ):
        pattern = top_pattern
        data["pattern_id"] = top_pattern.pattern_id
        data["address"] = top_pattern.address
        data["type_tags"] = top_pattern.type_tags

    if pattern is None:
        return

    filled = fill_pattern_args(pattern, nl)
    if filled is not None:
        data["args"] = filled
    elif data.get("args"):
        data["args"] = normalize_unit_float_args(list(data["args"]), pattern)


def _normalize_parsed(
    data: dict[str, Any],
    profile: DeviceProfile,
    nl: str | None = None,
) -> dict[str, Any]:
    data.setdefault("schema_version", "1.0")
    data.setdefault("device_id", profile.device_id)
    data.setdefault("profile_version", profile.profile_version)
    _infer_kind(data)
    _normalize_refusal_reason(data)
    _enrich_from_profile(data, profile)
    if nl:
        _refine_intent_with_nl(data, profile, nl)
    return data


def _parse_llm_output(
    raw: str,
    profile: DeviceProfile,
    nl: str | None = None,
) -> SuccessIntent | RefusalIntent:
    data = _normalize_parsed(extract_json_object(raw), profile, nl)
    return parse_intent(data)


def resolve_nl_llm(
    text: str,
    profile: DeviceProfile,
    *,
    few_shot: bool = False,
    llm: IntentLLM | None = None,
    model_id: str | None = None,
    adapter_path: str | Path | None = None,
    top_k: int = 8,
) -> SuccessIntent | RefusalIntent:
    patterns = patterns_for_context(text, profile.patterns, k=top_k)
    few_shot_examples = load_few_shot_examples(profile.device_id) if few_shot else []
    messages = build_messages(
        text,
        profile,
        patterns,
        few_shot_examples=few_shot_examples,
    )

    model = llm or get_qwen_model(model_id, adapter_path=adapter_path)
    last_error: str | None = None

    for attempt in range(2):
        attempt_messages = list(messages)
        if last_error and attempt == 1:
            attempt_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Previous output failed validation: "
                        f"{last_error}. Output corrected JSON only. "
                        f"Refusal reason must be one of: "
                        f"{', '.join(sorted(VALID_REFUSAL_REASONS))}."
                    ),
                }
            )
        try:
            raw = model.complete(attempt_messages)
            _debug(f"attempt {attempt + 1} raw output:\n{raw}")
            parsed = _parse_llm_output(raw, profile, text)
        except Exception as exc:
            last_error = str(exc)
            _debug(f"attempt {attempt + 1} parse error: {last_error}")
            continue

        if isinstance(parsed, RefusalIntent):
            return parsed

        try:
            validate_intent(parsed, profile)
            return parsed
        except ValidationError as exc:
            last_error = f"{exc.reason.value}: {exc.message}"
            _debug(f"attempt {attempt + 1} tier3 error: {last_error}")

    _debug(f"failed after retry; last_error={last_error!r}")

    return RefusalIntent(
        device_id=profile.device_id,
        profile_version=profile.profile_version,
        reason=RefusalReason.INVALID_INPUT,
        message=f"LLM output failed validation after retry: {last_error}",
    )
