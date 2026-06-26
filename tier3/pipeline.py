from __future__ import annotations

import hashlib
from dataclasses import dataclass

from llm4osc.models import DeviceProfile, RefusalIntent, SuccessIntent, parse_intent
from tier3.clamp import clamp_intent
from tier3.encode import OscPreview, encode_intent, preview_intent
from tier3.send import SendTarget, send_bytes
from tier3.validate import ValidationError, validate_intent


@dataclass(frozen=True)
class PipelineResult:
    intent: SuccessIntent
    preview: OscPreview
    intent_hash: str
    sent: bool


def intent_hash(intent: SuccessIntent) -> str:
    payload = (
        f"{intent.profile_version}|{intent.pattern_id}|"
        f"{intent.address}|{intent.type_tags}|{intent.args!r}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def run_pipeline(
    intent_data: dict,
    profile: DeviceProfile,
    *,
    dry_run: bool = False,
    target: SendTarget | None = None,
) -> PipelineResult:
    parsed = parse_intent(intent_data)
    if isinstance(parsed, RefusalIntent):
        raise ValidationError(parsed.reason, parsed.message)

    validate_intent(parsed, profile)
    clamped = clamp_intent(parsed, profile)
    preview = preview_intent(clamped)
    digest = intent_hash(clamped)

    if not dry_run:
        send_bytes(preview.dgram, target=target)

    return PipelineResult(
        intent=clamped,
        preview=preview,
        intent_hash=digest,
        sent=not dry_run,
    )
