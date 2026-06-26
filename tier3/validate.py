from __future__ import annotations

from llm4osc.models import DeviceProfile, RefusalReason, SuccessIntent


class ValidationError(Exception):
    def __init__(self, reason: RefusalReason, message: str) -> None:
        self.reason = reason
        self.message = message
        super().__init__(message)


def _expected_arg_count(type_tags: str) -> int:
    return len(type_tags)


def validate_intent(intent: SuccessIntent, profile: DeviceProfile) -> None:
    if intent.device_id != profile.device_id:
        raise ValidationError(
            RefusalReason.UNKNOWN_DEVICE,
            f"Intent device_id {intent.device_id!r} != profile {profile.device_id!r}",
        )
    if intent.profile_version != profile.profile_version:
        raise ValidationError(
            RefusalReason.INVALID_INPUT,
            f"Intent profile_version {intent.profile_version!r} != "
            f"profile {profile.profile_version!r}",
        )

    pattern = next(
        (p for p in profile.patterns if p.pattern_id == intent.pattern_id),
        None,
    )
    if pattern is None:
        raise ValidationError(
            RefusalReason.UNKNOWN_PATTERN,
            f"Unknown pattern_id: {intent.pattern_id!r}",
        )
    if intent.address != pattern.address:
        raise ValidationError(
            RefusalReason.INVALID_INPUT,
            f"Intent address {intent.address!r} != pattern address {pattern.address!r}",
        )
    if intent.type_tags != pattern.type_tags:
        raise ValidationError(
            RefusalReason.INVALID_INPUT,
            f"Intent type_tags {intent.type_tags!r} != pattern {pattern.type_tags!r}",
        )

    expected = _expected_arg_count(pattern.type_tags)
    if len(intent.args) != expected:
        raise ValidationError(
            RefusalReason.MISSING_SLOT,
            f"Expected {expected} args, got {len(intent.args)}",
        )

    for slot in pattern.slots:
        if slot.arg_index >= len(intent.args):
            raise ValidationError(
                RefusalReason.MISSING_SLOT,
                f"Missing slot {slot.name!r} at arg_index {slot.arg_index}",
            )
        value = intent.args[slot.arg_index]
        if slot.type == "int" and not isinstance(value, int):
            raise ValidationError(
                RefusalReason.INVALID_INPUT,
                f"Slot {slot.name!r} expects int",
            )
        if slot.type == "float" and not isinstance(value, (int, float)):
            raise ValidationError(
                RefusalReason.INVALID_INPUT,
                f"Slot {slot.name!r} expects float",
            )
        if slot.type == "string" and not isinstance(value, str):
            raise ValidationError(
                RefusalReason.INVALID_INPUT,
                f"Slot {slot.name!r} expects string",
            )
