from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field


class RefusalReason(str, Enum):
    UNKNOWN_DEVICE = "unknown_device"
    UNKNOWN_PATTERN = "unknown_pattern"
    AMBIGUOUS_PATTERN = "ambiguous_pattern"
    OUT_OF_RANGE = "out_of_range"
    MISSING_SLOT = "missing_slot"
    UNSUPPORTED_COMMAND = "unsupported_command"
    INVALID_INPUT = "invalid_input"


class SlotSpec(BaseModel):
    name: str
    type: Literal["int", "float", "string"]
    arg_index: int = Field(ge=0)


class RangeSpec(BaseModel):
    min: float
    max: float


class PatternRecord(BaseModel):
    pattern_id: str
    address: str
    type_tags: str
    description: str
    ranges: dict[str, RangeSpec] = Field(default_factory=dict)
    slots: list[SlotSpec] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    manual_ref: str | None = None


class ManualSource(BaseModel):
    ref: str
    hash: str | None = None


class DeviceProfile(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    device_id: str
    profile_version: str
    patterns: list[PatternRecord]
    manual_sources: list[ManualSource] = Field(default_factory=list)
    reviewed_at: str | None = None
    reviewed_by: str | None = None


class SuccessIntent(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["intent"] = "intent"
    device_id: str
    profile_version: str
    pattern_id: str
    address: str
    type_tags: str
    args: list[Any]
    bundle_id: str | None = None
    stream_spec: dict[str, Any] | None = None
    confidence: float | None = None


class RefusalIntent(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["refusal"] = "refusal"
    device_id: str
    profile_version: str
    reason: RefusalReason
    message: str
    candidates: list[str] = Field(default_factory=list)


Intent = Annotated[Union[SuccessIntent, RefusalIntent], Field(discriminator="kind")]


def parse_intent(data: dict[str, Any]) -> SuccessIntent | RefusalIntent:
    kind = data.get("kind")
    if kind == "intent":
        return SuccessIntent.model_validate(data)
    if kind == "refusal":
        return RefusalIntent.model_validate(data)
    raise ValueError(f"Unknown intent kind: {kind!r}")
