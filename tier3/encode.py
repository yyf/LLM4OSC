from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pythonosc import osc_message_builder
from pythonosc.osc_message import OscMessage

from llm4osc.models import SuccessIntent


class EncodeError(ValueError):
    pass


def _coerce_arg(value: Any, tag: str) -> Any:
    if tag == "i":
        return int(value)
    if tag == "f":
        return float(value)
    if tag == "s":
        return str(value)
    if tag == "T":
        return True
    if tag == "F":
        return False
    raise EncodeError(f"Unsupported OSC type tag: {tag!r}")


def build_message(intent: SuccessIntent) -> OscMessage:
    builder = osc_message_builder.OscMessageBuilder(address=intent.address)
    tags = intent.type_tags
    if len(tags) != len(intent.args):
        raise EncodeError(
            f"type_tags length {len(tags)} != args length {len(intent.args)}"
        )
    for value, tag in zip(intent.args, tags):
        builder.add_arg(_coerce_arg(value, tag))
    return builder.build()


def encode_intent(intent: SuccessIntent) -> bytes:
    return build_message(intent).dgram


def decode_packet(data: bytes) -> tuple[str, list[Any]]:
    msg = OscMessage(data)
    return msg.address, list(msg.params)


@dataclass(frozen=True)
class OscPreview:
    address: str
    type_tags: str
    args: list[Any]
    dgram: bytes


def preview_intent(intent: SuccessIntent) -> OscPreview:
    msg = build_message(intent)
    return OscPreview(
        address=msg.address,
        type_tags=intent.type_tags,
        args=list(msg.params),
        dgram=msg.dgram,
    )
