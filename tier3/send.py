from __future__ import annotations

import os
import socket
from dataclasses import dataclass


@dataclass(frozen=True)
class SendTarget:
    host: str
    port: int


def default_target() -> SendTarget:
    host = os.environ.get("LLM4OSC_HOST", "127.0.0.1")
    port = int(os.environ.get("LLM4OSC_PORT", "7400"))
    return SendTarget(host=host, port=port)


def send_bytes(data: bytes, target: SendTarget | None = None) -> None:
    tgt = target or default_target()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.sendto(data, (tgt.host, tgt.port))
