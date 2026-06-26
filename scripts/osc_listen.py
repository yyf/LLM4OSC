#!/usr/bin/env python3
"""UDP OSC loopback listener for demos."""

from __future__ import annotations

import argparse
import os
import socket

from pythonosc.osc_message import OscMessage


def main() -> int:
    parser = argparse.ArgumentParser(description="Listen for OSC UDP packets")
    parser.add_argument(
        "--host",
        default=os.environ.get("LLM4OSC_HOST", "127.0.0.1"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("LLM4OSC_PORT", "7400")),
    )
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    print(f"Listening on {args.host}:{args.port} (Ctrl+C to stop)")

    while True:
        data, addr = sock.recvfrom(4096)
        try:
            msg = OscMessage(data)
            print(f"{addr[0]}:{addr[1]}  {msg.address}  {list(msg.params)}")
        except Exception:
            print(f"{addr[0]}:{addr[1]}  raw {data!r}")


if __name__ == "__main__":
    raise SystemExit(main())
