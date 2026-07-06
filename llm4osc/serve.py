from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from llm4osc.llm import default_adapter_path, get_qwen_model
from llm4osc.models import RefusalIntent, SuccessIntent, parse_intent
from llm4osc.profile import find_committed_profile
from llm4osc.resolver import Backend, resolve_nl

DEFAULT_SERVE_HOST = "127.0.0.1"
DEFAULT_SERVE_PORT = 8765


def serve_url(explicit: str | None = None) -> str | None:
    url = explicit or os.environ.get("LLM4OSC_SERVE_URL")
    if not url:
        return None
    return url.rstrip("/")


def handle_resolve(body: dict[str, Any]) -> dict[str, Any]:
    nl = body.get("nl")
    if not nl or not isinstance(nl, str):
        raise ValueError("missing or invalid 'nl'")

    device_id = body.get("device_id", "max-msp")
    backend = body.get("backend", "b1")
    if backend not in ("b0", "b1", "b2", "b3"):
        raise ValueError("backend must be b0, b1, b2, or b3")

    profile = find_committed_profile(device_id)
    result = resolve_nl(
        nl,
        profile,
        backend=backend,  # type: ignore[arg-type]
        model_id=body.get("model_id"),
        adapter_path=body.get("adapter_path"),
        serve_url=None,
    )
    return result.model_dump(mode="json")


def resolve_remote(
    base_url: str,
    nl: str,
    profile_device_id: str,
    *,
    backend: Backend = "b1",
    model_id: str | None = None,
    adapter_path: str | None = None,
) -> SuccessIntent | RefusalIntent:
    payload = json.dumps(
        {
            "device_id": profile_device_id,
            "nl": nl,
            "backend": backend,
            "model_id": model_id,
            "adapter_path": adapter_path,
        }
    ).encode("utf-8")
    req = Request(
        f"{base_url.rstrip('/')}/v1/resolve",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"serve HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"serve unreachable at {base_url}: {exc.reason}") from exc

    if not data.get("ok"):
        raise RuntimeError(data.get("error", "serve request failed"))
    return parse_intent(data["result"])


def preload_model(
    model_id: str | None = None,
    *,
    adapter_path: str | None = None,
) -> str:
    resolved_adapter = adapter_path
    if resolved_adapter is None and default_adapter_path().is_dir():
        resolved_adapter = str(default_adapter_path())
    model = get_qwen_model(model_id, adapter_path=resolved_adapter)
    return model.model_id


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class _ServeHandler(BaseHTTPRequestHandler):
    model_id: str | None = None
    adapter_path: str | None = None

    def log_message(self, format: str, *args: Any) -> None:
        if os.environ.get("LLM4OSC_DEBUG", "").lower() in ("1", "true", "yes"):
            sys.stderr.write(f"[serve] {self.address_string()} - {format % args}\n")

    def do_GET(self) -> None:
        if self.path == "/health":
            _json_response(
                self,
                200,
                {
                    "ok": True,
                    "model_id": getattr(self.server, "model_id", None),
                    "adapter_path": getattr(self.server, "adapter_path", None),
                },
            )
            return
        _json_response(self, 404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/v1/resolve":
            _json_response(self, 404, {"ok": False, "error": "not found"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
            result = handle_resolve(body)
            _json_response(self, 200, {"ok": True, "result": result})
        except Exception as exc:
            _json_response(self, 400, {"ok": False, "error": str(exc)})


def run_server(
    host: str = DEFAULT_SERVE_HOST,
    port: int = DEFAULT_SERVE_PORT,
    *,
    model_id: str | None = None,
    adapter_path: str | None = None,
    preload: bool = True,
) -> None:
    if preload:
        loaded = preload_model(model_id, adapter_path=adapter_path)
        print(f"Loaded model: {loaded}", file=sys.stderr)
        if adapter_path or default_adapter_path().is_dir():
            ap = adapter_path or str(default_adapter_path())
            print(f"Adapter: {ap}", file=sys.stderr)

    server = ThreadingHTTPServer((host, port), _ServeHandler)
    server.model_id = model_id or os.environ.get("LLM4OSC_MODEL")  # type: ignore[attr-defined]
    server.adapter_path = adapter_path or os.environ.get("LLM4OSC_ADAPTER")  # type: ignore[attr-defined]
    print(f"LLM4OSC serve listening on http://{host}:{port}", file=sys.stderr)
    print("  GET  /health", file=sys.stderr)
    print("  POST /v1/resolve  {device_id, nl, backend, model_id?, adapter_path?}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", file=sys.stderr)
        server.shutdown()
