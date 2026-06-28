from __future__ import annotations

import json
import threading
from urllib.request import urlopen

import pytest

from llm4osc.models import SuccessIntent
from llm4osc.profile import find_committed_profile
from llm4osc.resolver import resolve_nl
from llm4osc.serve import handle_resolve, resolve_remote, run_server

PROFILE = find_committed_profile("max-msp")


def test_handle_resolve_b0() -> None:
    result = handle_resolve(
        {
            "device_id": "max-msp",
            "nl": "set gain to 50%",
            "backend": "b0",
        }
    )
    assert result["kind"] == "intent"
    assert result["pattern_id"] == "gain_set"
    assert result["args"] == [0.5]


def test_handle_resolve_missing_nl() -> None:
    with pytest.raises(ValueError, match="nl"):
        handle_resolve({"device_id": "max-msp", "backend": "b0"})


@pytest.fixture
def serve_thread():
    import llm4osc.serve as serve_mod

    original = serve_mod.preload_model

    def _noop(_model_id: str | None = None) -> str:
        return "mock"

    serve_mod.preload_model = _noop  # type: ignore[assignment]
    thread = threading.Thread(
        target=run_server,
        kwargs={"host": "127.0.0.1", "port": 18765, "preload": False},
        daemon=True,
    )
    thread.start()
    yield "http://127.0.0.1:18765"
    serve_mod.preload_model = original  # type: ignore[assignment]


def test_serve_http_b0(serve_thread: str) -> None:
    import time

    for _ in range(50):
        try:
            with urlopen(f"{serve_thread}/health", timeout=0.2) as resp:
                health = json.loads(resp.read().decode())
            break
        except OSError:
            time.sleep(0.05)
    else:
        pytest.fail("serve did not start")

    assert health["ok"] is True

    result = resolve_remote(
        serve_thread,
        "set gain to 50%",
        "max-msp",
        backend="b0",
    )
    assert isinstance(result, SuccessIntent)
    assert result.pattern_id == "gain_set"


def test_resolve_nl_uses_serve_url(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def _fake_remote(
        base_url: str,
        nl: str,
        profile_device_id: str,
        *,
        backend: str = "b1",
        model_id: str | None = None,
    ) -> SuccessIntent:
        calls.append(base_url)
        from llm4osc.resolver import resolve_nl_b0

        return resolve_nl_b0(nl, PROFILE)  # type: ignore[return-value]

    monkeypatch.setattr("llm4osc.serve.resolve_remote", _fake_remote)
    result = resolve_nl(
        "set gain to 50%",
        PROFILE,
        backend="b1",
        serve_url="http://127.0.0.1:8765",
    )
    assert calls == ["http://127.0.0.1:8765"]
    assert isinstance(result, SuccessIntent)
