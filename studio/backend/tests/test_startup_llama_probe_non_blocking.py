# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The llama.cpp startup probe must run OFF the FastAPI lifespan critical path.

Regression guard for the macOS slow-startup bug: on macOS the first `llama-server --help` exec
can stall on Gatekeeper verification, which must not block `Application startup complete`. The
probe is capability-only and local: there is no release-freshness lookup at startup any more.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import main  # noqa: E402
import utils.llama_cpp_freshness as freshness  # noqa: E402
from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402

# Deadlock backstop, not pacing: a regression hangs for 30s and fails rather than forever.
_BACKSTOP_S = 30.0


class _FakeApp:
    class _State:
        pass

    def __init__(self) -> None:
        self.state = _FakeApp._State()
        self.state.llama_cpp_capabilities = None
        self.state.llama_cpp_freshness = None


def test_probe_does_not_block_startup(monkeypatch):
    """`_start_llama_cpp_probes_if_enabled` returns at once even while the capability probe is
    stalled, then populates app.state later. The freshness check is never consulted."""
    entered = threading.Event()
    release = threading.Event()

    def _slow_capabilities(_bin):
        entered.set()
        assert release.wait(_BACKSTOP_S), "the test never released the capability probe"
        return {"found": False}

    def _no_freshness(*_args, **_kwargs):
        raise AssertionError("startup consulted the release-freshness check")

    monkeypatch.setattr(
        LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: "/no/such/llama-server")
    )
    monkeypatch.setattr(
        LlamaCppBackend, "probe_server_capabilities", staticmethod(_slow_capabilities)
    )
    monkeypatch.setattr(freshness, "check_prebuilt_freshness", _no_freshness)

    app = _FakeApp()
    t0 = time.monotonic()
    main._start_llama_cpp_probes_if_enabled(app)
    elapsed = time.monotonic() - t0
    assert elapsed < 0.5, f"startup probe blocked the caller for {elapsed:.2f}s"

    # The stall has to be real for the timing above to mean anything.
    assert entered.wait(_BACKSTOP_S), "the probe thread never reached the capability probe"
    release.set()

    deadline = time.monotonic() + _BACKSTOP_S
    while app.state.llama_cpp_capabilities is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert app.state.llama_cpp_capabilities == {"found": False}
    assert app.state.llama_cpp_freshness is None
