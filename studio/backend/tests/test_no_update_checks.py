# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio never checks for updates: no PyPI lookup for the update popup, no GitHub fetch for its
release notes. Both routes keep their response shape, so the frontend needs no special case."""

import socket
import urllib.request

import pytest

from utils import release_notes, update_status


@pytest.fixture(autouse = True)
def no_network(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("an update check touched the network")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def test_update_status_is_always_disabled(monkeypatch):
    monkeypatch.delenv(update_status.DISABLE_ENV_VAR, raising = False)
    # The dev-only fake-update switch is gone too.
    monkeypatch.setenv("UNSLOTH_STUDIO_FAKE_UPDATE", "2099.1.1")
    status = update_status.get_studio_update_status("2026.9.11")
    assert status["update_available"] is False
    assert status["latest_version"] is None
    assert status["reason"] == "disabled"
    assert status["can_show_web_notification"] is False
    assert status["current_version"] == "2026.9.11"


def test_release_notes_are_empty(monkeypatch):
    notes = release_notes.get_release_notes("2026.9.11", refresh = True)
    assert notes["version"] == "2026.9.11"
    assert notes["markdown"] is None
    assert notes["matched"] is False
    assert notes["error"] is None


@pytest.mark.parametrize("version", ["latest", "main", "../../etc", "", "x" * 80])
def test_unsupported_version_queries_are_still_rejected(version):
    assert not release_notes.is_supported_version_query(version)
    assert release_notes.get_release_notes(version)["error"] == "Unsupported version."


def test_the_lookups_are_gone_from_the_source():
    for module in (release_notes, update_status):
        source = open(module.__file__, encoding = "utf-8").read()
        for host in ("pypi.org/pypi", "api.github.com"):
            assert host not in source, (module.__name__, host)
