# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The launcher refresh only ever runs an installer that is already on disk.

`unsloth studio update` re-runs the installer with --shortcuts-only. Upstream fetched
install.sh / install.ps1 from unsloth.ai when no checkout was found and piped it into bash;
in this fork that would replace the hardened install with upstream's, so nothing is fetched:
the checkout's installer runs, and with none on disk the refresh is skipped.
"""

from __future__ import annotations

import socket
import sys
import urllib.request
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _studio():
    from unsloth_cli.commands import studio as _studio_mod
    return _studio_mod


class _Result:
    returncode = 0


@pytest.fixture(autouse = True)
def no_network(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("the launcher refresh touched the network")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


@pytest.fixture
def linux(monkeypatch, tmp_path):
    """POSIX refresh with no checkout in sight unless a test adds one. The tests run from a
    clone, so _PACKAGE_ROOT would otherwise always supply an installer."""
    studio = _studio()
    monkeypatch.setattr(studio.platform, "system", lambda: "Linux")
    monkeypatch.setattr(studio, "_PACKAGE_ROOT", tmp_path / "no-checkout-here")
    monkeypatch.delenv("STUDIO_LOCAL_REPO", raising = False)
    return studio


def test_the_download_path_is_gone():
    studio = _studio()
    for name in (
        "_fetch_installer",
        "_run_fetched_installer_bash",
        "_run_fetched_installer_ps1",
        "_INSTALLER_URL_BASH",
        "_INSTALLER_URL_PWSH",
    ):
        assert not hasattr(studio, name), name


def test_a_checkout_installer_is_run(linux, monkeypatch, tmp_path):
    checkout = tmp_path / "install.sh"
    checkout.write_text("#!/bin/sh\n")
    monkeypatch.setenv("STUDIO_LOCAL_REPO", str(tmp_path))
    runs = []
    monkeypatch.setattr(linux.subprocess, "run", lambda argv, **kw: runs.append(argv) or _Result())
    linux._refresh_desktop_shortcuts()
    assert runs == [["bash", str(checkout), "--shortcuts-only"]]


def test_no_installer_on_disk_skips_the_refresh(linux, monkeypatch, capsys):
    runs = []
    monkeypatch.setattr(linux.subprocess, "run", lambda argv, **kw: runs.append(argv) or _Result())
    linux._refresh_desktop_shortcuts()
    assert runs == []
    assert "installers are never downloaded" in capsys.readouterr().out


def test_an_unlaunchable_installer_is_skipped_not_replaced_by_a_download(linux, monkeypatch, tmp_path, capsys):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "install.sh").write_text("#!/bin/sh\n")
    monkeypatch.setenv("STUDIO_LOCAL_REPO", str(checkout))
    seen = []

    def _run(argv, **kwargs):
        seen.append(list(argv))
        raise OSError(8, "Exec format error")

    monkeypatch.setattr(linux.subprocess, "run", _run)
    linux._refresh_desktop_shortcuts()
    assert len(seen) == 1 and str(checkout / "install.sh") in seen[0]
    assert "installers are never downloaded" in capsys.readouterr().out


def test_a_second_on_disk_installer_is_tried(monkeypatch, tmp_path):
    """An unlaunchable candidate leaves the next one on disk to try."""
    studio = _studio()
    local = tmp_path / "local"
    pkg = tmp_path / "pkg"
    for d in (local, pkg):
        d.mkdir()
        (d / "install.sh").write_text("#!/bin/sh\n")
    monkeypatch.setenv("STUDIO_LOCAL_REPO", str(local))
    monkeypatch.setattr(studio, "_PACKAGE_ROOT", pkg)
    monkeypatch.setattr(studio.platform, "system", lambda: "Linux")
    seen = []

    def _run(argv, **kwargs):
        seen.append(list(argv))
        if str(local / "install.sh") in argv:
            raise OSError(11, "Resource temporarily unavailable")
        return _Result()

    monkeypatch.setattr(studio.subprocess, "run", _run)
    studio._refresh_desktop_shortcuts()
    assert len(seen) == 2, seen
    assert str(local / "install.sh") in seen[0]
    assert str(pkg / "install.sh") in seen[1]
