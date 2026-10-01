# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Web update status helpers for browser-served Unsloth Studio.

There is no update check: Studio never asks PyPI (or anywhere) for a newer version. Every
status says so (reason "disabled"); only the local install source is reported.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any

PACKAGE_NAME = "unsloth"
RELEASE_NOTES_URL = "https://unsloth.ai/docs/new/changelog"
# Read by the llama.cpp / whisper.cpp updaters, which also refuse to run while it is set.
DISABLE_ENV_VAR = "UNSLOTH_DISABLE_UPDATE_CHECK"

LOCAL_INSTALL_SOURCES = {"editable", "local_path", "vcs", "local_repo"}


def reset_update_status_cache() -> None:
    """Nothing is cached any more; kept so callers and tests need not change."""


def detect_install_source() -> str:
    """Return a coarse install source without exposing local paths.

    Conservative: PEP 610 local/vcs metadata wins. Legacy source
    installs count as local only when package files resolve outside
    site-packages/dist-packages and under a Git checkout.
    """
    try:
        dist = distribution(PACKAGE_NAME)
    except PackageNotFoundError:
        return "local_repo" if _path_has_git_parent(_repo_root_from_this_file()) else "unknown"

    try:
        direct_url = dist.read_text("direct_url.json")
    except Exception:
        return "unknown"
    if direct_url:
        return _source_from_direct_url(direct_url)

    for package_path in _distribution_package_paths(dist):
        if not _path_is_under_python_package_dir(package_path) and _path_has_git_parent(
            package_path
        ):
            return "local_repo"

    return "pypi"


def update_checks_disabled() -> bool:
    """The operator's opt-out for the llama.cpp / whisper.cpp updaters. Their release lookups never
    leave the machine either way (utils.prebuilt.freshness_flow), so they never find an update."""
    return os.environ.get(DISABLE_ENV_VAR) == "1"


def get_studio_install_source_status(current_version: str) -> dict[str, Any]:
    """Return install-source metadata without remote update checks."""
    install_source = detect_install_source()
    reason = None
    if install_source in LOCAL_INSTALL_SOURCES:
        reason = "local_source"
    elif install_source == "unknown":
        reason = "unknown_source"

    return _status_response(
        current_version = current_version,
        latest_version = None,
        install_source = install_source,
        reason = reason,
    )


def get_studio_update_status(current_version: str) -> dict[str, Any]:
    """Public, read-only update status for the web UI: never an update, never a lookup."""
    return _status_response(
        current_version = current_version,
        latest_version = None,
        install_source = detect_install_source(),
        reason = "disabled",
    )


def _status_response(
    *,
    current_version: str,
    latest_version: str | None,
    install_source: str,
    reason: str | None = None,
    error: str | None = None,
    update_available: bool = False,
    can_show_web_notification: bool = False,
    checked_at: str | None = None,
) -> dict[str, Any]:
    return {
        "current_version": current_version,
        "latest_version": latest_version,
        "update_available": update_available,
        "install_source": install_source,
        "can_show_web_notification": can_show_web_notification,
        "release_notes_url": RELEASE_NOTES_URL,
        "checked_at": checked_at or _utc_now_iso(),
        "reason": reason,
        "error": error,
    }


def _source_from_direct_url(direct_url: str) -> str:
    try:
        payload = json.loads(direct_url)
    except json.JSONDecodeError:
        return "unknown"

    if not isinstance(payload, dict):
        return "unknown"

    dir_info = payload.get("dir_info")
    if isinstance(dir_info, dict) and dir_info.get("editable") is True:
        return "editable"

    if isinstance(payload.get("vcs_info"), dict):
        return "vcs"

    url = payload.get("url")
    if isinstance(url, str) and url.startswith("file:"):
        return "local_path"

    return "unknown"


def _distribution_package_paths(dist: Any) -> list[Path]:
    paths: list[Path] = []
    files = getattr(dist, "files", None) or []
    for file in files:
        text = str(file)
        if not text.startswith(("unsloth/", "unsloth_cli/", "studio/")):
            continue
        try:
            paths.append(Path(dist.locate_file(file)).resolve())
        except OSError:
            continue
    return paths


def _path_is_under_python_package_dir(path: Path) -> bool:
    return any(part in {"site-packages", "dist-packages"} for part in path.parts)


def _path_has_git_parent(path: Path) -> bool:
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return True
    return False


def _repo_root_from_this_file() -> Path:
    # update_status.py -> utils -> backend -> studio -> repo root
    try:
        return Path(__file__).resolve().parents[3]
    except IndexError:
        return Path(__file__).resolve().parent


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond = 0).isoformat().replace("+00:00", "Z")
