# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Release notes for the update popup. Studio never fetches them: there is no update check, so the
popup never has a release to describe. The route keeps its response shape (notes absent, the UI
links out) so the frontend needs no special case."""

from __future__ import annotations

import re
from typing import Any

from packaging.version import InvalidVersion, Version

from .update_status import RELEASE_NOTES_URL

_SAFE_VERSION_PATTERN = re.compile(r"^[0-9A-Za-z][0-9A-Za-z.!+-]{0,63}$")


def reset_release_notes_cache() -> None:
    """Nothing is cached any more; kept so callers and tests need not change."""


def is_supported_version_query(version: str) -> bool:
    """Whether `version` is shaped like a version the popup could be offering. The version is echoed back rather than used to select a release, so the UI can drop a stale response. `latest`, `main` or a path is rejected outright."""
    candidate = version.strip()
    if not _SAFE_VERSION_PATTERN.match(candidate):
        return False
    try:
        Version(candidate)
    except InvalidVersion:
        return False
    return True


def get_release_notes(version: str, refresh: bool = False) -> dict[str, Any]:
    """No notes, and no lookup. `version` is echoed back; `refresh` is accepted and ignored."""
    version = version.strip()
    return {
        # Echoed, so the UI can drop an answer to a version it has moved on from.
        "version": version,
        "markdown": None,
        "heading": None,
        "tag": None,
        "html_url": None,
        # False means there are no notes; the UI links out.
        "matched": False,
        "truncated": False,
        "source": None,
        "release_notes_url": RELEASE_NOTES_URL,
        "error": None if is_supported_version_query(version) else "Unsupported version.",
    }
