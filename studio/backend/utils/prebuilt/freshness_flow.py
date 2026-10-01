# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared mechanics of the llama.cpp / whisper.cpp prebuilt freshness checks. The component modules (utils.llama_cpp_freshness / utils.whisper_cpp_freshness) keep their public names, per-module caches and version-comparison policy; everything mechanical (marker walk-up, GitHub release fetch, memo + disk cache, the freshness report skeleton) lives here, parameterized by call-time callables so the modules' monkeypatch seams keep working."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import structlog

from utils.update_status import update_checks_disabled

logger = structlog.get_logger(__name__)

# 24h TTL keeps the GitHub call off the hot path and within rate limits.
RELEASE_CACHE_TTL_SECONDS = 24 * 60 * 60
# Briefly memoize failed lookups so recurring status reads do not retry an unreachable GitHub endpoint on every request.
RELEASE_FAILURE_CACHE_TTL_SECONDS = 60


def read_install_marker(
    binary_path: Optional[str],
    *,
    marker_name: str,
    cache: dict[str, Optional[dict]],
    log_message: str,
) -> Optional[dict]:
    """Walk up from binary_path to find the install marker JSON. None = no marker (source build / custom path) or unusable JSON. "Unusable" includes JSON that parses but is not an object: a marker holding ``[]`` or ``123`` reaches every caller as something without ``.get``, and the update planner, the backend picker and crash recovery then raise AttributeError on what is only a corrupt file."""
    if not binary_path:
        return None
    cached = cache.get(binary_path)
    if cached is not None or binary_path in cache:
        return cached
    p = Path(binary_path)
    marker: Optional[dict] = None
    # Cover all managed binary layouts (binary is 1-4 dirs deep).
    for parent in p.parents[:5]:
        candidate = parent / marker_name
        if candidate.is_file():
            try:
                marker = json.loads(candidate.read_text(encoding = "utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.debug(log_message, path = str(candidate), error = str(exc))
                marker = None
            else:
                if not isinstance(marker, dict):
                    logger.debug(
                        log_message,
                        path = str(candidate),
                        error = f"marker is {type(marker).__name__}, not an object",
                    )
                    marker = None
            break
    cache[binary_path] = marker
    return marker


def cache_path_for(repo: str, cache_dir: Path) -> Path:
    safe = repo.replace("/", "__")
    # v2: caches written while releases were still fetched from GitHub are never read, or their
    # last-good tag would keep raising an update banner now that nothing can replace it.
    return cache_dir / f"{safe}.v2.json"


def load_disk_cache(repo: str, cache_dir: Path) -> Optional[tuple[float, Optional[str]]]:
    path = cache_path_for(repo, cache_dir)
    try:
        payload = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    ts = payload.get("fetched_at")
    tag = payload.get("latest_tag")
    if not isinstance(ts, (int, float)):
        return None
    return float(ts), tag if isinstance(tag, str) else None


def save_disk_cache(
    repo: str, latest_tag: Optional[str], cache_dir: Path, *, log_message: str
) -> None:
    path = cache_path_for(repo, cache_dir)
    try:
        path.parent.mkdir(parents = True, exist_ok = True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"fetched_at": time.time(), "latest_tag": latest_tag}),
            encoding = "utf-8",
        )
        tmp.replace(path)
    except OSError as exc:
        logger.debug(log_message, repo = repo, error = str(exc))


# Release lookups never leave this machine: the two fetchers below are the only code that
# reached GitHub, and they now always answer "unknown" (None), which every caller already treats
# as "no update, no banner". The caching above them is unchanged, so an injected fetcher (tests)
# still flows through it.


def fetch_latest_release_tag(
    repo: str,
    timeout: float = 5.0,
    *,
    log_message: str,
) -> Optional[str]:
    return None


def fetch_latest_release_assets(
    repo: str,
    timeout: float = 5.0,
    *,
    log_message: str,
) -> Optional[dict[str, int]]:
    return None


def latest_published_release(
    repo: str,
    *,
    force_refresh: bool,
    memo: dict[str, tuple[float, Optional[str]]],
    cache_dir: Callable[[], Path],
    fetch: Callable[[str], Optional[str]],
    save: Callable[[str, Optional[str]], None],
    failed_at: Optional[dict[str, float]] = None,
) -> Optional[str]:
    """Latest release tag with optional short-lived failure caching. Successes use the 24h memory and disk cache; supplying ``failed_at`` also caches failures for ``RELEASE_FAILURE_CACHE_TTL_SECONDS``, while omitting it keeps retry-on-every-call behavior."""
    if not repo:
        return None
    # Success timestamps persist to disk and need wall time. Failure timestamps are process-local and use monotonic time so clock changes cannot extend them.
    wall_now = time.time()
    if not force_refresh:
        last_failure = failed_at.get(repo) if failed_at is not None else None
        if (
            last_failure is not None
            and time.monotonic() - last_failure < RELEASE_FAILURE_CACHE_TTL_SECONDS
        ):
            cached = memo.get(repo)
            if cached:
                return cached[1]
            disk = load_disk_cache(repo, cache_dir())
            return disk[1] if disk else None
        cached = memo.get(repo)
        if cached and wall_now - cached[0] < RELEASE_CACHE_TTL_SECONDS:
            return cached[1]
        disk = load_disk_cache(repo, cache_dir())
        if disk and wall_now - disk[0] < RELEASE_CACHE_TTL_SECONDS:
            memo[repo] = disk
            return disk[1]
    latest = fetch(repo)
    if latest is None:
        if failed_at is not None:
            failed_at[repo] = time.monotonic()
        # Keep the last-good disk value rather than poison it with None.
        disk = load_disk_cache(repo, cache_dir())
        if disk:
            memo[repo] = disk
            return disk[1]
        return None
    if failed_at is not None:
        failed_at.pop(repo, None)
    memo[repo] = (wall_now, latest)
    save(repo, latest)
    return latest


def latest_release_assets(
    repo: str,
    *,
    force_refresh: bool,
    memo: dict[str, tuple[float, dict[str, int]]],
    fetch: Callable[[str], Optional[dict[str, int]]],
) -> Optional[dict[str, int]]:
    """Newest-release asset sizes for `repo`, memoized (24h TTL). None when offline and never fetched. In-memory only, so a restart re-fetches."""
    if not repo:
        return None
    now = time.time()
    if not force_refresh:
        cached = memo.get(repo)
        if cached and now - cached[0] < RELEASE_CACHE_TTL_SECONDS:
            return cached[1]
    assets = fetch(repo)
    if assets is None:
        cached = memo.get(repo)
        return cached[1] if cached else None
    memo[repo] = (now, assets)
    return assets


def parse_installed_at(value: object) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    s = value.replace("Z", "+00:00") if value.endswith("Z") else value
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo = timezone.utc)
    return dt


def check_freshness(
    binary_path: Optional[str],
    *,
    threshold_days: int,
    now: Optional[datetime],
    read_marker: Callable[[Optional[str]], Optional[dict]],
    latest_release: Callable[[str], Optional[str]],
    behind: Callable[[Optional[str], Optional[str]], bool],
    display_tag: Callable[[dict], Any],
    compare_tag: Callable[[dict], Any],
) -> dict:
    """Freshness report skeleton shared by both components; the component's marker-tag choice and is_behind policy come in as callables. Fails open on missing data (behind/stale stay False)."""
    out: dict = {
        "has_marker": False,
        "stale": False,
        "behind": False,
        "installed_tag": None,
        "latest_tag": None,
        "installed_at_utc": None,
        "age_days": None,
        "published_repo": None,
        "threshold_days": int(threshold_days),
    }
    marker = read_marker(binary_path)
    if not marker:
        return out
    out["has_marker"] = True
    out["installed_tag"] = display_tag(marker)
    out["installed_at_utc"] = marker.get("installed_at_utc")
    out["published_repo"] = marker.get("published_repo")

    installed_full = compare_tag(marker)
    repo = out["published_repo"]
    if not repo or not installed_full or update_checks_disabled():
        return out
    latest = latest_release(repo)
    out["latest_tag"] = latest
    out["behind"] = behind(installed_full, latest)
    if not out["behind"]:
        return out

    installed_at = parse_installed_at(out["installed_at_utc"])
    if installed_at is None:
        return out
    now = now or datetime.now(tz = timezone.utc)
    age_seconds = (now - installed_at).total_seconds()
    out["age_days"] = max(0, int(age_seconds // 86400))
    if age_seconds >= threshold_days * 86400:
        out["stale"] = True
    return out


def format_stale_warning(info: dict, *, component: str) -> str:
    """Human-readable one-liner for stale prebuilt info."""
    age = info.get("age_days")
    installed = info.get("installed_tag") or "unknown"
    latest = info.get("latest_tag") or "unknown"
    age_str = f"{age} day{'s' if age != 1 else ''}" if age is not None else "some time"
    return (
        f"{component} prebuilt is {age_str} behind: installed "
        f"{installed}, latest {latest}. Run `unsloth studio update` "
        f"to refresh."
    )


def reset_caches(
    caches: tuple[dict, ...], *, drop_disk: bool, cache_dir: Callable[[], Path]
) -> None:
    """Drop the in-memory freshness caches; with drop_disk also the on-disk 24h release cache (see the component modules for why)."""
    for cache in caches:
        cache.clear()
    if drop_disk:
        import shutil

        # cache_dir() is a freshness-only subdir re-created on the next save_disk_cache, and ignore_errors so a missing or locked dir cannot break an otherwise successful install.
        shutil.rmtree(cache_dir(), ignore_errors = True)
