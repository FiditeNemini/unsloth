# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The installation's network policy, persisted in the owner's studio.db.

The policy itself lives in unsloth_network_policy (shipped with unsloth-zoo): one environment
variable that every process started from Studio inherits, plus a socket guard. This module
loads the stored policy into that variable at startup and on every change, so the parent and
any worker started afterwards enforce it. Workers already running keep the policy they started
with; the API says so.

Default, and the answer whenever the stored value is missing or unreadable: everything off.
"""

from __future__ import annotations

from typing import Any, Mapping

try:
    import unsloth_network_policy as network_policy
except ImportError as exc:  # pragma: no cover - an install without the fork's unsloth-zoo
    raise SystemExit(
        "Unsloth Studio: unsloth_network_policy is missing, so the network policy cannot be "
        "enforced and Studio will not start. Reinstall unsloth-zoo from this repository's "
        "external/unsloth-zoo (studio/setup.sh with STUDIO_LOCAL_INSTALL=1)."
    ) from exc

NETWORK_POLICY_SETTING_KEY = "network_policy"


def _read_stored() -> Any:
    from storage.studio_db import get_app_setting
    from utils.account_context import OWNER, is_owner_context, run_as

    if is_owner_context():
        return get_app_setting(NETWORK_POLICY_SETTING_KEY, None)
    return run_as(OWNER, get_app_setting, NETWORK_POLICY_SETTING_KEY, None)


def _write_stored(value: dict) -> None:
    from storage.studio_db import upsert_app_settings
    from utils.account_context import OWNER, is_owner_context, run_as

    if is_owner_context():
        upsert_app_settings({NETWORK_POLICY_SETTING_KEY: value})
    else:
        run_as(OWNER, upsert_app_settings, {NETWORK_POLICY_SETTING_KEY: value})


def stored_policy() -> network_policy.Policy:
    """The persisted policy; everything off when absent or unreadable."""
    try:
        stored = _read_stored()
    except Exception:
        stored = None
    if not isinstance(stored, Mapping):
        return network_policy.Policy()
    return network_policy._parse(network_policy.encode(stored))


def activate_stored_policy() -> network_policy.Policy:
    """Startup: make the stored policy this process's (and its children's), then enforce it.

    The stored value always wins over an inherited UNSLOTH_NETWORK_POLICY, so a launcher
    environment cannot quietly widen what the owner configured.
    """
    network_policy.set_policy(stored_policy())
    return network_policy.activate()


def get_network_policy() -> network_policy.Policy:
    return network_policy.current()


def set_network_policy(
    *, enabled: Any, services: Any, allow_lan: Any = False
) -> network_policy.Policy:
    """Validate, persist and apply. Raises ValueError on bad input."""
    if not isinstance(enabled, bool) or not isinstance(allow_lan, bool):
        raise ValueError("enabled and allow_lan must be true or false.")
    if isinstance(services, Mapping):
        if not all(isinstance(v, bool) for v in services.values()):
            raise ValueError("Each service must be true or false.")
        names = [name for name, on in services.items() if on]
        known = list(services)
    elif isinstance(services, (list, tuple)):
        names = known = list(services)
    else:
        raise ValueError("services must be a list of service ids or a map of id to true/false.")
    for name in known:
        if not isinstance(name, str) or name not in network_policy.SERVICES:
            raise ValueError(f"Unknown network service: {name!r}.")

    policy = network_policy.Policy(enabled, frozenset(names), allow_lan)
    _write_stored(policy.to_dict())
    return network_policy.set_policy(policy)


def service_catalog() -> list[dict[str, str]]:
    return [
        {"id": service_id, "label": label, "description": description}
        for service_id, (label, description) in network_policy.SERVICES.items()
    ]
