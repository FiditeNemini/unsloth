# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The installation's network policy: stored in studio.db, applied through the environment.

Pins the default (everything off), that the stored value beats an inherited environment, that
saving applies at once to this process, and that only a UI session can widen it.
"""

from __future__ import annotations

from pathlib import Path
import sys
import types as _types

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

import json
import os
import socket

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.settings as settings
import storage.studio_db as studio_db
import unsloth_network_policy as network_policy
import utils.network_policy_settings as policy_settings

_ENV_VARS = (
    network_policy.ENV_VAR,
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "UNSLOTH_NETWORK_POLICY_OWNS_OFFLINE",
)


@pytest.fixture(autouse = True)
def clean_env(monkeypatch):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising = False)
    yield
    network_policy._uninstall_socket_guard()


@pytest.fixture
def store(monkeypatch):
    """An in-memory app_settings."""
    values: dict = {}
    monkeypatch.setattr(
        studio_db, "get_app_setting", lambda key, fallback = None: values.get(key, fallback)
    )
    monkeypatch.setattr(
        studio_db, "upsert_app_settings", lambda updates: values.update(updates) or values
    )
    return values


def _env_policy() -> dict:
    return json.loads(os.environ[network_policy.ENV_VAR])


# ── storage and activation ──────────────────────────────────────────────────


def test_nothing_stored_means_everything_off(store):
    assert policy_settings.stored_policy() == network_policy.Policy()
    assert store == {}


@pytest.mark.parametrize("junk", ["on", 1, ["hf_hub"], {"enabled": "true"}])
def test_junk_on_disk_reads_as_off(store, junk):
    store[policy_settings.NETWORK_POLICY_SETTING_KEY] = junk
    assert not policy_settings.stored_policy().enabled


def test_unreadable_db_reads_as_off(monkeypatch):
    def locked(*_a, **_k):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(studio_db, "get_app_setting", locked)
    assert policy_settings.stored_policy() == network_policy.Policy()


def test_activate_applies_the_stored_policy(store):
    store[policy_settings.NETWORK_POLICY_SETTING_KEY] = {"enabled": True, "services": ["wandb"]}
    policy = policy_settings.activate_stored_policy()
    assert policy.allows("wandb") and not policy.allows("hf_hub")
    assert _env_policy()["services"] == ["wandb"]
    # Hub access is still off, so the HF libraries go offline.
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_stored_policy_beats_an_inherited_environment(store, monkeypatch):
    """A launcher cannot widen what the owner configured by exporting the variable."""
    monkeypatch.setenv(network_policy.ENV_VAR, json.dumps({"enabled": True, "services": ["hf_hub"]}))
    policy = policy_settings.activate_stored_policy()
    assert not policy.enabled
    assert not network_policy.allowed("hf_hub")


def test_activate_installs_the_socket_guard(store):
    policy_settings.activate_stored_policy()
    with pytest.raises(network_policy.NetworkDisabledError):
        socket.getaddrinfo("example.com", 443)


def test_set_persists_and_applies(store):
    policy = policy_settings.set_network_policy(
        enabled = True, services = {"hf_hub": True, "wandb": False}, allow_lan = True
    )
    assert store[policy_settings.NETWORK_POLICY_SETTING_KEY] == {
        "enabled": True,
        "services": ["hf_hub"],
        "allow_lan": True,
    }
    assert policy.allows("hf_hub") and policy.allow_lan
    assert network_policy.allowed("hf_hub")
    assert "HF_HUB_OFFLINE" not in os.environ


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enabled": "yes", "services": []},
        {"enabled": True, "services": ["hf"]},
        {"enabled": True, "services": {"hf_hub": "on"}},
        {"enabled": True, "services": "hf_hub"},
        {"enabled": True, "services": [], "allow_lan": 1},
    ],
)
def test_set_rejects_bad_input_and_stores_nothing(store, kwargs):
    with pytest.raises(ValueError):
        policy_settings.set_network_policy(**kwargs)
    assert store == {}


def test_catalog_lists_every_service():
    ids = [entry["id"] for entry in policy_settings.service_catalog()]
    assert ids == list(network_policy.SERVICES)


# ── routes ──────────────────────────────────────────────────────────────────


def _client(via_api_key: bool) -> TestClient:
    app = FastAPI()
    app.include_router(settings.router)
    app.dependency_overrides[settings.get_current_subject] = lambda: "admin"
    app.dependency_overrides[settings.authenticated_via_api_key] = lambda: via_api_key
    return TestClient(app, raise_server_exceptions = False)


@pytest.fixture
def client(store):
    return _client(via_api_key = False)


def test_route_reports_off_with_the_catalog(client):
    body = client.get("/network-policy").json()
    assert body["enabled"] is False
    assert body["allow_lan"] is False
    assert set(body["services"]) == set(network_policy.SERVICES)
    assert not any(body["services"].values())
    assert {entry["id"] for entry in body["catalog"]} == set(network_policy.SERVICES)
    assert body["applies_to_running_jobs"] is False


def test_route_saves_and_applies(client, store):
    services = {service: service == "hf_hub" for service in network_policy.SERVICES}
    response = client.put(
        "/network-policy", json = {"enabled": True, "services": services, "allow_lan": False}
    )
    assert response.status_code == 200
    assert response.json()["services"]["hf_hub"] is True
    assert store[policy_settings.NETWORK_POLICY_SETTING_KEY]["services"] == ["hf_hub"]
    assert network_policy.allowed("hf_hub")
    assert client.get("/network-policy").json()["enabled"] is True


def test_route_rejects_unknown_service(client, store):
    response = client.put("/network-policy", json = {"enabled": True, "services": {"telepathy": True}})
    assert response.status_code == 400
    assert store == {}


def test_route_rejects_non_bool(client, store):
    response = client.put("/network-policy", json = {"enabled": "true", "services": {}})
    assert response.status_code == 422
    assert store == {}


def test_api_key_cannot_change_the_policy(store):
    client = _client(via_api_key = True)
    response = client.put("/network-policy", json = {"enabled": True, "services": {"hf_hub": True}})
    assert response.status_code == 403
    assert store == {}
    assert not network_policy.allowed("hf_hub")
    # Reading is fine: the UI shows the state to everyone.
    assert client.get("/network-policy").status_code == 200
