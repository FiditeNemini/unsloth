# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A wildcard bind used to ask ifconfig.me for the public IP, the GCE metadata server for its
external address, and check-host.net whether the port was reachable from outside. None of that
happens any more: the address comes from the local routing table and reachability is only
classified locally.
"""

import urllib.request

import pytest

import run


@pytest.fixture(autouse = True)
def no_urlopen(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("run.py opened a URL")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)


def test_external_ip_comes_from_the_routing_table():
    ip = run._resolve_external_ip()
    assert isinstance(ip, str) and ip


@pytest.mark.parametrize("host", ["1.1.1.1", "8.8.8.8", "example.com", "2606:4700::1111"])
def test_a_public_address_stays_unknown(host):
    run._verify_global_reachability(host, 8888)
    assert run._public_reachable is None


@pytest.mark.parametrize("host", ["192.168.1.20", "10.0.0.5", "127.0.0.1", "fe80::1"])
def test_a_private_address_is_classified_locally(host, capsys):
    run._verify_global_reachability(host, 8888)
    assert run._public_reachable is False
    assert "private/LAN address" in capsys.readouterr().out


def test_the_lookups_are_gone_from_the_source():
    source = open(run.__file__, encoding = "utf-8").read()
    for host in ("ifconfig.me", "check-host.net", "metadata.google.internal"):
        assert host not in source, host
