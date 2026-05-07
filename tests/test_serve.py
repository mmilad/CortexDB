from __future__ import annotations

import pytest

from app.serve import _env_port


def test_env_port_defaults_to_5000(monkeypatch):
    monkeypatch.delenv("CORTEXDB_API_PORT", raising=False)
    assert _env_port() == 5000


def test_env_port_accepts_override(monkeypatch):
    monkeypatch.setenv("CORTEXDB_API_PORT", "5001")
    assert _env_port() == 5001


def test_env_port_rejects_invalid_value(monkeypatch):
    monkeypatch.setenv("CORTEXDB_API_PORT", "not-a-port")
    with pytest.raises(SystemExit):
        _env_port()
