from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def probe(monkeypatch):
    path = Path(__file__).parents[1] / "scripts/check_rk3588_contract.py"
    spec = importlib.util.spec_from_file_location("executor_health_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("PERSON_SEARCH_PROBE_TOKEN_FILE", raising=False)
    monkeypatch.setattr(module.sys, "argv", [str(path)])
    return module


def test_container_probe_only_checks_liveness(probe, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(probe, "_get_json", lambda base, path: calls.append(path) or {"status": "ok"})
    assert probe.main() == 0
    assert calls == ["/livez"]
    assert "readiness and authentication are not checked" in capsys.readouterr().out


def test_readiness_gate_requires_explicit_credential(probe, monkeypatch):
    monkeypatch.setattr(probe.sys, "argv", ["probe", "--require-ready"])
    assert probe.main() == 1


def test_readiness_uses_owner_only_token_and_required_capabilities(probe, monkeypatch, tmp_path):
    token = tmp_path / "probe.jwt"
    token.write_text("synthetic-probe-token")
    token.chmod(0o600)
    monkeypatch.setenv("PERSON_SEARCH_PROBE_TOKEN_FILE", str(token))
    calls = []

    def get_json(base, path, supplied):
        calls.append((path, supplied))
        return {"status": "ok", "capabilities": list(probe._REQUIRED_CAPABILITIES)} if path == "/healthz" else {"status": "ready"}

    monkeypatch.setattr(probe, "_get_json", get_json)
    assert probe.main() == 0
    assert calls == [("/healthz", "synthetic-probe-token"), ("/readyz", "synthetic-probe-token")]
    token.chmod(0o644)
    assert probe.main() == 1


def test_failed_readiness_is_not_replaced_by_liveness(probe, monkeypatch, tmp_path):
    token = tmp_path / "probe.jwt"
    token.write_text("synthetic-probe-token")
    token.chmod(0o600)
    monkeypatch.setenv("PERSON_SEARCH_PROBE_TOKEN_FILE", str(token))
    monkeypatch.setattr(probe, "_get_json", lambda *args: {"status": "ok"})
    assert probe.main() == 1
