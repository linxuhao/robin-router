import json

import pytest

from robin.config import Providers, Routes
from robin.quota import Cooldowns
from robin.routing import Router


@pytest.fixture
def tables(tmp_path, monkeypatch):
    """Three plans (a, b, c) plus a pay-as-you-go tail, all keyed."""
    (tmp_path / "llm_providers.json").write_text(json.dumps({
        "a": {"base_url": "https://a.test/v1", "api_key_env": "A_KEY"},
        "b": {"base_url": "https://b.test/v1", "api_key_env": "B_KEY"},
        "c": {"base_url": "https://c.test/v1", "api_key_env": "C_KEY"},
        "payg": {"base_url": "https://p.test/v1", "api_key_env": "P_KEY"},
    }))
    (tmp_path / "model_routes.json").write_text(json.dumps({
        "flash": {"rotate": ["a/m", "b/m", "c/m"], "fallback": ["payg/m"]},
        "plain": ["a/m", "payg/m"],
    }))
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    for k in ("A_KEY", "B_KEY", "C_KEY", "P_KEY"):
        (secrets / k).write_text("sk-test")
    monkeypatch.setenv("ROBIN_SECRETS_DIR", str(secrets))
    return tmp_path, secrets


@pytest.fixture
def router(tables):
    tmp_path, _ = tables
    return Router(Providers(tmp_path / "llm_providers.json"),
                  Routes(tmp_path / "model_routes.json"),
                  Cooldowns())
