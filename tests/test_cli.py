"""`robin --check` is the first command the README gives a stranger.

It had zero coverage, and shipped with a NameError on its own failure path —
the exact branch a first-time user hits, reached only when nothing is
configured yet. A test that runs the command is the cheapest possible guard
against that class.
"""
import json

import pytest

from robin import cli


@pytest.fixture
def tables_in(tmp_path, monkeypatch):
    (tmp_path / "llm_providers.json").write_text(json.dumps({
        "a": {"base_url": "https://a.test/v1", "api_key_env": "A_KEY"},
        "b": {"base_url": "https://b.test/v1", "api_key_env": "B_KEY"}}))
    (tmp_path / "model_routes.json").write_text(json.dumps({
        "flash": {"rotate": ["a/m", "b/m"], "fallback": []},
        "pro": ["a/p"]}))
    secrets = tmp_path / "s"
    secrets.mkdir()
    monkeypatch.setenv("ROBIN_SECRETS_DIR", str(secrets))
    return tmp_path, secrets


def _check(tmp_path):
    return cli.main(["--check",
                     "--providers", str(tmp_path / "llm_providers.json"),
                     "--routes", str(tmp_path / "model_routes.json")])


def test_check_with_no_keys_reports_every_model_and_says_what_to_write(
        tables_in, capsys):
    tmp_path, secrets = tables_in
    assert _check(tmp_path) == 1
    err = capsys.readouterr().err
    assert "flash" in err and "pro" in err
    assert str(secrets) in err          # where to put the key
    assert "KEY_NAME" in err


def test_check_is_green_once_every_model_has_one_usable_endpoint(tables_in,
                                                                 capsys):
    tmp_path, secrets = tables_in
    (secrets / "A_KEY").write_text("k")     # serves both routes
    assert _check(tmp_path) == 0
    out = capsys.readouterr().out
    assert "key A_KEY present" in out
    assert "no key file B_KEY" in out       # still reported, just not fatal


def test_a_model_with_no_usable_endpoint_is_not_hidden_by_a_healthy_one(
        tables_in, capsys):
    """`usable_any` used to be global: a config where one model was entirely
    dead exited 0, so a systemd ExecStartPre check went green on it."""
    tmp_path, secrets = tables_in
    (secrets / "B_KEY").write_text("k")     # serves flash only; pro is dead
    assert _check(tmp_path) == 1
    assert "pro" in capsys.readouterr().err


def test_init_writes_starter_tables_and_never_clobbers(tmp_path, monkeypatch,
                                                       capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["--init"]) == 0
    assert (tmp_path / "llm_providers.json").is_file()
    (tmp_path / "model_routes.json").write_text('{"mine": ["a/b"]}')
    assert cli.main(["--init"]) == 0
    assert json.loads((tmp_path / "model_routes.json").read_text()) == {
        "mine": ["a/b"]}


def test_a_broken_table_is_a_config_error_not_a_traceback(tmp_path, capsys):
    (tmp_path / "model_routes.json").write_text("{ not json")
    (tmp_path / "llm_providers.json").write_text("{}")
    assert _check(tmp_path) == 2
    assert "config error" in capsys.readouterr().err
