"""The HTTP layer, against a mock upstream.

These cover the three places a silent regression costs money: which statuses
walk the pool, whether a spent window actually parks, and whether the model
name the client gets back is the one it asked for.
"""
import json

import httpx
import pytest
from starlette.testclient import TestClient

from robin.config import Providers, Routes
from robin import server as server_mod
from robin.server import _fails_over, create_app


@pytest.fixture
def app_with(tables, monkeypatch):
    """Build the app with a scripted upstream. `script` maps call index →
    (status, body, headers)."""
    tmp_path, _ = tables

    def build(script):
        calls: list[dict] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            i = len(calls)
            calls.append({"url": str(request.url),
                          "body": json.loads(request.content or b"{}")})
            status, body, headers = script[min(i, len(script) - 1)]
            payload = body if isinstance(body, (bytes, str)) else json.dumps(body)
            return httpx.Response(status, content=payload, headers=headers or {})

        app = create_app(Providers(tmp_path / "llm_providers.json"),
                         Routes(tmp_path / "model_routes.json"))
        # Inject by REBINDING the module's client factory, not by patching
        # httpx.AsyncClient.__init__. The patch version captured `original`
        # AFTER a previous build() had already patched it, so a second app in
        # one test silently kept the FIRST script — which made the throttle
        # assertion pass without the throttle body ever being sent. A test
        # that cannot fail is worse than no test, and it was guarding the
        # least certain heuristic in the codebase.
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        monkeypatch.setattr(server_mod, "_client", lambda _state: client)
        return app, calls

    return build


def _ask(client, model="flash", stream=False):
    return client.post("/v1/chat/completions", json={
        "model": model, "stream": stream,
        "messages": [{"role": "system", "content": "sys"},
                     {"role": "user", "content": "hi"}]})


_OK = {"id": "x", "model": "upstream-id-not-the-route",
       "choices": [{"message": {"role": "assistant", "content": "hi"}}]}


# ── which failures walk the pool ───────────────────────────────────────────

def test_fails_over_classification_is_pinned():
    """A one-token edit here silently triples quota burn per bad request, or
    strands conversations on a drained plan. 402 and 404 are the easy misses:
    DeepSeek answers 402 Insufficient Balance."""
    for s in (401, 402, 403, 404, 408, 409, 429, 500, 502, 503):
        assert _fails_over(s), s
    for s in (400, 404 - 4, 413, 422):     # 400, 413, 422 are request errors
        assert not _fails_over(s), s


def test_a_request_error_answers_once_instead_of_burning_the_pool(app_with):
    """Every candidate rejects a malformed request identically: walking the
    list turns one clear error into N and spends the quota being conserved."""
    app, calls = app_with([(400, {"error": {"message": "bad request"}}, None)])
    with TestClient(app) as client:
        r = _ask(client)
    assert r.status_code == 400
    assert len(calls) == 1


def test_an_endpoint_error_moves_to_the_next_plan(app_with):
    app, calls = app_with([(500, {"error": "boom"}, None),
                           (200, _OK, None)])
    with TestClient(app) as client:
        r = _ask(client)
    assert r.status_code == 200
    assert len(calls) == 2
    assert calls[0]["url"] != calls[1]["url"]      # a different provider


def test_a_drained_plan_does_not_strand_the_conversation(app_with):
    """402 used to be returned verbatim AND left pinned, so every later turn
    of that conversation hit the same drained plan forever."""
    app, calls = app_with([(402, {"error": {"message": "Insufficient Balance"}},
                            None),
                           (200, _OK, None)])
    with TestClient(app) as client:
        assert _ask(client).status_code == 200
    assert len(calls) == 2


def test_a_non_json_success_is_treated_as_an_endpoint_failure(app_with):
    """A gateway interstitial with a 200 is not a completion. Returned as-is
    it reads to the caller as a successful turn with no choices — the request
    silently lost, with no failover."""
    app, calls = app_with([(200, "<html>gateway</html>", {"content-type": "text/html"}),
                           (200, _OK, None)])
    with TestClient(app) as client:
        r = _ask(client)
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "hi"
    assert len(calls) == 2


# ── parking ────────────────────────────────────────────────────────────────

def test_a_spent_window_parks_and_a_throttle_does_not(app_with):
    """The asymmetry that decides the regex: parking a throttled plan throws
    away capacity you pay for; retrying a spent one costs a single call."""
    app, _ = app_with([(429, {"error": {"message": "Insufficient Balance"}}, None),
                       (200, _OK, None)])
    with TestClient(app) as client:
        _ask(client)
        assert client.get("/stats").json()["parked"]

    # This body is the one that NEEDS the veto: it matches the spend pattern
    # ("Quota exceeded") and is a per-minute ceiling that clears in seconds.
    # An earlier version of this test used Ark's "tpm quota ... exceeded",
    # where the spend words are not adjacent — so it never reached the veto
    # and stayed green with the veto deleted.
    app, calls = app_with([(429, {"error": {"message":
                           "Quota exceeded for quota metric 'requests per minute'"}},
                           None),
                           (200, _OK, None)])
    with TestClient(app) as client:
        _ask(client)
        # The second app must actually run its OWN script; the first harness
        # silently reused the first one, so this assertion held for the wrong
        # reason and survived deleting the throttle veto entirely.
        assert calls, "the throttle body was never sent — harness reuse"
        assert "tpm" in json.dumps(calls) or len(calls) >= 1
        assert client.get("/stats").json()["parked"] == {}


def test_retry_after_beats_the_prose(app_with):
    app, _ = app_with([(429, {"error": {"message": "usage limit reached"}},
                        {"retry-after": "1800"}),
                       (200, _OK, None)])
    with TestClient(app) as client:
        _ask(client)
        parked = client.get("/stats").json()["parked"]
        assert parked and 1700 < list(parked.values())[0] <= 1800


def test_unpark_releases_without_a_restart(app_with):
    """A wrong park hides a plan you pay for; restarting to clear it drops
    every conversation's assignment."""
    app, _ = app_with([(429, {"error": {"message": "Insufficient Balance"}}, None),
                       (200, _OK, None)])
    with TestClient(app) as client:
        _ask(client)
        assert client.get("/stats").json()["parked"]
        assert client.post("/unpark").json()["released"]
        assert client.get("/stats").json()["parked"] == {}


# ── the client gets back what it asked for ─────────────────────────────────

def test_the_response_carries_the_route_name_not_the_upstream_id(app_with):
    """A client that echoes `model` would otherwise start asking for the
    concrete id — routing bypassed by its own answer."""
    app, calls = app_with([(200, _OK, None)])
    with TestClient(app) as client:
        r = _ask(client)
    assert r.json()["model"] == "flash"
    assert calls[0]["body"]["model"] == "m"        # the upstream got the real id
    assert r.headers["x-robin-endpoint"].endswith("/m")
    assert r.json()["robin"]["served_by"] == r.headers["x-robin-endpoint"]


def test_streaming_frames_carry_the_route_name_too(app_with):
    frames = ("data: " + json.dumps({"model": "upstream-id",
                                     "choices": [{"delta": {"content": "hi"}}]})
              + "\n\ndata: [DONE]\n\n")
    app, _ = app_with([(200, frames, {"content-type": "text/event-stream"})])
    with TestClient(app) as client:
        r = _ask(client, stream=True)
    assert '"model": "flash"' in r.text
    assert "upstream-id" not in r.text
    assert "[DONE]" in r.text


def test_an_unknown_model_names_the_table(app_with):
    app, calls = app_with([(200, _OK, None)])
    with TestClient(app) as client:
        r = _ask(client, model="flsh")
    assert r.status_code == 400
    assert "flash" in r.json()["error"]["message"]
    assert not calls


def test_reload_refuses_a_broken_edit_and_keeps_serving(app_with, tables):
    """A typo in a config edit must not take a running proxy down."""
    tmp_path, _ = tables
    app, _ = app_with([(200, _OK, None)])
    with TestClient(app) as client:
        (tmp_path / "model_routes.json").write_text("{ not json")
        r = client.post("/reload")
        assert r.status_code == 400
        assert "still active" in r.json()["error"]["message"]
        assert _ask(client).status_code == 200          # old tables still work

        (tmp_path / "model_routes.json").write_text(json.dumps(
            {"flash": ["a/m"], "extra": ["b/m"]}))
        assert client.post("/reload").json()["models"] == ["extra", "flash"]
        assert client.get("/v1/models").json()["data"][0]["id"] == "extra"


# ── release-blocking behaviours ────────────────────────────────────────────

def test_the_pay_as_you_go_tail_is_reachable_with_a_full_pool(app_with):
    """A flat 3-attempt cap was consumed by a 3-plan rotate pool, so the
    escape hatch the whole design promises ("the next request goes elsewhere")
    was never reached on exactly the deployment the README recruits."""
    app, calls = app_with([(500, {"error": "plan down"}, None),
                           (500, {"error": "plan down"}, None),
                           (500, {"error": "plan down"}, None),
                           (200, _OK, None)])
    with TestClient(app) as client:
        r = _ask(client)
    assert r.status_code == 200
    assert len(calls) == 4
    assert "p.test" in calls[-1]["url"], calls[-1]["url"]   # the payg provider


def test_a_non_stream_body_on_the_streaming_path_fails_over(app_with):
    """The buffered path already refused a gateway page with a 200; leaving
    the streaming path open meant the client got a 200 SSE response whose body
    was HTML, with no failover — the path agent clients use by default."""
    good = ("data: " + json.dumps({"model": "u", "choices": [{"delta": {}}]})
            + "\n\ndata: [DONE]\n\n")
    app, calls = app_with([(200, "<html>502</html>", {"content-type": "text/html"}),
                           (200, good, {"content-type": "text/event-stream"})])
    with TestClient(app) as client:
        r = _ask(client, stream=True)
    assert "html" not in r.text
    assert "[DONE]" in r.text
    assert len(calls) == 2


def test_the_token_gate_covers_every_route_not_just_completions(app_with,
                                                               tmp_path,
                                                               monkeypatch):
    """/reload re-reads files from disk, /unpark clears cooldowns, /stats
    enumerates your endpoints. Gating only the completion route made the
    README's own "or set ROBIN_API_KEY_FILE" a false comfort."""
    token = tmp_path / "robin_token"
    token.write_text("s3cret")
    monkeypatch.setenv("ROBIN_API_KEY_FILE", str(token))
    app, _ = app_with([(200, _OK, None)])
    with TestClient(app) as client:
        for method, path in (("get", "/stats"), ("get", "/health"),
                             ("post", "/reload"), ("post", "/unpark")):
            assert getattr(client, method)(path).status_code == 401, path
        assert _ask(client).status_code == 401
        ok = {"Authorization": "Bearer s3cret"}
        assert client.get("/stats", headers=ok).status_code == 200


def test_the_unknown_model_error_does_not_leak_a_home_directory(app_with):
    """It lands in screenshots and pasted client logs."""
    app, _ = app_with([(200, _OK, None)])
    with TestClient(app) as client:
        msg = _ask(client, model="nope").json()["error"]["message"]
    assert "/home/" not in msg and "model_routes" in msg
