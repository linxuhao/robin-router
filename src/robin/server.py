"""The OpenAI-compatible face.

Everything interesting already happened in `routing`; this layer forwards, and
gets three things right that a naive proxy gets wrong:

**Which failures move to the next plan.** A dead key, a spent window, a 5xx or
a dropped connection are ENDPOINT failures: another plan can serve. A malformed
request or an over-long context is not — every candidate rejects it identically,
so walking the list turns one clear error into N and burns the quota the whole
exercise is meant to conserve.

**Streaming can only be retried before the first byte.** Once a token has
reached the client the response is committed. So the upstream is opened and its
status inspected BEFORE the body starts flowing: failures during the handshake
still fail over, failures mid-stream are surfaced honestly rather than papered
over with a second answer stitched onto the first.

**The client asked for a MODEL, not an endpoint.** `model` is rewritten to the
concrete upstream id on the way out, and restored to the name the client used
on the way back — otherwise a client that echoes the response's model field
starts asking for `deepseek-v4-flash` directly and the routing layer is bypassed
by its own answer.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import httpx
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import Providers, Routes, read_key, split_endpoint
from .quota import Cooldowns, is_spent
from .routing import NoEndpoint, Router, conversation_id

log = logging.getLogger("robin")

# Read timeout: agent turns with long reasoning legitimately take minutes, and
# a proxy that gives up sooner than the client would is a proxy that invents
# failures. Connect stays short — an unreachable endpoint should fail over fast.
_TIMEOUT = httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0)

# How many endpoints one request may try before giving up. Not the whole list:
# a client waiting on a doomed request wants an answer, and every attempt costs
# it latency.
_MAX_ATTEMPTS = 3


class _Upstream:
    """Shared connection pool. Per-request clients would discard TLS sessions."""

    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    async def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=_TIMEOUT,
                                             follow_redirects=False)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _fails_over(status: int) -> bool:
    """Is this an ENDPOINT failure (another plan can serve) or a REQUEST one?"""
    if status in (401, 403, 408, 409, 429):
        return True
    return status >= 500


def _client_authorized(request: Request) -> bool:
    """Optional shared secret between a client and Robin itself.

    Unset means no check: on loopback that is the sane default, and demanding
    a token to talk to your own proxy is friction for no gain. Set it the
    moment Robin listens on anything else — it holds every key you own.
    """
    path = os.getenv("ROBIN_API_KEY_FILE")
    if not path:
        return True
    try:
        want = os.path.expanduser(path)
        with open(want, encoding="utf-8") as f:
            expected = f.read().strip()
    except OSError:
        log.error("ROBIN_API_KEY_FILE is set but unreadable (%s) — refusing "
                  "every request rather than silently serving unauthenticated",
                  path)
        return False
    if not expected:
        return True
    header = request.headers.get("authorization", "")
    token = header[7:].strip() if header.lower().startswith("bearer ") else ""
    return bool(token) and token == expected


def create_app(providers: Providers | None = None,
               routes: Routes | None = None) -> FastAPI:
    providers = providers or Providers()
    routes = routes or Routes()
    unknown = routes.validate_against(providers)
    for endpoint in unknown:
        # Loud, but not fatal: a typo in a FALLBACK slot should not stop a
        # working deployment from starting — and it must never be silent,
        # because the failure it causes points nowhere near the typo.
        log.warning("route candidate %s names a provider that is not in %s "
                    "— it will be skipped", endpoint, providers.path)

    router = Router(providers, routes, Cooldowns())
    upstream = _Upstream()
    app = FastAPI(title="Robin", version="0.1.0")
    api = APIRouter()

    @app.on_event("shutdown")
    async def _shutdown():
        await upstream.aclose()

    @api.get("/health")
    async def health():
        return {"status": "ok", "models": routes.names(),
                "providers": providers.names()}

    @api.get("/stats")
    async def stats():
        """What Robin is doing right now: cursors, live conversations, parks.

        The parked list is the one an operator actually wants — "why is
        everything on the expensive endpoint" has exactly one honest answer
        and it is here.
        """
        s = router.stats()
        now = time.time()
        s["parked"] = {k: round(v - now) for k, v in s["parked"].items()}
        return s

    @api.get("/v1/models")
    async def list_models():
        """The route names, shaped like OpenAI's list so clients can populate
        a picker. Concrete endpoints are deliberately NOT exposed: which
        reseller serves a model is not the client's business, and a client
        that learns the endpoint name will start asking for it directly."""
        return {"object": "list", "data": [
            {"id": name, "object": "model", "owned_by": "robin"}
            for name in routes.names()]}

    @api.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        if not _client_authorized(request):
            return JSONResponse(status_code=401, content={"error": {
                "message": "Robin requires a bearer token (ROBIN_API_KEY_FILE)",
                "type": "invalid_request_error"}})
        try:
            body: dict[str, Any] = json.loads(await request.body())
        except ValueError:
            return JSONResponse(status_code=400, content={"error": {
                "message": "request body is not JSON",
                "type": "invalid_request_error"}})
        if not isinstance(body, dict):
            return JSONResponse(status_code=400, content={"error": {
                "message": "request body must be a JSON object",
                "type": "invalid_request_error"}})

        wanted = str(body.get("model") or "")
        convo = conversation_id(body)
        streaming = bool(body.get("stream"))

        tried: set[str] = set()
        last: tuple[int, str] | None = None

        for _ in range(_MAX_ATTEMPTS):
            try:
                endpoint, skipped = router.pick(wanted, convo,
                                                exclude=frozenset(tried))
            except KeyError:
                return JSONResponse(status_code=400, content={"error": {
                    "message": (f"unknown model '{wanted}'. Robin serves the "
                                f"routes in {routes.path}: "
                                f"{', '.join(routes.names())}"),
                    "type": "invalid_request_error"}})
            except NoEndpoint as e:
                return JSONResponse(status_code=503, content={"error": {
                    "message": str(e), "type": "no_usable_endpoint"}})
            tried.add(endpoint)
            for line in skipped:
                log.info("skip %s (%s)", line, wanted)

            provider, upstream_model = split_endpoint(endpoint)
            payload = dict(body)
            payload["model"] = upstream_model
            headers = {"content-type": "application/json"}
            key = providers.key(provider)
            if key:
                headers["authorization"] = f"Bearer {key}"

            url = providers.base_url(provider) + "/chat/completions"
            client = await upstream.client()

            try:
                req = client.build_request("POST", url, json=payload,
                                           headers=headers)
                resp = await client.send(req, stream=True)
            except httpx.HTTPError as e:
                # Never reached the provider: unambiguously an endpoint
                # failure, and the next plan may well be up.
                log.warning("transport failure on %s: %s", endpoint, e)
                last = (502, f"{endpoint}: {type(e).__name__}: {e}")
                continue

            if resp.status_code >= 400:
                text = (await resp.aread()).decode("utf-8", "replace")
                await resp.aclose()
                if resp.status_code == 429 and is_spent(text):
                    until = router.cooldowns.park(endpoint, text)
                    log.warning("parked %s until %s (window spent)", endpoint,
                                time.strftime("%H:%M:%S", time.localtime(until)))
                if not _fails_over(resp.status_code):
                    # Every candidate rejects this identically. Answer with the
                    # upstream's own words rather than walking the list.
                    return JSONResponse(status_code=resp.status_code,
                                        content=_as_error(text, endpoint))
                last = (resp.status_code, f"{endpoint}: {text[:400]}")
                router.forget(convo)     # this endpoint is not this convo's home
                continue

            if streaming:
                return StreamingResponse(
                    _stream(resp, wanted, endpoint),
                    status_code=resp.status_code,
                    media_type="text/event-stream",
                    headers={"x-robin-endpoint": endpoint,
                             "cache-control": "no-store"})

            raw = await resp.aread()
            await resp.aclose()
            return JSONResponse(
                status_code=resp.status_code,
                content=_restore_model(raw, wanted, endpoint),
                headers={"x-robin-endpoint": endpoint})

        status, detail = last or (503, "no attempt succeeded")
        return JSONResponse(status_code=status, content={"error": {
            "message": f"every endpoint tried failed. Last: {detail}",
            "type": "upstream_error"}})

    app.include_router(api)
    return app


def _as_error(text: str, endpoint: str) -> dict:
    """Pass the upstream's error through, tagged with who produced it."""
    try:
        parsed = json.loads(text)
    except ValueError:
        parsed = {"error": {"message": text[:1000], "type": "upstream_error"}}
    if isinstance(parsed, dict):
        parsed.setdefault("robin", {})["served_by"] = endpoint
    return parsed


def _restore_model(raw: bytes, wanted: str, endpoint: str) -> Any:
    """Give the client back the model name it asked for, plus attribution.

    Echoing the upstream id would teach a client that remembers it to ask for
    the concrete model next time — routing bypassed by its own answer.
    """
    try:
        data = json.loads(raw)
    except ValueError:
        return {"error": {"message": "upstream returned non-JSON",
                          "type": "upstream_error"},
                "robin": {"served_by": endpoint}}
    if isinstance(data, dict):
        data["model"] = wanted
        data["robin"] = {"served_by": endpoint}
    return data


async def _stream(resp: httpx.Response, wanted: str, endpoint: str):
    """Relay SSE, rewriting only the `model` field of each data frame.

    Byte-for-byte otherwise: an OpenAI stream carries `[DONE]`, comment lines
    and provider-specific extra fields, and a proxy that re-serialises what it
    does not understand is a proxy that quietly drops it.
    """
    try:
        async for line in resp.aiter_lines():
            if not line:
                yield "\n"
                continue
            if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                try:
                    frame = json.loads(line[6:])
                    if isinstance(frame, dict) and "model" in frame:
                        frame["model"] = wanted
                        yield f"data: {json.dumps(frame, ensure_ascii=False)}\n"
                        continue
                except ValueError:
                    pass    # not JSON: relay untouched
            yield line + "\n"
    finally:
        await resp.aclose()
