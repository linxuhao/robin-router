"""Which endpoint serves this request.

The whole product is one decision: an agent client replays its entire
transcript on every turn, so the SAME conversation must keep hitting the same
endpoint (provider prefix caches are per-provider; one real workload measures
26:1 prefill:decode at an 89.4% hit rate, so rotating per call converts cached
input into full-price input and costs more than a second plan saves), while a
NEW conversation should start on the next plan (or the windows you pay for
expire unused).

A proxy cannot see "a new session began" — but it does not need to be told.
The conversation IS the prefix: hash the system prompt plus the first user
message and you have a stable id for as long as that conversation lives.

The one place the heuristic misfires is compaction, which rewrites the prefix
and reads as a new conversation. That is exactly the moment the provider's
prefix cache was invalidated anyway, so rotating there is free. The weak point
of the heuristic lands precisely where it costs nothing.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time

from .config import Providers, Routes
from .quota import Cooldowns

# How long a conversation's endpoint assignment is remembered after its last
# turn. Long enough that a user's thinking pause does not re-roll the dice;
# short enough that the table cannot grow without bound.
_STICKY_TTL_S = 3600.0
_STICKY_MAX = 4096


def conversation_id(body: dict) -> str:
    """A stable id for the conversation this request belongs to.

    System prompt + first user message: the part an agent client repeats
    verbatim every turn, and the part a provider's prefix cache keys on. NOT
    the whole message list — that changes every turn, which is the bug this
    function exists to avoid.
    """
    messages = body.get("messages")
    parts: list[str] = []
    if isinstance(messages, list):
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            if role == "system":
                parts.append("s:" + _text_of(m.get("content")))
            elif role == "user":
                parts.append("u:" + _text_of(m.get("content")))
                break       # first user turn is enough; later ones churn
    if not parts:
        # No messages (a /completions-style call, or an empty body): fall back
        # to the model name so such calls still spread rather than all landing
        # on one endpoint.
        parts = ["m:" + str(body.get("model") or "")]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:32]


def _text_of(content) -> str:
    """Message content as text — string, or the multipart list form."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text")
    return json.dumps(content, sort_keys=True, default=str) if content else ""


class Router:
    """Picks an endpoint per request and remembers the choice per conversation."""

    def __init__(self, providers: Providers, routes: Routes,
                 cooldowns: Cooldowns | None = None):
        self.providers = providers
        self.routes = routes
        self.cooldowns = cooldowns or Cooldowns()
        self._lock = threading.Lock()
        self._rot: dict[str, int] = {}                  # model -> rotation cursor
        # (conversation, model) -> (endpoint, last seen). Keyed on BOTH: a
        # client that asks two models with the same prefix (a coding agent
        # using a cheap model for subagents and an expensive one for the main
        # thread shares its system prompt) produced ONE slot the two fought
        # over — and since `advance` was derived from "is there a pin", the
        # second model's cursor never moved: one whole model stopped rotating
        # and one of its plans expired unused. Verified 2026-08-27.
        self._sticky: dict[tuple[str, str], tuple[str, float]] = {}

    # ── candidate order ────────────────────────────────────────────────────

    def _ordered(self, model: str) -> list[str]:
        """Candidates for `model`, rotated to the pool's current head.

        Reads the cursor, never moves it: the cursor is set AFTER a pick, from
        the position actually chosen (`_advance_past`). Bumping it by one here
        looked equivalent and was not — with one plan parked, the plan after
        it received every skipped turn as well as its own (measured 2:1 over
        12 conversations), so it burned twice as fast, parked next, and handed
        ITS doubled share onward. A cascade that concentrates load exactly
        when the whole point is to spread it, and parking is the steady state.
        """
        cands = self.routes.candidates(model)
        n = self.routes.rotate_size(model)
        if n <= 1:
            return cands
        with self._lock:
            k = self._rot.get(model, 0) % n
        return cands[k:n] + cands[:k] + cands[n:]

    def _advance_past(self, model: str, endpoint: str) -> None:
        """Next new conversation starts one past the endpoint just handed out."""
        n = self.routes.rotate_size(model)
        if n <= 1:
            return
        try:
            i = self.routes.candidates(model).index(endpoint)
        except ValueError:
            return
        if i >= n:
            return      # the fallback tail is not part of the rotation
        with self._lock:
            self._rot[model] = (i + 1) % n

    def _stranded(self, model: str, pinned: str, cands: list[str]) -> bool:
        """Has something this conversation would have PREFERRED come back?

        Two shapes, one question. A rotate route prefers any pool member over
        the paid tail. A plain list is ordered failover — "bind the first" —
        so it prefers anything earlier than where the conversation ended up.
        The first version of this check bailed on `n <= 0`, which is exactly
        the plain-list shape the README draws (plan A → plan B → pay-as-you-go
        last): the rejoin never fired there, and a run that began during a
        park billed to the paid endpoint for its whole life — verbatim the
        thing the check was added to stop.
        """
        i = cands.index(pinned)
        n = self.routes.rotate_size(model)
        preferred = cands[:n] if n > 0 else cands[:i]
        if n > 0 and i < n:
            return False        # already in the pool; peers, not preferences
        return any(self._usable(c) is None for c in preferred)

    def _usable(self, endpoint: str) -> str | None:
        """None if usable, else why not — the reason is worth reporting."""
        provider, _ = endpoint.split("/", 1)
        if provider not in self.providers:
            return "provider not registered"
        key_env = self.providers.key_env(provider)
        if key_env and not self.providers.key(provider):
            # No key file = "I do not hold this plan". Binding it buys a
            # guaranteed auth failure; skipping it is what the absence means.
            return f"no key ({key_env})"
        if not self.cooldowns.available(endpoint):
            return f"parked ({self.cooldowns.reason(endpoint)})"
        return None

    # ── the decision ───────────────────────────────────────────────────────

    def pick(self, model: str, convo: str,
             exclude: frozenset[str] = frozenset()) -> tuple[str, list[str]]:
        """(endpoint, skipped-with-reasons) for this request.

        Raises KeyError if the model is not a route — deliberately NOT a
        passthrough: a typo'd model name would otherwise reach the upstream as
        a bad request that names neither the table nor the route it belongs in.
        """
        if model not in self.routes:
            raise KeyError(model)

        cands = self.routes.candidates(model)
        pinned = self._sticky_get(convo, model)
        if pinned is not None and pinned not in cands:
            # The table changed under us (a /reload that renamed endpoints).
            # A pin that names nothing is not a pin: leaving it set made every
            # affected conversation re-pick at the cursor head AND kept the
            # cursor still, so the whole pool collapsed onto one plan on the
            # first turn after any such reload.
            pinned = None
        rejoining = False
        if pinned and pinned not in exclude:
            if self._usable(pinned) is None:
                if not self._stranded(model, pinned, cands):
                    return pinned, []
                # Pinned to the pay-as-you-go tail (every plan was parked when
                # this conversation started) and a plan has since reopened.
                # This is the one stickiness case where holding still costs
                # money instead of saving it: a long agent run that began
                # during a park would bill to the paid endpoint for its whole
                # life. Rejoin the pool and take a real turn in it.
                rejoining = True
            # else: the conversation's endpoint went unusable mid-flight.
            # Re-pick WITHOUT advancing the cursor — one conversation's
            # problem is not everyone's turn to move.

        skipped: list[str] = []
        first_keyed: str | None = None
        first_recoverable: str | None = None
        # A conversation this model has never served (or one rejoining the
        # pool) takes a turn in the rotation. A re-pick after a failure does
        # not: `exclude` marks it, and it must not drag everyone's start along.
        fresh = (pinned is None or rejoining) and not exclude
        for endpoint in self._ordered(model):
            if endpoint in exclude:
                continue
            why = self._usable(endpoint)
            if why is None:
                self._sticky_set(convo, model, endpoint)
                if fresh:
                    self._advance_past(model, endpoint)
                return endpoint, skipped
            skipped.append(f"{endpoint}: {why}")
            if why.startswith("parked"):
                # Parked, but it has a key. Prefer one that will RECOVER: a
                # spent window reopens on its own, a rejected credential does
                # not, so "the first might answer" is false for the latter.
                if self.cooldowns.recoverable(endpoint):
                    if first_recoverable is None:
                        first_recoverable = endpoint
                elif first_keyed is None:
                    first_keyed = endpoint

        # Everything is parked or unusable. Degrade to "try it anyway" rather
        # than refuse: parking is parsed from provider prose and can be wrong,
        # and a misread timestamp must never make the router unusable. Prefer
        # a parked-but-keyed endpoint over a keyless one — the first might
        # answer, the second cannot.
        degraded = first_recoverable or first_keyed
        if degraded is not None:
            self._sticky_set(convo, model, degraded)
            if fresh:
                # The degrade is an exit from `pick` like any other, and it
                # was the ONE exit that did not advance the cursor: with the
                # whole pool parked (the steady state this product assumes)
                # every new conversation piled onto the same endpoint, so a
                # misparsed park was never re-probed on the other plans.
                self._advance_past(model, degraded)
            return degraded, skipped
        raise NoEndpoint(model, skipped)

    # ── stickiness ─────────────────────────────────────────────────────────

    def _sticky_get(self, convo: str, model: str) -> str | None:
        now = time.time()
        key = (convo, model)
        with self._lock:
            hit = self._sticky.get(key)
            if hit is None:
                return None
            endpoint, seen = hit
            if now - seen > _STICKY_TTL_S:
                self._sticky.pop(key, None)
                return None
            self._sticky[key] = (endpoint, now)
            return endpoint

    def _sticky_set(self, convo: str, model: str, endpoint: str) -> None:
        now = time.time()
        with self._lock:
            if len(self._sticky) >= _STICKY_MAX:
                self._sticky = {k: v for k, v in self._sticky.items()
                                if now - v[1] <= _STICKY_TTL_S}
                if len(self._sticky) >= _STICKY_MAX:
                    # Drop the OLDEST quarter, never the whole table. Clearing
                    # it re-rolled every live conversation at once — a cliff
                    # rather than an eviction, and it lands hardest on the
                    # client that fills the table fastest, which is the one
                    # whose conversations are being cut short.
                    ordered = sorted(self._sticky.items(), key=lambda kv: kv[1][1])
                    for k, _ in ordered[:max(1, _STICKY_MAX // 4)]:
                        self._sticky.pop(k, None)
            self._sticky[(convo, model)] = (endpoint, now)

    def forget(self, convo: str, model: str) -> None:
        """Drop one conversation's assignment for one model.

        The server calls this after an endpoint failed, so a LATER request in
        the same conversation is not pinned to an endpoint that just proved
        unhealthy. Within one request the retry needs nothing from this:
        `pick`'s `exclude` both skips the failed endpoint and holds the pool
        cursor still (a failing request used to advance it TWICE, skipping a
        plan in the rotation every time anything failed).
        """
        with self._lock:
            self._sticky.pop((convo, model), None)

    def reload(self, providers, routes) -> None:
        """Swap the tables in place, keeping cursors and assignments.

        Deliberately NOT a new Router: rebuilding one would drop every live
        conversation's endpoint — a reload during a long agent run would
        re-roll the dice mid-conversation and throw away the provider-side
        prefix cache this whole design exists to protect.
        """
        with self._lock:
            self.providers = providers
            self.routes = routes

    def stats(self) -> dict:
        with self._lock:
            sticky = len(self._sticky)
            cursors = dict(self._rot)
        return {"conversations": sticky, "cursors": cursors,
                "parked": self.cooldowns.active()}


class NoEndpoint(RuntimeError):
    def __init__(self, model: str, skipped: list[str]):
        self.model = model
        self.skipped = skipped
        super().__init__(
            f"no usable endpoint for '{model}': " + "; ".join(skipped))
