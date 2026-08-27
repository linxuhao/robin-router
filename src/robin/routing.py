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
        self._sticky: dict[str, tuple[str, float]] = {}  # convo -> (endpoint, seen)

    # ── candidate order ────────────────────────────────────────────────────

    def _ordered(self, model: str, advance: bool) -> list[str]:
        """Candidates for `model`, rotated if the route has a pool.

        `advance` is what separates a new conversation (take the next plan and
        move the cursor) from a retry of one already assigned (same order, no
        cursor movement) — otherwise a failing conversation would walk the
        whole pool and drag every other conversation's starting point with it.
        """
        cands = self.routes.candidates(model)
        n = self.routes.rotate_size(model)
        if n <= 1:
            return cands
        with self._lock:
            k = self._rot.get(model, 0)
            if advance:
                self._rot[model] = (k + 1) % n
        return cands[k:n] + cands[:k] + cands[n:]

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
            return "window spent"
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

        pinned = self._sticky_get(convo)
        if pinned and pinned not in exclude and pinned in self.routes.candidates(model):
            if self._usable(pinned) is None:
                return pinned, []
            # The conversation's endpoint went unusable mid-flight (window
            # spent). Re-pick WITHOUT advancing the pool cursor: this is one
            # conversation's problem, not everyone's turn to move.

        skipped: list[str] = []
        first_keyed: str | None = None
        for endpoint in self._ordered(model, advance=pinned is None):
            if endpoint in exclude:
                continue
            why = self._usable(endpoint)
            if why is None:
                self._sticky_set(convo, endpoint)
                return endpoint, skipped
            skipped.append(f"{endpoint}: {why}")
            if why == "window spent" and first_keyed is None:
                first_keyed = endpoint     # parked, but it HAS a key

        # Everything is parked or unusable. Degrade to "try it anyway" rather
        # than refuse: parking is parsed from provider prose and can be wrong,
        # and a misread timestamp must never make the router unusable. Prefer
        # a parked-but-keyed endpoint over a keyless one — the first might
        # answer, the second cannot.
        if first_keyed is not None:
            self._sticky_set(convo, first_keyed)
            return first_keyed, skipped
        raise NoEndpoint(model, skipped)

    # ── stickiness ─────────────────────────────────────────────────────────

    def _sticky_get(self, convo: str) -> str | None:
        now = time.time()
        with self._lock:
            hit = self._sticky.get(convo)
            if hit is None:
                return None
            endpoint, seen = hit
            if now - seen > _STICKY_TTL_S:
                self._sticky.pop(convo, None)
                return None
            self._sticky[convo] = (endpoint, now)
            return endpoint

    def _sticky_set(self, convo: str, endpoint: str) -> None:
        now = time.time()
        with self._lock:
            if len(self._sticky) >= _STICKY_MAX:
                # Drop what has aged out; if that frees nothing, drop it all.
                # These are one-hour hints: losing them costs each live
                # conversation one re-pick, and a wrong eviction policy here
                # would be more code than the cache.
                self._sticky = {k: v for k, v in self._sticky.items()
                                if now - v[1] <= _STICKY_TTL_S}
                if len(self._sticky) >= _STICKY_MAX:
                    self._sticky.clear()
            self._sticky[convo] = (endpoint, now)

    def forget(self, convo: str) -> None:
        with self._lock:
            self._sticky.pop(convo, None)

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
