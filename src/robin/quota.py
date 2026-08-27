"""Telling a spent plan from a burst, and believing the provider's own clock.

Both arrive as HTTP 429. Only the prose says which, and the difference decides
everything: a burst wants a retry in seconds, a spent 5-hour window wants that
endpoint skipped until it reopens. Re-probing a spent plan once per call is how
you turn one dead plan into a per-call tax.
"""
from __future__ import annotations

import datetime
import re
import time

# "resets at 2026-08-27T18:00:00Z", "try again after 2026-08-27 18:00",
# "quota resets 2026-08-27T18:00:00+08:00" — providers phrase it differently
# but they all name an instant, and the instant is worth more than any guess.
_RESET_RE = re.compile(
    r"(?:reset|resets|available|try\s+again|retry)\D{0,24}"
    r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?)",
    re.I)

# A spent PLAN says so in words; a burst says "too many requests". When the
# text matches neither, treat it as a burst: parking an endpoint that was only
# throttled costs real capacity, while re-trying a spent one costs one call.
_SPENT_RE = re.compile(
    r"quota|usage limit|out of credit|insufficient balance|plan .*exhaust"
    r"|exceeded your current|monthly limit|weekly limit|5[- ]hour limit",
    re.I)

_MAX_S = 6 * 3600      # never trust one report further than this
_FALLBACK_S = 300      # provider named no instant


def is_spent(text: str) -> bool:
    """Does this 429/403 body describe an exhausted allowance (not a burst)?"""
    return bool(_SPENT_RE.search(text or ""))


def reset_at(text: str) -> float | None:
    """Epoch seconds the provider says the window reopens, if it named one."""
    m = _RESET_RE.search(text or "")
    if not m:
        return None
    stamp = m.group(1).replace(" ", "T")
    if not re.search(r"(?:Z|[+-]\d{2}:?\d{2})$", stamp):
        stamp += "+00:00"      # naive instants from an API are UTC
    try:
        return datetime.datetime.fromisoformat(
            stamp.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


class Cooldowns:
    """Endpoints parked until an instant. Keyed on `provider/model`.

    The key width is a real trade: too narrow costs one wasted call per model
    per window; too broad silently retires every model behind one key. Per
    endpoint is the honest middle — a plan is usually per-key, but a model can
    be individually rate-limited.

    In-process on purpose. Persisting it would add a store to keep correct
    across restarts, and the cost of forgetting is one call.
    """

    def __init__(self):
        self._until: dict[str, float] = {}

    def available(self, endpoint: str) -> bool:
        return self._until.get(endpoint, 0.0) <= time.time()

    def park(self, endpoint: str, text: str) -> float:
        """Park `endpoint` until the provider's stated reset (or a default)."""
        when = reset_at(text)
        now = time.time()
        if when is None or when <= now:
            when = now + _FALLBACK_S
        when = min(when, now + _MAX_S)
        self._until[endpoint] = when
        return when

    def active(self) -> dict[str, float]:
        now = time.time()
        return {k: v for k, v in self._until.items() if v > now}

    def clear(self) -> None:
        self._until.clear()
