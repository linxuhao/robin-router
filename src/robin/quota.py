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
    r"usage limit|out of credit|insufficient balance|insufficient credit"
    r"|plan .*exhaust|exceeded your current|monthly limit|weekly limit"
    r"|5[- ]hour limit|quota exhausted|quota exceeded"
    # A per-DAY cap is a window like any other: it does not clear in seconds,
    # so re-probing it costs one call per request until midnight. The veto
    # below still excludes the sub-minute ceilings that DO clear on their own.
    r"|daily limit|per[- ]day|\brpd\b", re.I)

# A THROTTLE wears the same words as a spent plan. Ark says "the tpm quota for
# this model has been exceeded"; Google says "Quota exceeded for quota metric
# 'requests per minute'". Both are per-minute ceilings that clear in seconds —
# parking either for the 5-minute default throws away the capacity you pay for,
# on the primary plan, during exactly the burst that tripped it. The asymmetry
# decides: wrongly parking costs real throughput, wrongly retrying costs one
# call. So a rate word vetoes the spend words.
# NARROW on purpose. An earlier version vetoed on `requests? per` and `too
# many requests`, which also matched per-DAY caps ("Limit 200 requests per
# day, Used 200") and the plain 429 reason phrase that gateways echo into the
# body — so a genuinely spent plan was never parked and got re-probed on every
# request for the rest of the day: the per-call tax this module exists to
# remove, reintroduced by its own safety valve. Only sub-minute windows, which
# clear on their own in seconds, may veto.
_THROTTLE_RE = re.compile(
    r"\btpm\b|\brpm\b|per[- ](?:minute|second)|concurren", re.I)

_MAX_S = 6 * 3600      # never trust one report further than this
_FALLBACK_S = 300      # provider named no instant


def is_spent(text: str) -> bool:
    """Does this body describe an exhausted ALLOWANCE, not a burst?"""
    text = text or ""
    if _THROTTLE_RE.search(text):
        return False
    return bool(_SPENT_RE.search(text))


def reset_at(text: str) -> float | None:
    """Epoch seconds the provider says the window reopens, if it named one."""
    m = _RESET_RE.search(text or "")
    if not m:
        return None
    stamp = m.group(1).replace(" ", "T")
    if not re.search(r"(?:Z|[+-]\d{2}:?\d{2})$", stamp):
        # No offset: unparseable, NOT "assume UTC". Several providers report
        # local time, and reading a +08:00 instant as UTC parks a healthy plan
        # eight hours out — clamped to the 6h ceiling, which then bounds the
        # damage instead of preventing it. Fall back to the short default and
        # re-probe; that costs one call, the wrong guess costs a whole plan.
        return None
    try:
        return datetime.datetime.fromisoformat(
            stamp.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _retry_after_at(value: str, now: float) -> float | None:
    """`Retry-After` as an epoch: either delta-seconds or an HTTP date."""
    value = (value or "").strip()
    if not value:
        return None
    try:
        return now + max(0.0, float(value))
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(value).timestamp()
    except (TypeError, ValueError):
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
        self._why: dict[str, str] = {}

    def available(self, endpoint: str) -> bool:
        return self._until.get(endpoint, 0.0) <= time.time()

    def reason(self, endpoint: str) -> str:
        """Why this endpoint is parked. '' when it is not.

        Kept because the reasons are not interchangeable: a spent window
        reopens on its own, a rejected credential does not. Reporting both as
        "window spent" made /stats lie and made the degrade path prefer an
        endpoint that is known to be dead.
        """
        return self._why.get(endpoint, "") if not self.available(endpoint) else ""

    # Parks that waiting alone will not clear: they need a human to rotate a
    # key or top up a balance. The degrade path must not prefer one of these
    # over a spent window that genuinely reopens.
    _NEEDS_A_HUMAN = frozenset({"credential rejected", "balance drained"})

    def recoverable(self, endpoint: str) -> bool:
        """Will waiting fix it?"""
        return self.reason(endpoint) not in self._NEEDS_A_HUMAN

    def park(self, endpoint: str, text: str, retry_after: str = "",
             why: str = "window spent") -> float:
        """Park `endpoint` until the provider says it reopens.

        `Retry-After` first when present: it is a header with one meaning,
        while the prose is a sentence a provider is free to reword — and a
        reworded sentence silently degrades to the 5-minute default, which
        re-probes a spent plan twelve times an hour.
        """
        now = time.time()
        when = _retry_after_at(retry_after, now) or reset_at(text)
        if when is None or when <= now:
            when = now + _FALLBACK_S
        when = min(when, now + _MAX_S)
        self._until[endpoint] = when
        self._why[endpoint] = why
        return when

    def unpark(self, endpoint: str | None = None) -> list[str]:
        """Release one park, or all of them. Returns what was released.

        Parking is inferred from provider prose and headers, so it can be
        wrong — and a wrong park hides a plan you are paying for. Without
        this the only remedy was restarting the proxy, which also drops every
        conversation's endpoint assignment: a heuristic you cannot correct
        cheaply is a heuristic you cannot afford to be wrong.
        """
        if endpoint is None:
            released = sorted(self.active())
            self._until.clear()
            self._why.clear()
            return released
        self._why.pop(endpoint, None)
        return [endpoint] if self._until.pop(endpoint, None) else []

    def active(self) -> dict[str, float]:
        now = time.time()
        return {k: v for k, v in self._until.items() if v > now}

