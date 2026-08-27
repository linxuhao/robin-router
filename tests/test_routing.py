"""The product is these semantics; the HTTP layer is plumbing around them."""
import pytest

from robin.routing import NoEndpoint, Router, conversation_id


def _c(system="you are helpful", user="hello"):
    return conversation_id({"model": "flash", "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]})


# ── the two halves of the thesis ───────────────────────────────────────────

def test_the_same_conversation_stays_on_one_endpoint(router):
    """Prefix caches are per-provider and agent clients replay the whole
    transcript every turn: moving a live conversation converts cached input
    into full-price input."""
    convo = _c()
    first, _ = router.pick("flash", convo)
    for _ in range(10):
        again, _ = router.pick("flash", convo)
        assert again == first


def test_a_new_conversation_starts_on_the_next_plan(router):
    """Otherwise one plan burns out while the others' windows expire idle —
    the entire reason this exists."""
    seen = [router.pick("flash", _c(user=f"q{i}"))[0] for i in range(3)]
    assert seen == ["a/m", "b/m", "c/m"]
    assert router.pick("flash", _c(user="q3"))[0] == "a/m"     # wraps


def test_the_growing_transcript_is_still_one_conversation(router):
    """The id must key on the PREFIX, not the message list: keying on the
    whole list would re-roll on every single turn — a per-call rotator with
    extra steps."""
    turns = [{"role": "system", "content": "sys"},
             {"role": "user", "content": "first"}]
    first = conversation_id({"messages": list(turns)})
    for i in range(5):
        turns.append({"role": "assistant", "content": f"a{i}"})
        turns.append({"role": "user", "content": f"u{i}"})
        assert conversation_id({"messages": list(turns)}) == first


def test_pay_as_you_go_never_rotates_into_the_head(router):
    """Money is the last resort, not a peer in the pool."""
    for i in range(12):
        assert router.pick("flash", _c(user=f"q{i}"))[0] != "payg/m"


# ── parking ────────────────────────────────────────────────────────────────

def test_a_spent_window_is_skipped_not_re_probed(router):
    router.cooldowns.park("a/m", "quota exhausted, resets at 2099-01-01T00:00:00Z")
    picks = {router.pick("flash", _c(user=f"q{i}"))[0] for i in range(6)}
    assert "a/m" not in picks
    assert picks == {"b/m", "c/m"}


def test_a_conversation_whose_endpoint_parks_moves_without_moving_everyone(router):
    """One conversation's problem is not everyone's turn to rotate: re-picking
    must not advance the shared cursor, or a single failing conversation drags
    every other conversation's starting point with it."""
    stuck = _c(user="stuck")
    assert router.pick("flash", stuck)[0] == "a/m"
    router.cooldowns.park("a/m", "quota exhausted")
    moved, _ = router.pick("flash", stuck)
    assert moved != "a/m"
    # The next NEW conversation still gets the cursor's next slot (b/m), not
    # a position shoved along by the re-pick above.
    assert router.pick("flash", _c(user="fresh"))[0] == "b/m"


def test_everything_parked_degrades_to_trying_anyway(router):
    """Parking is parsed from provider prose. A misread timestamp must never
    make the router unusable — but the degrade must prefer an endpoint that
    HAS a key: a parked one might answer, a keyless one cannot."""
    for e in ("a/m", "b/m", "c/m", "payg/m"):
        router.cooldowns.park(e, "quota exhausted")
    picked, skipped = router.pick("flash", _c())
    assert picked in {"a/m", "b/m", "c/m", "payg/m"}
    assert len(skipped) == 4


def test_a_keyless_plan_is_skipped_with_a_reason(router, tables):
    """A key name with no file means 'I do not hold this plan' — binding it
    buys a guaranteed auth failure."""
    _, secrets = tables
    (secrets / "A_KEY").unlink()
    picked, skipped = router.pick("flash", _c())
    assert picked != "a/m"
    assert any("no key (A_KEY)" in s for s in skipped)


def test_a_keyless_plan_loses_to_a_parked_one_in_the_degrade(router, tables):
    _, secrets = tables
    (secrets / "A_KEY").unlink()
    (secrets / "C_KEY").unlink()
    (secrets / "P_KEY").unlink()
    router.cooldowns.park("b/m", "quota exhausted")
    assert router.pick("flash", _c())[0] == "b/m"      # parked, but keyed


def test_no_usable_endpoint_at_all_raises_naming_every_reason(router, tables):
    _, secrets = tables
    for k in ("A_KEY", "B_KEY", "C_KEY", "P_KEY"):
        (secrets / k).unlink()
    with pytest.raises(NoEndpoint) as e:
        router.pick("flash", _c())
    assert e.value.skipped and all("no key" in s for s in e.value.skipped)


# ── shapes ─────────────────────────────────────────────────────────────────

def test_a_plain_list_route_does_not_rotate(router):
    """Ordered failover only — the pre-existing shape must keep its meaning."""
    assert {router.pick("plain", _c(user=f"q{i}"))[0] for i in range(5)} == {"a/m"}


def test_an_unknown_model_raises_rather_than_passing_through(router):
    """A typo'd model name must name the table, not reach the upstream as a
    bad request that points nowhere near the cause."""
    with pytest.raises(KeyError):
        router.pick("flsh", _c())


def test_exclusion_lets_a_caller_retry_past_a_failed_endpoint(router):
    first, _ = router.pick("flash", _c())
    second, _ = router.pick("flash", _c(), exclude=frozenset({first}))
    assert second != first
