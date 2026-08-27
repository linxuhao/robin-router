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


# ── (conversation, model) is the assignment key, not conversation alone ────

def test_two_models_sharing_a_prefix_each_rotate_independently(router):
    """A coding agent uses a cheap model for subagents and an expensive one
    for the main thread — same system prompt, so the SAME conversation id.
    Keyed on the conversation alone, the two fought over one sticky slot and,
    because `advance` was derived from "is there a pin", the second model's
    cursor never moved: one whole model stopped rotating and one of its plans
    expired unused — the exact failure this project exists to prevent."""
    seen_flash, seen_plain = [], []
    for i in range(4):
        convo = _c(user=f"q{i}")
        seen_flash.append(router.pick("flash", convo)[0])
        seen_plain.append(router.pick("plain", convo)[0])
    assert seen_flash == ["a/m", "b/m", "c/m", "a/m"]      # rotates
    assert seen_plain == ["a/m"] * 4                       # plain list, no pool
    # And each model keeps its OWN pin for the same conversation.
    convo = _c(user="shared")
    f = router.pick("flash", convo)[0]
    router.pick("plain", convo)
    assert router.pick("flash", convo)[0] == f             # not clobbered


def test_a_failed_attempt_does_not_advance_the_cursor_twice(router):
    """The server unpins a conversation after an endpoint fails so the retry
    is not held to it — but that must not read as a brand-new conversation.
    One failing request used to move the cursor twice, skipping a plan in the
    rotation every time anything failed."""
    convo = _c(user="fails")
    first, _ = router.pick("flash", convo)
    router.forget(convo, "flash")                  # what the server does
    second, _ = router.pick("flash", convo, exclude=frozenset({first}))
    assert second != first
    assert router.pick("flash", _c(user="next"))[0] == "b/m"   # not c/m


# ── the degrade and the rejoin, on every route shape ───────────────────────

def test_the_degrade_path_also_takes_a_turn_in_the_rotation(router):
    """It was the one exit from `pick` that never moved the cursor, so with
    the whole pool parked — the steady state this product assumes — every new
    conversation piled onto one endpoint and the other plans were never
    re-probed, even though a misparsed park is exactly why the degrade
    exists."""
    for e in ("a/m", "b/m", "c/m", "payg/m"):
        router.cooldowns.park(e, "quota exhausted")
    picks = [router.pick("flash", _c(user=f"q{i}"))[0] for i in range(3)]
    assert len(set(picks)) == 3, picks


def test_a_plain_list_route_rejoins_its_preferred_endpoint(router):
    """Ordered failover means "bind the first". A conversation that fell to
    the tail while the head was parked must come back when it reopens — the
    first version of this check bailed on plain lists, so a run that began
    during a park billed to the paid endpoint for its whole life."""
    convo = _c(user="plain")
    router.cooldowns.park("a/m", "quota exhausted")
    assert router.pick("plain", convo)[0] == "payg/m"
    router.cooldowns.unpark("a/m")
    assert router.pick("plain", convo)[0] == "a/m"


def test_a_credential_park_is_not_the_preferred_degrade(router, tables):
    """"The first might answer" is true of a spent window and false of a
    rejected credential: only one of them recovers by waiting."""
    router.cooldowns.park("a/m", "", why="credential rejected")
    router.cooldowns.park("b/m", "", why="window spent")
    router.cooldowns.park("c/m", "", why="credential rejected")
    router.cooldowns.park("payg/m", "", why="credential rejected")
    assert router.pick("flash", _c())[0] == "b/m"


def test_a_pin_the_table_no_longer_contains_is_not_a_pin(router, tables):
    """After a /reload that renames endpoints, a stale pin used to keep
    `fresh` False — so every affected conversation re-picked at the cursor
    head AND the cursor never moved: the whole pool collapsed onto one plan."""
    import json as _json
    from robin.config import Routes
    tmp_path, _ = tables
    convos = [_c(user=f"q{i}") for i in range(3)]
    before = [router.pick("flash", c)[0] for c in convos]
    assert len(set(before)) == 3

    (tmp_path / "model_routes.json").write_text(_json.dumps(
        {"flash": {"rotate": ["a/m2", "b/m2", "c/m2"], "fallback": []},
         "plain": ["a/m"]}))
    router.reload(router.providers, Routes(tmp_path / "model_routes.json"))
    after = [router.pick("flash", c)[0] for c in convos]
    assert len(set(after)) == 3, after
