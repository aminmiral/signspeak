"""Debouncing state-machine behavior (REQUIREMENTS FR-6)."""

from signspeak.sentence import SentenceBuilder


def make_builder(stable=3, threshold=0.75):
    return SentenceBuilder(confidence_threshold=threshold, stable_frames=stable)


def test_commits_after_stable_frames():
    b = make_builder(stable=3)
    assert b.update("HELLO", 0.9) is None
    assert b.update("HELLO", 0.9) is None
    assert b.update("HELLO", 0.9) == "HELLO"
    assert b.text == "HELLO"


def test_low_confidence_never_commits():
    b = make_builder(stable=3)
    for _ in range(10):
        assert b.update("HELLO", 0.5) is None
    assert b.words == []


def test_flicker_does_not_commit():
    b = make_builder(stable=3)
    b.update("HELLO", 0.9)
    b.update("YES", 0.9)  # interruption resets stability
    b.update("HELLO", 0.9)
    assert b.words == []


def test_no_immediate_repeat():
    b = make_builder(stable=2)
    b.update("YES", 0.9)
    assert b.update("YES", 0.9) == "YES"
    # continued stable prediction of the same sign must not re-commit
    b.update("YES", 0.9)
    assert b.update("YES", 0.9) is None
    assert b.words == ["YES"]


def test_repeat_allowed_after_hand_leaves_frame():
    b = make_builder(stable=2)
    b.update("YES", 0.9)
    assert b.update("YES", 0.9) == "YES"
    b.update(None)  # hand left the frame
    b.update("YES", 0.9)
    assert b.update("YES", 0.9) == "YES"
    assert b.words == ["YES", "YES"]


def test_confirm_mode_holds_candidate_out_of_sentence():
    b = SentenceBuilder(stable_frames=2, require_confirm=True)
    b.update("HELLO", 0.9)
    assert b.update("HELLO", 0.9) == "HELLO"
    assert b.pending == "HELLO" and b.words == []
    assert b.confirm() == "HELLO"
    assert b.words == ["HELLO"] and b.pending is None


def test_reject_discards_and_allows_resigning():
    b = SentenceBuilder(stable_frames=2, require_confirm=True)
    b.update("YES", 0.9)
    b.update("YES", 0.9)
    assert b.reject() == "YES"
    assert b.words == [] and b.pending is None
    # same sign can be re-signed immediately after rejection
    b.update("YES", 0.9)
    assert b.update("YES", 0.9) == "YES"
    assert b.pending == "YES"


def test_confirm_with_no_pending_is_noop():
    b = SentenceBuilder(require_confirm=True)
    assert b.confirm() is None and b.words == []


def test_undo_and_clear():
    b = make_builder(stable=1)
    b.update("HELLO", 0.9)
    b.update(None)
    b.update("YES", 0.9)
    assert b.words == ["HELLO", "YES"]
    assert b.undo() == "YES"
    assert b.text == "HELLO"
    b.clear()
    assert b.words == [] and b.undo() is None
