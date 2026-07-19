"""Pure debouncing state machine + sentence buffer.

A sign is committed only when (FR-6):
  - prediction confidence >= confidence_threshold, AND
  - the same label was predicted for `stable_frames` consecutive updates, AND
  - it differs from the last committed sign.

The hand leaving the frame (update with label=None) resets the repeat guard so
the same sign can be signed twice in a row.
"""

from collections import deque


class SentenceBuilder:
    """Debouncer + sentence buffer, with optional candidate-confirm mode (FR-8).

    With ``require_confirm=True`` a stable sign becomes a *pending candidate*
    instead of entering the sentence; the signer explicitly ``confirm()``s or
    ``reject()``s it. This keeps the signer in control of what gets spoken —
    the core UX requirement from deaf-community research (PRD §4.2).
    """

    def __init__(
        self,
        confidence_threshold: float = 0.75,
        stable_frames: int = 15,
        require_confirm: bool = False,
    ):
        self.confidence_threshold = confidence_threshold
        self.stable_frames = stable_frames
        self.require_confirm = require_confirm
        self._recent: deque[str | None] = deque(maxlen=stable_frames)
        self._last_committed: str | None = None
        self.words: list[str] = []
        self.pending: str | None = None

    def update(self, label: str | None, confidence: float = 0.0) -> str | None:
        """Feed one prediction; returns the newly stable sign, or None."""
        if label is None:
            self._recent.append(None)
            self._last_committed = None  # hand left frame -> allow repeats
            return None

        self._recent.append(label if confidence >= self.confidence_threshold else None)

        if (
            len(self._recent) == self.stable_frames
            and all(p == label for p in self._recent)
            and label != self._last_committed
        ):
            self._last_committed = label
            self._recent.clear()
            if self.require_confirm:
                self.pending = label
            else:
                self.words.append(label)
            return label
        return None

    def confirm(self) -> str | None:
        """Accept the pending candidate into the sentence."""
        if self.pending is None:
            return None
        word, self.pending = self.pending, None
        self.words.append(word)
        return word

    def reject(self) -> str | None:
        """Discard the pending candidate."""
        word, self.pending = self.pending, None
        self._last_committed = None  # allow re-signing the same word immediately
        return word

    def undo(self) -> str | None:
        return self.words.pop() if self.words else None

    def clear(self) -> None:
        self.words.clear()
        self._recent.clear()
        self._last_committed = None
        self.pending = None

    @property
    def text(self) -> str:
        return " ".join(self.words)
