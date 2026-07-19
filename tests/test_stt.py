"""Two-way STT streaming logic, tested with an injected fake recognizer."""

import json

from signspeak.stt import Transcriber


class FakeRecognizer:
    """Mimics vosk.KaldiRecognizer: finalizes an utterance every N chunks."""

    def __init__(self, utterances, every=2):
        self._utterances = list(utterances)
        self._every = every
        self._count = 0
        self._current = None

    def AcceptWaveform(self, _chunk) -> bool:  # noqa: N802 (vosk API)
        self._count += 1
        if self._count % self._every == 0 and self._utterances:
            self._current = self._utterances.pop(0)
            return True
        return False

    def Result(self) -> str:  # noqa: N802
        return json.dumps({"text": self._current})

    def FinalResult(self) -> str:  # noqa: N802
        return json.dumps({"text": self._utterances.pop(0) if self._utterances else ""})


def test_streams_finalized_utterances():
    rec = FakeRecognizer(["hello there", "how are you"], every=2)
    out = list(Transcriber(rec).transcribe_stream([b"x"] * 4))
    assert out == ["hello there", "how are you"]


def test_final_result_flushes_trailing_text():
    rec = FakeRecognizer(["partial end"], every=100)  # never finalizes mid-stream
    out = list(Transcriber(rec).transcribe_stream([b"x"] * 3))
    assert out == ["partial end"]


def test_empty_results_are_skipped():
    rec = FakeRecognizer([""], every=1)
    out = list(Transcriber(rec).transcribe_stream([b"x"]))
    assert out == []
