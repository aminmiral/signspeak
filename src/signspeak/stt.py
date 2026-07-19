"""Speech-to-text for two-way mode (REQUIREMENTS FR-11): the hearing person's
replies become on-screen text for the signer.

Backend: Vosk — offline, streaming, small models, the documented best choice
for real-time low-resource captioning. Optional dependency:
    pip install -e ".[stt]"
plus a model from https://alphacephei.com/vosk/models (e.g.
vosk-model-small-en-in-0.4 for Indian English) unpacked into models/vosk/.

The recognizer is injectable so the streaming logic is testable without audio.
"""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path


class Transcriber:
    """Streams audio chunks through a Vosk-style recognizer, yields text."""

    def __init__(self, recognizer):
        self._rec = recognizer

    @classmethod
    def from_model(cls, model_dir: str | Path, sample_rate: int = 16000):
        try:
            import vosk
        except ImportError as exc:
            raise ImportError(
                'Vosk is required for two-way mode: pip install -e ".[stt]"'
            ) from exc
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"No Vosk model at {model_dir}. Download one from "
                "https://alphacephei.com/vosk/models (en-in for Indian English) "
                "and unpack it there."
            )
        vosk.SetLogLevel(-1)
        model = vosk.Model(str(model_dir))
        return cls(vosk.KaldiRecognizer(model, sample_rate))

    def transcribe_stream(self, chunks: Iterable[bytes]) -> Iterator[str]:
        """Yield finalized utterances as they complete."""
        for chunk in chunks:
            if self._rec.AcceptWaveform(chunk):
                text = json.loads(self._rec.Result()).get("text", "").strip()
                if text:
                    yield text
        text = json.loads(self._rec.FinalResult()).get("text", "").strip()
        if text:
            yield text


def mic_chunks(sample_rate: int = 16000, block_ms: int = 250) -> Iterator[bytes]:
    """Microphone audio chunks via sounddevice ([stt] extra). Blocks forever."""
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise ImportError(
            'sounddevice is required for mic capture: pip install -e ".[stt]"'
        ) from exc
    import queue

    q: queue.Queue[bytes] = queue.Queue()

    def callback(indata, _frames, _time, _status) -> None:
        q.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=sample_rate,
        blocksize=int(sample_rate * block_ms / 1000),
        dtype="int16",
        channels=1,
        callback=callback,
    ):
        while True:
            yield q.get()
