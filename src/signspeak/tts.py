"""Tiered text-to-speech (REQUIREMENTS FR-10).

Tiers (accessibility research: docs/PRD.md §5/F-10):
  1. "online"  — edge-tts neural voices, incl. Indian English (en-IN-NeerjaNeural)
                 and Hindi (hi-IN-SwaraNeural). Needs internet + the [online]
                 extra + a CLI audio player (mpg123 or ffplay).
  2. "offline" — pyttsx3 via OS voices. Robotic but always works.

Backend "auto" picks online when its prerequisites exist, else offline.
Speech always runs off the main thread so the video loop never blocks.
"""

import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

DEFAULT_ONLINE_VOICE = "en-IN-NeerjaNeural"

_PLAYERS = {
    "mpg123": ["mpg123", "-q"],
    "ffplay": ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
}


def _find_player() -> list[str] | None:
    for name, cmd in _PLAYERS.items():
        if shutil.which(name):
            return cmd
    return None


def _edge_available() -> bool:
    try:
        import edge_tts  # noqa: F401
    except ImportError:
        return False
    return True


def choose_backend(preference: str = "auto") -> str:
    """Resolve a backend preference ('auto'|'online'|'offline') to a concrete one."""
    if preference == "offline":
        return "offline"
    online_ready = _edge_available() and _find_player() is not None
    if preference == "online" and not online_ready:
        raise RuntimeError(
            "Online TTS needs the [online] extra (pip install -e '.[online]') "
            "and an audio player (mpg123 or ffplay) on PATH."
        )
    return "online" if online_ready else "offline"


def _speak_offline(text: str) -> None:
    import pyttsx3  # imported here: audio backends can be absent in CI

    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def _speak_online(text: str, voice: str) -> None:
    import asyncio

    import edge_tts

    player = _find_player()
    if player is None:  # pragma: no cover - guarded by choose_backend
        return _speak_offline(text)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        asyncio.run(edge_tts.Communicate(text, voice).save(str(path)))
        subprocess.run([*player, str(path)], check=False)
    except Exception:
        _speak_offline(text)  # graceful degradation, never lose the utterance
    finally:
        path.unlink(missing_ok=True)


def speak_async(
    text: str,
    backend: str = "auto",
    voice: str = DEFAULT_ONLINE_VOICE,
) -> threading.Thread:
    """Speak in a background thread; returns the thread."""
    resolved = choose_backend(backend)

    def _run() -> None:
        if resolved == "online":
            _speak_online(text, voice)
        else:
            _speak_offline(text)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread
