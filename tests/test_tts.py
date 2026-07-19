"""TTS backend selection logic (FR-10). No audio is played in tests."""

import pytest

from signspeak import tts


def test_offline_preference_is_always_offline():
    assert tts.choose_backend("offline") == "offline"


def test_auto_falls_back_without_edge_tts(monkeypatch):
    monkeypatch.setattr(tts, "_edge_available", lambda: False)
    assert tts.choose_backend("auto") == "offline"


def test_auto_falls_back_without_player(monkeypatch):
    monkeypatch.setattr(tts, "_edge_available", lambda: True)
    monkeypatch.setattr(tts, "_find_player", lambda: None)
    assert tts.choose_backend("auto") == "offline"


def test_auto_picks_online_when_ready(monkeypatch):
    monkeypatch.setattr(tts, "_edge_available", lambda: True)
    monkeypatch.setattr(tts, "_find_player", lambda: ["mpg123", "-q"])
    assert tts.choose_backend("auto") == "online"


def test_explicit_online_raises_when_not_ready(monkeypatch):
    monkeypatch.setattr(tts, "_edge_available", lambda: False)
    with pytest.raises(RuntimeError, match=r"\[online\] extra"):
        tts.choose_backend("online")


def test_speak_async_uses_resolved_backend(monkeypatch):
    spoken = {}
    monkeypatch.setattr(tts, "choose_backend", lambda p: "offline")
    monkeypatch.setattr(tts, "_speak_offline", lambda t: spoken.setdefault("text", t))
    thread = tts.speak_async("HELLO WORLD")
    thread.join(timeout=5)
    assert spoken["text"] == "HELLO WORLD"
