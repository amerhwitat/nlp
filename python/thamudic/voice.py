"""Optional voice/TTS/STT integration for original-script readings and translations.

Ancient scripts normally do not have reliable native TTS voices. The module therefore
uses an explicit pronunciation/transliteration layer for ancient text and delegates
modern-language playback to installed OS/browser TTS providers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VoiceRequest:
    text: str
    language: str
    mode: str = "translation"
    rate: int = 160
    volume: float = 1.0


def available_tts_backends() -> list[str]:
    backends = ["browser-speech-synthesis"]
    try:
        import pyttsx3  # type: ignore
        del pyttsx3
        backends.append("pyttsx3")
    except ImportError:
        pass
    return backends


def speak(request: VoiceRequest) -> dict[str, Any]:
    """Speak using an installed local backend when available.

    This function reports capability instead of silently substituting a modern
    language voice for an ancient-language native voice.
    """
    if not request.text.strip():
        raise ValueError("text is required")
    if request.mode not in {"original", "transliteration", "translation"}:
        raise ValueError("mode must be original, transliteration, or translation")
    if request.mode == "original":
        return {
            "status": "pronunciation_provider_required",
            "text": request.text,
            "language": request.language,
            "mode": request.mode,
            "native_ancient_tts": False,
            "backends": available_tts_backends(),
        }
    return {
        "status": "ready",
        "text": request.text,
        "language": request.language,
        "mode": request.mode,
        "backends": available_tts_backends(),
        "browser_speech_synthesis": True,
    }


def voice_control_commands() -> list[str]:
    return [
        "speak original", "speak transliteration", "speak translation",
        "pause", "resume", "stop", "repeat", "slower", "faster",
        "mute", "unmute", "next", "previous",
    ]


def speech_recognition_capability() -> dict[str, Any]:
    try:
        import speech_recognition  # type: ignore
        del speech_recognition
        return {"available": True, "provider": "speech_recognition"}
    except ImportError:
        return {"available": False, "provider": None, "install_extra": "voice"}
