"""Desktop TTS controller shared by the Thamudic and NLP Tkinter GUIs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DesktopVoice:
    """Small pyttsx3 controller with non-blocking GUI-friendly operations."""

    engine: Any = None

    def _get(self):
        if self.engine is None:
            try:
                import pyttsx3  # type: ignore
                self.engine = pyttsx3.init()
            except Exception:
                return None
        return self.engine

    def available(self) -> bool:
        return self._get() is not None

    def voices(self) -> list[dict[str, Any]]:
        engine = self._get()
        if engine is None:
            return []
        result = []
        for voice in engine.getProperty("voices") or []:
            languages = getattr(voice, "languages", []) or []
            result.append({"id": getattr(voice, "id", ""), "name": getattr(voice, "name", ""), "languages": [str(x) for x in languages]})
        return result

    def speak(self, text: str, language: str = "en", rate: int = 160, volume: float = 1.0) -> bool:
        text = text.strip()
        if not text:
            return False
        engine = self._get()
        if engine is None:
            return False
        engine.setProperty("rate", max(60, min(300, int(rate))))
        engine.setProperty("volume", max(0.0, min(1.0, float(volume))))
        # Prefer a voice whose advertised language matches the requested BCP-47
        # base language, while falling back to the operating-system default.
        base = language.casefold().split("-")[0]
        for voice in engine.getProperty("voices") or []:
            langs = " ".join(str(x).casefold() for x in (getattr(voice, "languages", []) or []))
            name = str(getattr(voice, "name", "")).casefold()
            if base in langs or (base == "ar" and "arab" in (langs + " " + name)) or (base == "en" and "english" in (langs + " " + name)):
                try:
                    engine.setProperty("voice", voice.id)
                    break
                except Exception:
                    pass
        engine.say(text)
        engine.runAndWait()
        return True

    def stop(self) -> None:
        if self.engine is not None:
            try:
                self.engine.stop()
            except Exception:
                pass
