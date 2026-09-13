"""Tkinter desktop UI for scanning, transliteration, translation and voice controls."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .ancient_translation import supported_targets, translate
from .voice import VoiceRequest, speak as voice_capability


class ThamudicTranslatorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Ancient Script / Thamudic Translator + Voice")
        self.geometry("1080x820")
        self._tts_engine = None
        self._build()

    def _build(self) -> None:
        ttk.Label(self, text="Original inscription / scholarly transliteration").pack(anchor="w", padx=12, pady=(12, 4))
        self.source = tk.Text(self, height=8, wrap="word")
        self.source.pack(fill="x", padx=12)
        controls = ttk.Frame(self); controls.pack(fill="x", padx=12, pady=10)
        ttk.Label(controls, text="Script").pack(side="left")
        self.script = ttk.Combobox(controls, values=["Dadanitic", "Safaitic", "Hismaic", "Taymanitic", "Thamudic B"], state="readonly", width=18)
        self.script.set("Dadanitic"); self.script.pack(side="left", padx=6)
        ttk.Label(controls, text="Target").pack(side="left", padx=(12, 0))
        self.target = ttk.Combobox(controls, values=list(supported_targets()), state="readonly", width=10)
        self.target.set("en"); self.target.pack(side="left", padx=6)
        ttk.Button(controls, text="Translate + transliterate", command=self.translate).pack(side="left", padx=12)

        ttk.Label(self, text="Transliteration").pack(anchor="w", padx=12, pady=(4, 4))
        self.transliteration = tk.Text(self, height=5, wrap="word"); self.transliteration.pack(fill="x", padx=12)
        ttk.Label(self, text="Translation").pack(anchor="w", padx=12, pady=(12, 4))
        self.translation = tk.Text(self, height=7, wrap="word"); self.translation.pack(fill="both", expand=True, padx=12)

        voice_frame = ttk.LabelFrame(self, text="Voice controls")
        voice_frame.pack(fill="x", padx=12, pady=10)
        ttk.Button(voice_frame, text="▶ Original", command=lambda: self.speak_original()).pack(side="left", padx=4, pady=6)
        ttk.Button(voice_frame, text="▶ Transliteration", command=lambda: self.speak(self.transliteration.get("1.0", "end-1c"), self.target.get(), "transliteration")).pack(side="left", padx=4)
        ttk.Button(voice_frame, text="▶ Translation", command=lambda: self.speak(self.translation.get("1.0", "end-1c"), self.target.get(), "translation")).pack(side="left", padx=4)
        ttk.Button(voice_frame, text="Pause", command=self.pause_voice).pack(side="left", padx=4)
        ttk.Button(voice_frame, text="Resume", command=self.resume_voice).pack(side="left", padx=4)
        ttk.Button(voice_frame, text="Stop", command=self.stop_voice).pack(side="left", padx=4)

        self.status = ttk.Label(self, text="Ready")
        self.status.pack(anchor="w", padx=12, pady=8)

    def translate(self) -> None:
        text = self.source.get("1.0", "end-1c").strip()
        if not text:
            self.status.config(text="Enter source text first")
            return
        result = translate(text, script=self.script.get(), target_language=self.target.get())
        self.transliteration.delete("1.0", "end"); self.transliteration.insert("1.0", result["transliteration"])
        self.translation.delete("1.0", "end"); self.translation.insert("1.0", result["translation"] or "No corpus-backed translation is available for this reading.")
        self.status.config(text=f"{result['translation_status']} · confidence={result['confidence']} · corpus={result['corpus_id'] or 'none'}")

    def _engine(self):
        if self._tts_engine is None:
            try:
                import pyttsx3  # type: ignore
                self._tts_engine = pyttsx3.init()
            except Exception:
                return None
        return self._tts_engine

    def speak_original(self) -> None:
        result = voice_capability(VoiceRequest(self.source.get("1.0", "end-1c"), self.script.get(), "original"))
        if result["status"] != "ready":
            self.status.config(text="Native ancient-script pronunciation provider required; no modern voice substitution made")
            return
        self.speak(self.source.get("1.0", "end-1c"), "", "original")

    def speak(self, text: str, language: str, mode: str) -> None:
        if not text.strip(): return
        if mode == "original":
            self.status.config(text="Native ancient-script pronunciation provider required")
            return
        engine = self._engine()
        if engine is None:
            self.status.config(text="No local TTS backend installed; web UI can use browser SpeechSynthesis")
            return
        try:
            if language: engine.setProperty("rate", 160)
            engine.say(text); engine.runAndWait(); self.status.config(text=f"Voice: {mode}")
        except Exception as exc:
            self.status.config(text=f"Voice error: {exc}")

    def pause_voice(self) -> None:
        if self._tts_engine is not None and hasattr(self._tts_engine, "stop"): self._tts_engine.stop()
        self.status.config(text="Voice paused/stopped by local backend")

    def resume_voice(self) -> None:
        self.status.config(text="Resume is supported by browser SpeechSynthesis; local pyttsx3 may require replay")

    def stop_voice(self) -> None:
        if self._tts_engine is not None:
            try: self._tts_engine.stop()
            except Exception: pass
        self.status.config(text="Voice stopped")


def main() -> None:
    ThamudicTranslatorApp().mainloop()


if __name__ == "__main__": main()
