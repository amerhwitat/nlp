#!/usr/bin/env python3
"""Voice-enabled launcher for the all-in-one Thamudic scanner.

This keeps ``ThamudicScanner_AllInOne.py`` as the consolidated scanner implementation
and adds desktop speech controls for transliteration and translated output without
altering the scholarly/evidence-aware translation layer.
"""
from __future__ import annotations

import argparse
import tkinter as tk
from tkinter import ttk, messagebox

from ThamudicScanner_AllInOne import ThamudicScannerApp as BaseScannerApp, translate
from thamudic.desktop_voice import DesktopVoice


class VoiceThamudicScannerApp(BaseScannerApp):
    def __init__(self):
        super().__init__()
        self._voice = DesktopVoice()
        self._add_voice_controls()

    def _add_voice_controls(self):
        ttk.Separator(self.root, orient="horizontal").pack(fill="x", padx=10, pady=4)
        frame = ttk.Frame(self.root, padding=(10, 4))
        frame.pack(fill="x")
        ttk.Label(frame, text="Voice controls").pack(side="left", padx=(0, 8))
        ttk.Button(frame, text="▶ Read transliteration", command=self.speak_transliteration).pack(side="left", padx=4)
        ttk.Button(frame, text="▶ Read translation", command=self.speak_translation).pack(side="left", padx=4)
        ttk.Button(frame, text="■ Stop", command=self.stop_voice).pack(side="left", padx=4)
        ttk.Label(frame, text="Rate").pack(side="left", padx=(16, 4))
        self.voice_rate = tk.IntVar(value=160)
        ttk.Scale(frame, from_=60, to=300, variable=self.voice_rate, orient="horizontal", length=170).pack(side="left")
        self.voice_status = ttk.Label(frame, text="TTS: available" if self._voice.available() else "TTS: install pyttsx3")
        self.voice_status.pack(side="left", padx=10)

    def _read(self, text: str, language: str, label: str):
        text = text.strip()
        if not text:
            self.status.config(text=f"No {label} text to read")
            return
        if not self._voice.available():
            self.status.config(text="Install pyttsx3 for desktop voice playback")
            return
        try:
            self._voice.speak(text, language=language, rate=int(self.voice_rate.get()))
            self.status.config(text=f"Voice: {label}")
        except Exception as exc:
            self.status.config(text=f"Voice error: {exc}")

    def speak_transliteration(self):
        self._read(self.trans.get("1.0", "end-1c"), "en", "transliteration")

    def speak_translation(self):
        # The base all-in-one GUI stores its JSON result in `out`. Extract the
        # translation field when possible so metadata is not read aloud.
        raw = self.out.get("1.0", "end-1c").strip()
        text = raw
        try:
            import json
            data = json.loads(raw)
            text = data.get("translation") or raw
        except Exception:
            pass
        self._read(text, self.target.get(), "translation")

    def stop_voice(self):
        self._voice.stop()
        self.status.config(text="Voice stopped")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Voice-enabled all-in-one Thamudic scanner")
    parser.add_argument("--gui", action="store_true", help="start the voice-enabled GUI")
    args = parser.parse_args(argv)
    if not args.gui:
        args.gui = True
    VoiceThamudicScannerApp().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
