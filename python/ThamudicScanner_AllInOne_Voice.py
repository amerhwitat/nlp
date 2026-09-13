#!/usr/bin/env python3
"""Unified Thamudic all-in-one desktop scanner: media import, OCR, transliteration,
corpus-backed translation, voice playback, and the existing history/PDF features.

Workflow: image/PDF/text -> extraction/OCR -> script scan -> transliteration ->
corpus-backed translation -> GUI output -> spoken transliteration/translation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ThamudicScanner_AllInOne import ThamudicScannerApp as BaseScannerApp, translate
from thamudic.desktop_voice import DesktopVoice
from thamudic.media_pipeline import extract_media_text

SUPPORTED_MEDIA = [
    ("Images / PDF / text", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.pdf *.txt *.md *.csv"),
    ("All files", "*.*"),
]


class UnifiedThamudicScannerApp(BaseScannerApp):
    """Base all-in-one scanner plus media import and desktop voice controls."""

    def __init__(self):
        super().__init__()
        self._voice = DesktopVoice()
        self.media_path: Path | None = None
        self._add_media_controls()
        self._add_voice_controls()

    def _add_media_controls(self):
        frame = ttk.Frame(self.root, padding=(10, 4))
        frame.pack(fill="x")
        ttk.Button(frame, text="Import image/PDF/text", command=self.import_media).pack(side="left", padx=4)
        ttk.Button(frame, text="Process imported media", command=self.process_media).pack(side="left", padx=4)
        self.media_status = ttk.Label(frame, text="Media: ready")
        self.media_status.pack(side="left", padx=10)

    def import_media(self):
        path = filedialog.askopenfilename(title="Import inscription image/PDF/text", filetypes=SUPPORTED_MEDIA)
        if not path:
            return
        self.media_path = Path(path)
        self.process_media()

    def process_media(self):
        if not self.media_path:
            path = filedialog.askopenfilename(title="Choose inscription image/PDF/text", filetypes=SUPPORTED_MEDIA)
            if not path:
                return
            self.media_path = Path(path)
        try:
            text, meta = extract_media_text(self.media_path)
            source_widget = getattr(self, "source", None)
            if source_widget is not None:
                source_widget.delete("1.0", "end")
                source_widget.insert("1.0", text)
            result = translate(text, script=self.script.get(), target_language=self.target.get())
            trans_widget = getattr(self, "trans", None)
            if trans_widget is not None:
                trans_widget.delete("1.0", "end")
                trans_widget.insert("1.0", result.get("transliteration", ""))
            out_widget = getattr(self, "out", None)
            if out_widget is not None:
                out_widget.delete("1.0", "end")
                out_widget.insert("1.0", result.get("translation") or "Translation unavailable for this reading/provider.")
            self.media_status.config(text=f"Media: {self.media_path.name} · {meta.get('provider', 'unknown')}")
            self.status.config(text=f"{result.get('translation_status', 'processed')} · imported and translated")
        except Exception as exc:
            messagebox.showerror("Import/OCR error", str(exc))
            self.media_status.config(text="Media: processing failed")

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
        self._read(self.out.get("1.0", "end-1c"), self.target.get(), "translation")

    def stop_voice(self):
        self._voice.stop()
        self.status.config(text="Voice stopped")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Unified all-in-one Thamudic media scanner with voice")
    parser.add_argument("file", nargs="?", help="image/PDF/text file to import")
    parser.add_argument("--script", default="Dadanitic")
    parser.add_argument("--target", default="en")
    parser.add_argument("--gui", action="store_true", help="start GUI")
    args = parser.parse_args(argv)

    app = UnifiedThamudicScannerApp()
    if args.file:
        app.media_path = Path(args.file)
        app.process_media()
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
