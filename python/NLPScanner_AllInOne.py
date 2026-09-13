#!/usr/bin/env python3
"""Standalone general NLP/media scanner with the native Thamudic GUI look and feel.

Supports image/PDF/text import, local extraction/OCR, ancient-script scanning,
transliteration, evidence-backed translation, and visible desktop voice controls.
Unsupported OCR or translation is reported rather than fabricated.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from thamudic.ancient_translation import translate
from thamudic.desktop_voice import DesktopVoice
from thamudic.media_pipeline import extract_media_text, scan_translate_media
from thamudic.source_language_scanner import scan_source_language
from thamudic.ancient_alphabet_registry import supported_alphabet_languages
from thamudic.script_summary import build_script_summary

SUPPORTED_MEDIA = [
    ("Images / PDF / text", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.pdf *.txt *.md *.csv"),
    ("PDF", "*.pdf"),
    ("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"),
    ("Text", "*.txt *.md *.csv"),
    ("All files", "*.*"),
]


def _set(widget, text):
    widget.delete("1.0", "end")
    widget.insert("1.0", text)


class NLPScannerApp(tk.Tk):
    """General NLP scanner using the same toolbar/pane/voice layout as Thamudic."""

    def __init__(self):
        super().__init__()
        self.title("NLP Scanner — Media → OCR → Scan → Transliterate → Translate → Voice")
        self.geometry("1180x900")
        self.minsize(960, 720)
        self._voice = DesktopVoice()
        self.path: Path | None = None
        self._build()

    def _build(self):
        outer = ttk.Frame(self, padding=16)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="NLP / Ancient Language Scanner", font=("TkDefaultFont", 18, "bold")).pack(anchor="w")
        ttk.Label(outer, text="Import documents or inscriptions, inspect source scripts, transliterate, translate, and hear the resulting text.").pack(anchor="w", pady=(2, 10))

        media = ttk.LabelFrame(outer, text="Import / media")
        media.pack(fill="x", pady=(0, 8))
        ttk.Button(media, text="Import image / PDF", command=self.import_file).pack(side="left", padx=5, pady=6)
        ttk.Button(media, text="Import text", command=self.import_text).pack(side="left", padx=5, pady=6)
        ttk.Button(media, text="Process imported media", command=self.process).pack(side="left", padx=5, pady=6)
        self.media_status = ttk.Label(media, text="Media: ready")
        self.media_status.pack(side="left", padx=12)

        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(0, 8))
        ttk.Label(controls, text="Script").pack(side="left")
        scripts = ["Dadanitic", "Safaitic", "Hismaic", "Taymanitic", "Thamudic B"]
        self.script = ttk.Combobox(controls, values=scripts, state="readonly", width=20)
        self.script.set("Dadanitic")
        self.script.pack(side="left", padx=5)
        ttk.Label(controls, text="Target").pack(side="left", padx=(10, 4))
        self.target = ttk.Combobox(controls, values=["en", "ar"], state="readonly", width=8)
        self.target.set("en")
        self.target.pack(side="left")
        ttk.Button(controls, text="Scan", command=self.scan).pack(side="left", padx=5)
        ttk.Button(controls, text="Scan + translate", command=self.process).pack(side="left", padx=5)
        ttk.Button(controls, text="Script metadata", command=self.metadata).pack(side="left", padx=5)

        source_frame = ttk.LabelFrame(outer, text="Extracted source / OCR text")
        source_frame.pack(fill="x", pady=(0, 8))
        self.source = tk.Text(source_frame, height=8, wrap="word")
        self.source.pack(fill="x", padx=6, pady=6)

        panes = ttk.Panedwindow(outer, orient="vertical")
        panes.pack(fill="both", expand=True)
        f1 = ttk.LabelFrame(panes, text="Transliteration / script scan")
        f2 = ttk.LabelFrame(panes, text="Translation / evidence")
        panes.add(f1, weight=1)
        panes.add(f2, weight=2)
        self.trans = tk.Text(f1, height=7, wrap="word")
        self.trans.pack(fill="both", expand=True, padx=6, pady=6)
        self.out = tk.Text(f2, wrap="word")
        self.out.pack(fill="both", expand=True, padx=6, pady=6)

        voice = ttk.LabelFrame(outer, text="Voice controls")
        voice.pack(fill="x", pady=(8, 4))
        ttk.Button(voice, text="▶ Read transliteration", command=self.speak_transliteration).pack(side="left", padx=4, pady=7)
        ttk.Button(voice, text="▶ Read translation", command=self.speak_translation).pack(side="left", padx=4, pady=7)
        ttk.Button(voice, text="■ Stop", command=self.stop_voice).pack(side="left", padx=4, pady=7)
        ttk.Label(voice, text="Rate").pack(side="left", padx=(18, 4))
        self.rate = tk.IntVar(value=160)
        ttk.Scale(voice, from_=60, to=300, variable=self.rate, orient="horizontal", length=190).pack(side="left")
        ttk.Label(voice, text="TTS: available" if self._voice.available() else "TTS: install pyttsx3").pack(side="left", padx=12)

        self.status = ttk.Label(outer, text="Ready")
        self.status.pack(anchor="w", pady=(5, 0))

    def import_file(self):
        path = filedialog.askopenfilename(title="Import image / PDF / text", filetypes=SUPPORTED_MEDIA)
        if not path:
            return
        self.path = Path(path)
        self.process()

    def import_text(self):
        path = filedialog.askopenfilename(title="Import text", filetypes=[("Text", "*.txt *.md *.csv"), ("All files", "*.*")])
        if not path:
            return
        self.path = Path(path)
        self.process()

    def process(self):
        try:
            if self.path:
                text, meta = extract_media_text(self.path)
                _set(self.source, text)
                self.media_status.config(text=f"Media: {self.path.name} · {meta.get('provider', 'unknown')}")
            text = self.source.get("1.0", "end-1c").strip()
            if not text:
                self.status.config(text="Import media or enter text first")
                return
            scan = scan_source_language(text, language="ancient-north-arabian")
            result = translate(text, self.script.get(), self.target.get())
            _set(self.trans, result.get("transliteration", ""))
            evidence = result.get("translation") or "No corpus-backed translation is available for this reading."
            _set(self.out, evidence + "\n\n" + str({"status": result.get("translation_status"), "confidence": result.get("confidence"), "provenance": result.get("provenance")}))
            self.status.config(text=f"{result.get('translation_status')} · {scan['matched_character_count']} script characters")
        except Exception as exc:
            messagebox.showerror("Processing error", str(exc))
            self.status.config(text="Processing failed")

    def scan(self):
        text = self.source.get("1.0", "end-1c")
        if not text.strip():
            self.status.config(text="Import media or enter text first")
            return
        try:
            result = scan_source_language(text, language="ancient-north-arabian")
            _set(self.trans, str(result))
            self.status.config(text=f"Matched characters: {result['matched_character_count']}")
        except Exception as exc:
            self.status.config(text=f"Scan error: {exc}")

    def metadata(self):
        try:
            _set(self.out, str(build_script_summary("ancient-north-arabian")))
            self.status.config(text="Script metadata loaded")
        except Exception as exc:
            self.status.config(text=f"Metadata error: {exc}")

    def _speak(self, text, language, label):
        text = text.strip()
        if not text:
            self.status.config(text=f"No {label} text to read")
            return
        if not self._voice.available():
            self.status.config(text="Install pyttsx3 for desktop voice playback")
            return
        try:
            self._voice.speak(text, language=language, rate=int(self.rate.get()))
            self.status.config(text=f"Voice: {label}")
        except Exception as exc:
            self.status.config(text=f"Voice error: {exc}")

    def speak_transliteration(self):
        self._speak(self.trans.get("1.0", "end-1c"), "en", "transliteration")

    def speak_translation(self):
        self._speak(self.out.get("1.0", "end-1c"), self.target.get(), "translation")

    def stop_voice(self):
        self._voice.stop()
        self.status.config(text="Voice stopped")


def main():
    parser = argparse.ArgumentParser(description="All-in-one NLP/media scanner")
    parser.add_argument("file", nargs="?", help="image/PDF/text file")
    parser.add_argument("--script", default="Dadanitic")
    parser.add_argument("--target", default="en", choices=["en", "ar"])
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    if args.file and not args.gui:
        result = scan_translate_media(args.file, args.script, args.target)
        print(result.get("transliteration", ""))
        print(result.get("translation") or "TRANSLATION_UNAVAILABLE")
        return 0
    app = NLPScannerApp()
    app.script.set(args.script)
    app.target.set(args.target)
    if args.file:
        app.path = Path(args.file)
        app.process()
    app.mainloop()


if __name__ == "__main__":
    main()
