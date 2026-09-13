#!/usr/bin/env python3
"""Standalone general NLP/media scanner with GUI and desktop voice controls.

Imports TXT/MD/CSV/PDF/images, extracts text locally where possible, optionally OCRs
images/scanned PDFs with EasyOCR, detects ancient scripts, transliterates Ancient
North Arabian text, performs evidence-backed corpus translation, and reads the
resulting transliteration/translation aloud through the operating-system TTS engine.
It deliberately reports unsupported translation/OCR instead of fabricating results.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from thamudic.media_pipeline import extract_media_text, scan_translate_media
from thamudic.source_language_scanner import scan_source_language
from thamudic.ancient_translation import translate
from thamudic.desktop_voice import DesktopVoice


class NLPScannerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NLP Scanner — Media → OCR → Scan → Transliterate → Translate → Voice")
        self.geometry("1200x900")
        self._voice = DesktopVoice()
        self._build()

    def _build(self):
        bar = ttk.Frame(self); bar.pack(fill="x", padx=10, pady=8)
        ttk.Button(bar, text="Import image/PDF/text", command=self.import_file).pack(side="left")
        ttk.Label(bar, text="Script").pack(side="left", padx=(16, 4))
        self.script = ttk.Combobox(bar, values=["Dadanitic", "Safaitic", "Hismaic", "Taymanitic", "Thamudic B"], state="readonly", width=16)
        self.script.set("Dadanitic"); self.script.pack(side="left")
        ttk.Label(bar, text="Target").pack(side="left", padx=(12, 4))
        self.target = ttk.Combobox(bar, values=["en", "ar"], state="readonly", width=8); self.target.set("en"); self.target.pack(side="left")
        ttk.Button(bar, text="Scan + translate", command=self.process).pack(side="left", padx=12)
        self.status = ttk.Label(bar, text="Ready"); self.status.pack(side="left", padx=10)

        ttk.Label(self, text="Extracted source / OCR text").pack(anchor="w", padx=10)
        self.source = tk.Text(self, height=12, wrap="word"); self.source.pack(fill="x", padx=10)
        panes = ttk.Panedwindow(self, orient="vertical"); panes.pack(fill="both", expand=True, padx=10, pady=8)
        f1 = ttk.LabelFrame(panes, text="Transliteration / script scan"); f2 = ttk.LabelFrame(panes, text="Translation / evidence")
        panes.add(f1, weight=1); panes.add(f2, weight=1)
        self.trans = tk.Text(f1, wrap="word"); self.trans.pack(fill="both", expand=True)
        self.out = tk.Text(f2, wrap="word"); self.out.pack(fill="both", expand=True)

        voice = ttk.LabelFrame(self, text="Voice controls")
        voice.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(voice, text="▶ Read transliteration", command=self.speak_transliteration).pack(side="left", padx=4, pady=6)
        ttk.Button(voice, text="▶ Read translation", command=self.speak_translation).pack(side="left", padx=4, pady=6)
        ttk.Button(voice, text="■ Stop", command=self.stop_voice).pack(side="left", padx=4, pady=6)
        ttk.Label(voice, text="Rate").pack(side="left", padx=(18, 4))
        self.rate = tk.IntVar(value=160)
        ttk.Scale(voice, from_=60, to=300, variable=self.rate, orient="horizontal", length=180).pack(side="left")
        self.voice_status = ttk.Label(voice, text="TTS: checking…"); self.voice_status.pack(side="left", padx=10)
        self._update_voice_status()
        self.path = None

    def _update_voice_status(self):
        self.voice_status.config(text="TTS: available" if self._voice.available() else "TTS: install pyttsx3")

    def import_file(self):
        p = filedialog.askopenfilename(filetypes=[("Supported media", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.pdf *.txt *.md *.csv"), ("All files", "*.*")])
        if not p: return
        self.path = Path(p)
        try:
            text, meta = extract_media_text(self.path)
            self.source.delete("1.0", "end"); self.source.insert("1.0", text)
            self.status.config(text=f"Imported: {self.path.name} · {meta.get('provider', 'unknown')}")
            self.process()
        except Exception as exc:
            messagebox.showerror("Import/OCR error", str(exc)); self.status.config(text="Import failed")

    def process(self):
        text = self.source.get("1.0", "end-1c").strip()
        if not text:
            self.status.config(text="Import media or enter text first"); return
        try:
            scan = scan_source_language(text, language="ancient-north-arabian")
            result = translate(text, self.script.get(), self.target.get())
            self.trans.delete("1.0", "end"); self.trans.insert("1.0", result["transliteration"])
            self.out.delete("1.0", "end")
            self.out.insert("1.0", result.get("translation") or "No corpus-backed translation is available for this reading.\n\n" + str({"status": result["translation_status"], "confidence": result["confidence"], "provenance": result["provenance"]}))
            self.status.config(text=f"{result['translation_status']} · {scan['matched_character_count']} script characters")
        except Exception as exc:
            messagebox.showerror("Processing error", str(exc))

    def _speak(self, text, language, label):
        text = text.strip()
        if not text:
            self.status.config(text=f"No {label} text to read"); return
        if not self._voice.available():
            self.status.config(text="Install pyttsx3 for desktop voice playback"); return
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
        self._voice.stop(); self.status.config(text="Voice stopped")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="?", help="image/PDF/text file to process")
    parser.add_argument("--script", default="Dadanitic"); parser.add_argument("--target", default="en")
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    if args.file and not args.gui:
        result = scan_translate_media(args.file, args.script, args.target)
        print(result["transliteration"]); print(result.get("translation") or "TRANSLATION_UNAVAILABLE")
        return
    app = NLPScannerApp()
    if args.file:
        app.path = Path(args.file)
        try:
            text, _ = extract_media_text(args.file); app.source.insert("1.0", text); app.process()
        except Exception as exc: messagebox.showerror("Import/OCR error", str(exc))
    app.mainloop()

if __name__ == "__main__": main()
