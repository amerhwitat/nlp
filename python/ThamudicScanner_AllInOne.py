#!/usr/bin/env python3
"""All-in-one Ancient North Arabian / Thamudic scanner GUI.

The GUI follows the native run_thammudic-style Tkinter layout while adding image/PDF
import, local extraction/OCR, transliteration, evidence-backed translation, metadata,
history/PDF actions, and visible desktop voice controls.

No Tesseract or camel_tools dependency is required. Unsupported ancient readings are
reported as unavailable rather than fabricated.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from thamudic.ancient_translation import translate
from thamudic.ancient_alphabet_registry import supported_alphabet_languages
from thamudic.desktop_voice import DesktopVoice
from thamudic.media_pipeline import extract_media_text
from thamudic.pdf_export import write_report_pdf
from thamudic.script_summary import build_script_summary
from thamudic.source_language_scanner import scan_source_language
from thamudic.translation_log import read_records, verify_records

try:
    from thamudic.pdf_export import write_records_pdf
except ImportError:
    write_records_pdf = None

SUPPORTED_MEDIA = [
    ("Images / PDF / text", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.pdf *.txt *.md *.csv"),
    ("PDF", "*.pdf"),
    ("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"),
    ("Text", "*.txt *.md *.csv"),
    ("All files", "*.*"),
]
TARGETS = ("en", "ar")


def _set_text(widget, value: str) -> None:
    widget.delete("1.0", "end")
    widget.insert("1.0", value)


def _safe_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


class ThamudicScannerApp:
    """Native Tkinter all-in-one scanner with run_thammudic-style controls."""

    def __init__(self):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = tk.Tk()
        self.root.title("Thamudic / Ancient North Arabian — All-in-One Scanner")
        self.root.geometry("1180x900")
        self.root.minsize(960, 720)
        self._voice = DesktopVoice()
        self.media_path: Path | None = None
        self._last_result = None
        self._build()

    def _build(self):
        tk, ttk = self.tk, self.ttk
        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text="Thamudic / Ancient North Arabian", font=("TkDefaultFont", 18, "bold")).pack(anchor="w")
        ttk.Label(
            outer,
            text="Import an inscription image/PDF/text, inspect the script, transliterate it, and translate only when evidence is available.",
        ).pack(anchor="w", pady=(2, 10))

        media = ttk.LabelFrame(outer, text="Import / media")
        media.pack(fill="x", pady=(0, 8))
        ttk.Button(media, text="Import image / PDF", command=self.import_media).pack(side="left", padx=5, pady=6)
        ttk.Button(media, text="Import text", command=self.import_text).pack(side="left", padx=5, pady=6)
        ttk.Button(media, text="Process imported media", command=self.process_media).pack(side="left", padx=5, pady=6)
        self.media_status = ttk.Label(media, text="Media: ready")
        self.media_status.pack(side="left", padx=12)

        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(0, 8))
        ttk.Label(controls, text="Script").pack(side="left")
        scripts = list(supported_alphabet_languages())
        if "ancient-north-arabian" not in scripts:
            scripts.insert(0, "ancient-north-arabian")
        self.script = ttk.Combobox(controls, values=scripts, width=27, state="readonly")
        self.script.set("ancient-north-arabian")
        self.script.pack(side="left", padx=5)
        ttk.Label(controls, text="Target").pack(side="left", padx=(10, 4))
        self.target = ttk.Combobox(controls, values=TARGETS, width=8, state="readonly")
        self.target.set("en")
        self.target.pack(side="left")
        ttk.Button(controls, text="Scan", command=self.scan).pack(side="left", padx=5)
        ttk.Button(controls, text="Translate", command=self.translate).pack(side="left", padx=5)
        ttk.Button(controls, text="Script metadata", command=self.metadata).pack(side="left", padx=5)
        ttk.Button(controls, text="History PDF", command=self.history_pdf).pack(side="left", padx=5)
        ttk.Button(controls, text="Print history", command=self.print_history).pack(side="left", padx=5)

        source_frame = ttk.LabelFrame(outer, text="Original script / inscription / scholarly transliteration")
        source_frame.pack(fill="x", pady=(0, 8))
        self.source = tk.Text(source_frame, height=8, wrap="word")
        self.source.pack(fill="x", padx=6, pady=6)

        panes = ttk.Panedwindow(outer, orient="vertical")
        panes.pack(fill="both", expand=True)
        f1 = ttk.LabelFrame(panes, text="Transliteration / scan")
        f2 = ttk.LabelFrame(panes, text="Translation / evidence / report")
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
        self.voice_rate = tk.IntVar(value=160)
        ttk.Scale(voice, from_=60, to=300, variable=self.voice_rate, orient="horizontal", length=190).pack(side="left")
        self.voice_status = ttk.Label(voice, text="TTS: available" if self._voice.available() else "TTS: install pyttsx3")
        self.voice_status.pack(side="left", padx=12)

        self.status = ttk.Label(outer, text="Ready")
        self.status.pack(anchor="w", pady=(5, 0))

    def import_media(self):
        from tkinter import filedialog, messagebox
        path = filedialog.askopenfilename(title="Import inscription image / PDF", filetypes=SUPPORTED_MEDIA)
        if not path:
            return
        self.media_path = Path(path)
        try:
            self.process_media()
        except Exception as exc:
            messagebox.showerror("Import/OCR error", str(exc))

    def import_text(self):
        from tkinter import filedialog, messagebox
        path = filedialog.askopenfilename(title="Import text", filetypes=[("Text", "*.txt *.md *.csv"), ("All files", "*.*")])
        if not path:
            return
        self.media_path = Path(path)
        try:
            text, meta = extract_media_text(self.media_path)
            _set_text(self.source, text)
            self.media_status.config(text=f"Media: {self.media_path.name} · {meta.get('provider', 'text')}")
            self.translate()
        except Exception as exc:
            messagebox.showerror("Text import error", str(exc))

    def process_media(self):
        from tkinter import filedialog, messagebox
        if not self.media_path:
            path = filedialog.askopenfilename(title="Choose inscription image / PDF / text", filetypes=SUPPORTED_MEDIA)
            if not path:
                return
            self.media_path = Path(path)
        text, meta = extract_media_text(self.media_path)
        _set_text(self.source, text)
        self.media_status.config(text=f"Media: {self.media_path.name} · {meta.get('provider', 'unknown')}")
        self.status.config(text="Media imported; processing…")
        self.translate()

    def scan(self):
        text = self.source.get("1.0", "end-1c")
        if not text.strip():
            self.status.config(text="Enter or import source text first")
            return
        language = self.script.get()
        try:
            result = scan_source_language(text, language=language)
            _set_text(self.trans, _safe_json(result))
            self.status.config(text=f"Matched characters: {result['matched_character_count']}")
        except Exception as exc:
            self.status.config(text=f"Scan error: {exc}")

    def translate(self):
        text = self.source.get("1.0", "end-1c").strip()
        if not text:
            self.status.config(text="Enter or import source text first")
            return
        try:
            result = translate(text, script=self.script.get(), target_language=self.target.get())
            self._last_result = result
            _set_text(self.trans, result.get("transliteration", ""))
            evidence = result.get("translation") or "Translation unavailable for this reading/provider."
            report = dict(result)
            report["display_translation"] = evidence
            _set_text(self.out, _safe_json(report))
            self.status.config(text=f"{result.get('translation_status')} · confidence={result.get('confidence')}")
        except Exception as exc:
            self.status.config(text=f"Translation error: {exc}")

    def metadata(self):
        try:
            _set_text(self.out, _safe_json(build_script_summary(self.script.get())))
            self.status.config(text="Script metadata loaded")
        except Exception as exc:
            self.status.config(text=f"Metadata error: {exc}")

    def history_pdf(self):
        from tkinter import filedialog, messagebox
        out = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF", "*.pdf")], initialfile="translation-history.pdf")
        if not out:
            return
        try:
            records = read_records()
            if write_records_pdf:
                write_records_pdf(records, output=out)
            else:
                raise RuntimeError("PDF record exporter is unavailable")
            self.status.config(text=f"PDF written: {out}")
        except Exception as exc:
            messagebox.showerror("PDF export", str(exc))

    def print_history(self):
        from tkinter import filedialog, messagebox
        out = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF", "*.pdf")], initialfile="translation-history-print.pdf")
        if not out:
            return
        try:
            records = read_records()
            if write_records_pdf:
                write_records_pdf(records, output=out)
            else:
                raise RuntimeError("PDF record exporter is unavailable")
            if sys.platform.startswith("win"):
                os.startfile(str(out), "print")
            else:
                subprocess.run(["lpr", str(out)], check=False)
            self.status.config(text=f"Print submitted: {out}")
        except Exception as exc:
            messagebox.showerror("Print", str(exc))

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
        language = self.target.get() or "en"
        self._read(self.out.get("1.0", "end-1c"), language, "translation")

    def stop_voice(self):
        self._voice.stop()
        self.status.config(text="Voice stopped")

    def run(self):
        self.root.mainloop()


def main(argv=None):
    parser = argparse.ArgumentParser(description="All-in-one Thamudic / Ancient Script scanner")
    parser.add_argument("file", nargs="?", help="image/PDF/text file to import")
    parser.add_argument("--gui", action="store_true", help="start Tkinter GUI")
    parser.add_argument("--text", default="", help="text to scan/translate")
    parser.add_argument("--script", default="ancient-north-arabian")
    parser.add_argument("--target", default="en", choices=list(TARGETS))
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--translate", action="store_true")
    args = parser.parse_args(argv)
    if args.file and not args.gui:
        text, _ = extract_media_text(args.file)
        if args.translate:
            print(_safe_json(translate(text, args.script, args.target)))
        else:
            print(text)
        return 0
    if args.text and args.scan:
        print(_safe_json(scan_source_language(args.text, args.script)))
        return 0
    if args.text and args.translate:
        print(_safe_json(translate(args.text, args.script, args.target)))
        return 0
    app = ThamudicScannerApp()
    if args.file:
        app.media_path = Path(args.file)
        app.process_media()
    elif args.text:
        _set_text(app.source, args.text)
        app.translate()
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
