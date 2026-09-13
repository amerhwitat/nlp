#!/usr/bin/env python3
"""Unified Thamudic + NLP scanner workbench.

This is the canonical all-in-one desktop/CLI entry point.  It combines the
existing ancient-script translation, source-language scanning, media import,
resilient OCR, voice playback, history/PDF export, and JSON CLI workflows.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

# Keep native OCR bounded unless the user explicitly overrides it.
os.environ.setdefault("THAMUDIC_OCR_TIMEOUT", "45")

from thamudic.ancient_translation import supported_targets, translate
from thamudic.pdf_export import write_records_pdf, write_report_pdf
from thamudic.resilient_ocr import DEFAULT_TIMEOUT, ocr_image
from thamudic.source_language_scanner import scan_source_language

SCRIPT_CHOICES = ("Dadanitic", "Safaitic", "Hismaic", "Taymanitic", "Thamudic B", "Old North Arabian / Thamudic")


def _worker() -> Path:
    return PYTHON / "thamudic" / "ocr_worker.py"


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def scan_text(text: str, script: str) -> dict[str, Any]:
    return scan_source_language(text, language="ancient-north-arabian") | {
        "script_profile": script,
        "scanner": "unicode-aware source-language scanner",
    }


def translate_text(text: str, script: str, target: str) -> dict[str, Any]:
    result = translate(text, script=script, target_language=target)
    result["source_scan"] = scan_text(text, script)
    result["workflow"] = "source -> scan -> scholarly transliteration -> evidence-backed translation"
    return result


def process_image(path: str | Path, script: str, target: str) -> dict[str, Any]:
    source = Path(path).expanduser()
    result = ocr_image(source, _worker(), languages=["en", "ar"], timeout=DEFAULT_TIMEOUT)
    text = str(result.get("text", "")) if result.get("ok") else ""
    translation = translate_text(text, script, target) if text else {
        "source_text": "", "script": script, "transliteration": "", "translation": None,
        "translation_status": "not_available", "confidence": "unknown", "corpus_id": None,
        "provenance": None, "source_scan": scan_text("", script),
    }
    translation["media"] = {
        "provider": result.get("provider", "easyocr-subprocess"),
        "media_type": "image",
        "source_file": str(source),
        "ocr_available": bool(result.get("ok") and text),
        "ocr_error": "" if result.get("ok") else f"{result.get('error_type', 'Error')}: {result.get('error', 'OCR unavailable')}",
        "ocr_timed_out": bool(result.get("timed_out")),
        "ocr_recoverable": True,
        "ocr_confidence": result.get("ocr_confidence", 0.0),
        "detections": result.get("detections", 0),
    }
    translation["recognition_status"] = (
        "easyocr_text_plus_source_scan" if text else "glyph/source scan after recoverable OCR failure"
    )
    return translation


def process_media(path: str | Path, script: str, target: str) -> dict[str, Any]:
    source = Path(path).expanduser()
    if source.suffix.casefold() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
        return process_image(source, script, target)
    from thamudic.media_pipeline import extract_media_text
    text, media = extract_media_text(source, ["en", "ar"])
    result = translate_text(text, script, target) if text else {
        "source_text": "", "script": script, "transliteration": "", "translation": None,
        "translation_status": "not_available", "confidence": "unknown", "corpus_id": None,
        "provenance": None, "source_scan": scan_text("", script),
    }
    result["media"] = media | {"source_file": str(source)}
    result["recognition_status"] = "media_text_plus_source_scan" if text else "media_import_without_text"
    return result


def append_history(record: dict[str, Any], history_path: Path) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if history_path.exists():
        try:
            rows = json.loads(history_path.read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                rows = []
        except (OSError, json.JSONDecodeError):
            rows = []
    rows.append({"timestamp": datetime.now(timezone.utc).isoformat(), **record})
    history_path.write_text(_safe_json(rows), encoding="utf-8")


def print_pdf(path: Path) -> bool:
    try:
        if platform.system() == "Windows":
            os.startfile(str(path), "print")  # type: ignore[attr-defined]
        elif platform.system() == "Darwin":
            subprocess.run(["lp", str(path)], check=False)
        else:
            subprocess.run(["lp", str(path)], check=False)
        return True
    except Exception:
        return False


class UnifiedApp:
    """Tkinter GUI exposing the combined NLP and Thamudic workflow."""

    def __init__(self, db_path: str = "ancient_objects.sqlite") -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = tk.Tk()
        self.root.title("Thamudic + NLP Scanner — All In One")
        self.root.geometry("1200x900")
        self.db_path = db_path
        self.history_path = Path(db_path).with_suffix(".history.json")
        self.last_result: dict[str, Any] = {}
        self.voice = None
        try:
            from thamudic.desktop_voice import DesktopVoice
            self.voice = DesktopVoice()
        except Exception:
            self.voice = None
        self._build()

    def _build(self) -> None:
        tk, ttk = self.tk, self.ttk
        top = ttk.Frame(self.root); top.pack(fill="x", padx=12, pady=10)
        ttk.Label(top, text="Mode").pack(side="left")
        self.mode = ttk.Combobox(top, values=["NLP", "Thamudic"], state="readonly", width=12)
        self.mode.set("NLP"); self.mode.pack(side="left", padx=6)
        ttk.Label(top, text="Script").pack(side="left", padx=(12, 4))
        self.script = ttk.Combobox(top, values=SCRIPT_CHOICES, state="readonly", width=28)
        self.script.set("Dadanitic"); self.script.pack(side="left")
        ttk.Label(top, text="Target").pack(side="left", padx=(12, 4))
        self.target = ttk.Combobox(top, values=list(supported_targets()), state="readonly", width=8)
        self.target.set("en"); self.target.pack(side="left")
        ttk.Button(top, text="Import / Scan", command=self.import_media).pack(side="left", padx=8)
        ttk.Button(top, text="Translate + Transliterate", command=self.translate_current).pack(side="left", padx=4)
        ttk.Button(top, text="Scan Source", command=self.scan_current).pack(side="left", padx=4)

        self.notebook = ttk.Notebook(self.root); self.notebook.pack(fill="both", expand=True, padx=12, pady=4)
        self.work_tab = ttk.Frame(self.notebook); self.result_tab = ttk.Frame(self.notebook); self.history_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.work_tab, text="Scanner / NLP")
        self.notebook.add(self.result_tab, text="Analysis / Evidence")
        self.notebook.add(self.history_tab, text="History / Reports")

        ttk.Label(self.work_tab, text="Imported OCR source / scholarly text").pack(anchor="w", padx=8, pady=(8, 2))
        self.source = tk.Text(self.work_tab, height=14, wrap="word")
        self.source.pack(fill="both", expand=True, padx=8)
        controls = ttk.Frame(self.work_tab); controls.pack(fill="x", padx=8, pady=8)
        ttk.Button(controls, text="▶ Transliteration", command=lambda: self.speak(self.translit.get("1.0", "end-1c"))).pack(side="left")
        ttk.Button(controls, text="▶ Translation", command=lambda: self.speak(self.translation.get("1.0", "end-1c"))).pack(side="left", padx=4)
        ttk.Button(controls, text="■ Stop", command=self.stop_voice).pack(side="left")
        self.status = ttk.Label(controls, text="Ready"); self.status.pack(side="right")

        panes = ttk.Panedwindow(self.result_tab, orient="vertical"); panes.pack(fill="both", expand=True, padx=8, pady=8)
        f1, f2, f3 = ttk.LabelFrame(panes, text="Transliteration"), ttk.LabelFrame(panes, text="Translation / Evidence"), ttk.LabelFrame(panes, text="Structured analysis (JSON)")
        panes.add(f1, weight=1); panes.add(f2, weight=1); panes.add(f3, weight=2)
        self.translit = tk.Text(f1, height=6, wrap="word"); self.translit.pack(fill="both", expand=True)
        self.translation = tk.Text(f2, height=8, wrap="word"); self.translation.pack(fill="both", expand=True)
        self.analysis = tk.Text(f3, wrap="none"); self.analysis.pack(fill="both", expand=True)

        htop = ttk.Frame(self.history_tab); htop.pack(fill="x", padx=8, pady=8)
        ttk.Button(htop, text="Export History PDF", command=self.export_history).pack(side="left")
        ttk.Button(htop, text="Print History PDF", command=self.print_history).pack(side="left", padx=6)
        ttk.Button(htop, text="Refresh", command=self.refresh_history).pack(side="left")
        self.history = tk.Text(self.history_tab, wrap="none"); self.history.pack(fill="both", expand=True, padx=8, pady=4)
        self.refresh_history()

    def _set_source(self, text: str) -> None:
        self.source.delete("1.0", "end"); self.source.insert("1.0", text)

    def import_media(self) -> None:
        from tkinter import filedialog, messagebox
        p = filedialog.askopenfilename(filetypes=[("Images/PDF/Text", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.pdf *.txt *.md *.csv"), ("All files", "*.*")])
        if not p: return
        try:
            result = process_media(p, self.script.get(), self.target.get())
            self._set_source(result.get("source_text", ""))
            self.show_result(result)
            self.save_result(result)
        except Exception as exc:
            messagebox.showerror("Import error", f"{type(exc).__name__}: {exc}")
            self.status.config(text="Import failed safely")

    def translate_current(self) -> None:
        text = self.source.get("1.0", "end-1c").strip()
        if not text:
            self.status.config(text="Enter or import source text first"); return
        result = translate_text(text, self.script.get(), self.target.get())
        self.show_result(result); self.save_result(result)

    def scan_current(self) -> None:
        text = self.source.get("1.0", "end-1c").strip()
        result = scan_text(text, self.script.get())
        self.analysis.delete("1.0", "end"); self.analysis.insert("1.0", _safe_json(result))
        self.notebook.select(self.result_tab); self.status.config(text=f"Scanned {result.get('matched_character_count', 0)} matching characters")

    def show_result(self, result: dict[str, Any]) -> None:
        self.last_result = result
        self.translit.delete("1.0", "end"); self.translit.insert("1.0", result.get("transliteration", ""))
        translation = result.get("translation")
        evidence = translation or "No corpus-backed translation is available for this reading.\n"
        evidence += f"\nStatus: {result.get('translation_status', 'not_available')}\nConfidence: {result.get('confidence', 'unknown')}\nCorpus: {result.get('corpus_id') or 'none'}\nProvenance: {result.get('provenance') or 'none'}"
        self.translation.delete("1.0", "end"); self.translation.insert("1.0", evidence)
        self.analysis.delete("1.0", "end"); self.analysis.insert("1.0", _safe_json(result))
        media = result.get("media", {})
        if media.get("ocr_timed_out"):
            self.status.config(text=f"OCR timed out after {DEFAULT_TIMEOUT}s; recovered and kept GUI running")
        else:
            self.status.config(text=f"{result.get('translation_status', 'analysis')} · mode={self.mode.get()}")
        self.notebook.select(self.result_tab)

    def save_result(self, result: dict[str, Any]) -> None:
        append_history({"mode": self.mode.get(), "script": self.script.get(), "target": self.target.get(), "result": result}, self.history_path)
        self.refresh_history()

    def refresh_history(self) -> None:
        self.history.delete("1.0", "end")
        if self.history_path.exists():
            self.history.insert("1.0", self.history_path.read_text(encoding="utf-8"))
        else:
            self.history.insert("1.0", "No history records yet.")

    def export_history(self) -> None:
        from tkinter import filedialog, messagebox
        try:
            records = json.loads(self.history_path.read_text(encoding="utf-8")) if self.history_path.exists() else []
            out = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF", "*.pdf")], initialfile="thamudic_nlp_history.pdf")
            if not out: return
            write_records_pdf(records, out, "Thamudic + NLP Translation / Scan History")
            self.status.config(text=f"PDF exported: {Path(out).name}")
        except Exception as exc:
            messagebox.showerror("PDF export error", str(exc))

    def print_history(self) -> None:
        from tkinter import filedialog, messagebox
        try:
            records = json.loads(self.history_path.read_text(encoding="utf-8")) if self.history_path.exists() else []
            out = self.history_path.with_name("thamudic_nlp_history_print.pdf")
            write_records_pdf(records, out, "Thamudic + NLP Translation / Scan History")
            if print_pdf(out): self.status.config(text="Print job sent")
            else: messagebox.showinfo("Print", f"PDF created at {out}; automatic printing is unavailable on this system.")
        except Exception as exc:
            messagebox.showerror("Print error", str(exc))

    def speak(self, text: str) -> None:
        if not text.strip() or not self.voice: return
        try: self.voice.speak(text, language="en", rate=160)
        except Exception as exc: self.status.config(text=f"Voice error: {exc}")

    def stop_voice(self) -> None:
        if self.voice:
            try: self.voice.stop()
            except Exception: pass
        self.status.config(text="Voice stopped")

    def mainloop(self) -> None:
        self.root.mainloop()


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="All-in-one resilient Thamudic + NLP scanner")
    parser.add_argument("--mode", choices=("nlp", "thamudic"), default="nlp")
    parser.add_argument("--db", default="ancient_objects.sqlite")
    parser.add_argument("--file")
    parser.add_argument("--text", default="")
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--translate", action="store_true")
    parser.add_argument("--script", default="Dadanitic", choices=SCRIPT_CHOICES)
    parser.add_argument("--target", default="en", choices=supported_targets())
    parser.add_argument("--history-pdf")
    parser.add_argument("--print-history", action="store_true")
    args = parser.parse_args(argv)

    if args.history_pdf:
        history_path = Path(args.db).with_suffix(".history.json")
        rows = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
        out = Path(args.history_pdf)
        write_records_pdf(rows, out, "Thamudic + NLP Translation / Scan History")
        if args.print_history: print_pdf(out)
        print(out)
        return 0
    if args.text and args.scan:
        print(_safe_json(scan_text(args.text, args.script))); return 0
    if args.text and args.translate:
        result = translate_text(args.text, args.script, args.target)
        print(_safe_json(result)); return 0
    if args.file:
        print(_safe_json(process_media(args.file, args.script, args.target))); return 0

    UnifiedApp(args.db).mainloop()
    return 0


def main(argv: list[str] | None = None) -> int:
    return cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
