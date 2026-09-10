#!/usr/bin/env python3
"""Tkinter desktop front end for the Ancient Script Scanner.

Main workflow:
    Import image -> scan -> inspect evidence -> export JSON/CSV/TXT

The GUI is intentionally human-in-the-loop. It does not claim automatic
translation of an inscription. Transliteration/translation fields are
editable evidence fields and can be exported with the scan record.
"""
from __future__ import annotations

import csv
import json
import tkinter as tk
from dataclasses import asdict
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from ancient_script_registry import SCRIPTS
from thamudic_scanner import scan_image


class AncientScriptScannerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Ancient Script Scanner & Transliteration — Thamudic / ANA")
        self.geometry("1200x760")
        self.minsize(900, 600)
        self.current_image: Path | None = None
        self.current_result: dict | None = None
        self.preview = None
        self._build_ui()

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self, padding=8)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Import Image", command=self.import_image).pack(side="left", padx=3)
        ttk.Button(toolbar, text="Scan", command=self.run_scan).pack(side="left", padx=3)
        ttk.Button(toolbar, text="Export JSON", command=lambda: self.export("json")).pack(side="left", padx=3)
        ttk.Button(toolbar, text="Export CSV", command=lambda: self.export("csv")).pack(side="left", padx=3)
        ttk.Button(toolbar, text="Export TXT", command=lambda: self.export("txt")).pack(side="left", padx=3)
        ttk.Button(toolbar, text="Clear", command=self.clear).pack(side="left", padx=3)

        main = ttk.Panedwindow(self, orient="horizontal")
        main.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        left = ttk.Frame(main, padding=6)
        right = ttk.Frame(main, padding=6)
        main.add(left, weight=3)
        main.add(right, weight=2)

        self.image_label = ttk.Label(left, text="Import an inscription photograph", anchor="center")
        self.image_label.pack(fill="both", expand=True)

        meta = ttk.LabelFrame(right, text="Script / Evidence", padding=8)
        meta.pack(fill="x")
        ttk.Label(meta, text="Script family / variety").pack(anchor="w")
        self.script_var = tk.StringVar(value="old_north_arabian")
        self.script_combo = ttk.Combobox(meta, textvariable=self.script_var, state="readonly", values=sorted(SCRIPTS))
        self.script_combo.pack(fill="x", pady=(2, 8))
        self.script_combo.bind("<<ComboboxSelected>>", self.script_changed)
        self.script_notes = tk.StringVar()
        ttk.Label(meta, textvariable=self.script_notes, wraplength=390).pack(anchor="w")

        text = ttk.LabelFrame(right, text="Transliteration / Translation (human-reviewed)", padding=8)
        text.pack(fill="both", expand=True, pady=8)
        ttk.Label(text, text="Transliteration").pack(anchor="w")
        self.translit = tk.Text(text, height=5, wrap="word")
        self.translit.pack(fill="x", pady=(2, 8))
        ttk.Label(text, text="Arabic translation / notes").pack(anchor="w")
        self.arabic = tk.Text(text, height=5, wrap="word")
        self.arabic.pack(fill="x", pady=(2, 8))
        ttk.Label(text, text="English translation / notes").pack(anchor="w")
        self.english = tk.Text(text, height=5, wrap="word")
        self.english.pack(fill="x", pady=(2, 8))
        ttk.Label(text, text="Research / provenance notes").pack(anchor="w")
        self.notes = tk.Text(text, height=6, wrap="word")
        self.notes.pack(fill="both", expand=True)

        self.status = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status, relief="sunken", anchor="w").pack(fill="x", side="bottom")
        self.script_changed()

    def script_changed(self, _event=None) -> None:
        profile = SCRIPTS[self.script_var.get()]
        self.script_notes.set(f"{profile.name} | {profile.family} | {profile.unicode_range} | {profile.direction}\n{profile.notes}")

    def import_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Import inscription image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp"), ("All files", "*.*")],
        )
        if not path:
            return
        self.current_image = Path(path)
        image = Image.open(path)
        image.thumbnail((700, 650))
        self.preview = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.preview, text="")
        self.status.set(f"Imported: {self.current_image.name}")

    def run_scan(self) -> None:
        if not self.current_image:
            messagebox.showinfo("Import image", "Please import an inscription image first.")
            return
        try:
            self.current_result = scan_image(self.current_image)
        except Exception as exc:
            messagebox.showerror("Scan error", str(exc))
            return
        self.current_result["script_profile"] = asdict(SCRIPTS[self.script_var.get()])
        self.current_result["human_review"] = {
            "transliteration": self.translit.get("1.0", "end-1c"),
            "arabic_translation_or_notes": self.arabic.get("1.0", "end-1c"),
            "english_translation_or_notes": self.english.get("1.0", "end-1c"),
            "research_notes": self.notes.get("1.0", "end-1c"),
        }
        self.status.set(f"Scan complete: {len(self.current_result['glyphs'])} candidate components")

    def _sync_review(self) -> None:
        if self.current_result is None:
            return
        self.current_result["script_profile"] = asdict(SCRIPTS[self.script_var.get()])
        self.current_result["human_review"] = {
            "transliteration": self.translit.get("1.0", "end-1c"),
            "arabic_translation_or_notes": self.arabic.get("1.0", "end-1c"),
            "english_translation_or_notes": self.english.get("1.0", "end-1c"),
            "research_notes": self.notes.get("1.0", "end-1c"),
        }

    def export(self, kind: str) -> None:
        if self.current_result is None:
            messagebox.showinfo("Nothing to export", "Run a scan first.")
            return
        self._sync_review()
        ext = "." + kind
        path = filedialog.asksaveasfilename(defaultextension=ext, filetypes=[(kind.upper(), "*" + ext)])
        if not path:
            return
        out = Path(path)
        if kind == "json":
            out.write_text(json.dumps(self.current_result, ensure_ascii=False, indent=2), encoding="utf-8")
        elif kind == "csv":
            rows = self.current_result.get("glyphs", [])
            with out.open("w", newline="", encoding="utf-8") as fh:
                fields = list(rows[0].keys()) if rows else ["index"]
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        else:
            review = self.current_result.get("human_review", {})
            lines = [
                "Ancient Script Scanner Evidence Record",
                f"Source: {self.current_result.get('source_image', '')}",
                f"Script: {self.current_result.get('script_profile', {}).get('name', '')}",
                f"Unicode: {self.current_result.get('unicode_range', '')}",
                f"Candidate glyph components: {len(self.current_result.get('glyphs', []))}",
                "",
                "Transliteration:", review.get("transliteration", ""),
                "Arabic translation/notes:", review.get("arabic_translation_or_notes", ""),
                "English translation/notes:", review.get("english_translation_or_notes", ""),
                "Research/provenance notes:", review.get("research_notes", ""),
                "",
                "WARNING: segmentation/OCR candidates are not automatically validated scholarly readings.",
            ]
            out.write_text("\n".join(lines), encoding="utf-8")
        self.status.set(f"Exported: {out}")

    def clear(self) -> None:
        self.current_image = None
        self.current_result = None
        self.preview = None
        self.image_label.configure(image="", text="Import an inscription photograph")
        for widget in (self.translit, self.arabic, self.english, self.notes):
            widget.delete("1.0", "end")
        self.status.set("Ready")


if __name__ == "__main__":
    AncientScriptScannerApp().mainloop()
