#!/usr/bin/env python3
"""Professional Python desktop shell for the Thamudic research system."""
from __future__ import annotations

import json
import tempfile
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from ancient_objects_db import ObjectDatabase
from ancient_script_registry import SCRIPTS
from historical_periods import PERIODS
from object_sources import source_catalog
from softr_export import export_softr_csv, export_softr_json
from thamudic_scanner import scan_image


class ThamudicDesktop(tk.Tk):
    """Single-window research environment with a stable navigation hierarchy."""

    def __init__(self, db_path: str = "ancient_objects.sqlite") -> None:
        super().__init__()
        self.title("Thamudic Scanner — Ancient Languages Research Workbench")
        self.geometry("1500x920")
        self.minsize(1120, 720)
        self.db = ObjectDatabase(db_path)
        self.current_image: Path | None = None
        self.current_result: dict | None = None
        self.preview = None
        self.status_var = tk.StringVar(value="Ready")
        self.search_var = tk.StringVar()
        self.section_var = tk.StringVar(value="Dashboard")
        self._configure_style()
        self._build_shell()
        self._show_section("Dashboard")
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("App.TFrame", background="#10151c")
        style.configure("Sidebar.TFrame", background="#0c1117")
        style.configure("Card.TFrame", background="#18212c", relief="flat")
        style.configure("Title.TLabel", background="#10151c", foreground="#edf2f7", font=("Segoe UI", 22, "bold"))
        style.configure("Muted.TLabel", background="#10151c", foreground="#9eacba", font=("Segoe UI", 10))
        style.configure("Nav.TButton", background="#0c1117", foreground="#c4cfda", anchor="w", padding=(14, 9), borderwidth=0)
        style.map("Nav.TButton", background=[("active", "#202c38")], foreground=[("active", "#ffffff")])
        style.configure("Primary.TButton", background="#8e6b25", foreground="#ffffff", padding=(12, 8))
        style.configure("CardTitle.TLabel", background="#18212c", foreground="#edf2f7", font=("Segoe UI", 14, "bold"))
        style.configure("CardValue.TLabel", background="#18212c", foreground="#d5a84b", font=("Segoe UI", 24, "bold"))
        style.configure("Treeview", rowheight=28)

    def _build_shell(self) -> None:
        self.configure(background="#10151c")
        shell = ttk.Frame(self, style="App.TFrame")
        shell.pack(fill="both", expand=True)
        sidebar = ttk.Frame(shell, width=250, style="Sidebar.TFrame", padding=14)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        ttk.Label(sidebar, text="THAMUDIC", background="#0c1117", foreground="#edf2f7", font=("Segoe UI", 17, "bold")).pack(anchor="w", padx=8, pady=(4, 0))
        ttk.Label(sidebar, text="SCANNER", background="#0c1117", foreground="#d5a84b", font=("Segoe UI", 17, "bold")).pack(anchor="w", padx=8, pady=(0, 20))
        for name in ["Dashboard", "Scanner", "Translator", "Historical Objects", "Inscriptions", "Ancient Scripts", "Sources & Rights", "Database", "Research"]:
            ttk.Button(sidebar, text=name, style="Nav.TButton", command=lambda n=name: self._show_section(n)).pack(fill="x", pady=2)
        ttk.Separator(sidebar).pack(fill="x", pady=15)
        ttk.Button(sidebar, text="Import Image / PDF", command=self.import_media).pack(fill="x", pady=2)
        ttk.Button(sidebar, text="Scan Current Image", style="Primary.TButton", command=self.scan).pack(fill="x", pady=2)

        main = ttk.Frame(shell, style="App.TFrame", padding=(24, 20, 24, 10))
        main.pack(side="left", fill="both", expand=True)
        header = ttk.Frame(main, style="App.TFrame")
        header.pack(fill="x", pady=(0, 16))
        self.title_label = ttk.Label(header, textvariable=self.section_var, style="Title.TLabel")
        self.title_label.pack(side="left")
        ttk.Label(header, textvariable=self.status_var, style="Muted.TLabel").pack(side="right")
        self.workspace = ttk.Frame(main, style="App.TFrame")
        self.workspace.pack(fill="both", expand=True)

    def _clear_workspace(self) -> None:
        for child in self.workspace.winfo_children():
            child.destroy()

    def _show_section(self, section: str) -> None:
        self.section_var.set(section)
        self._clear_workspace()
        builders = {
            "Dashboard": self._dashboard,
            "Scanner": self._scanner,
            "Translator": self._translator,
            "Historical Objects": self._catalog,
            "Inscriptions": self._catalog,
            "Ancient Scripts": self._scripts,
            "Sources & Rights": self._sources,
            "Database": self._database,
            "Research": self._research,
        }
        builders.get(section, self._dashboard)()

    def _card(self, parent, title: str, value: str, column: int) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=18)
        card.grid(row=0, column=column, sticky="nsew", padx=6)
        ttk.Label(card, text=title, style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(card, text=value, style="CardValue.TLabel").pack(anchor="w", pady=(7, 0))

    def _dashboard(self) -> None:
        with self.db as db:
            stats = db.statistics()
            recent = db.list_objects(limit=10)
        self.db = ObjectDatabase(db.path)
        cards = ttk.Frame(self.workspace, style="App.TFrame")
        cards.pack(fill="x", pady=(0, 20))
        for i in range(4): cards.columnconfigure(i, weight=1)
        for i, (label, value) in enumerate([("Objects", stats["objects"]), ("Annotations", stats["annotations"]), ("Sources", stats["sources"]), ("Schema", f"v{stats['schema_version']}")]):
            self._card(cards, label, str(value), i)
        panel = ttk.LabelFrame(self.workspace, text="Recent research records", padding=10)
        panel.pack(fill="both", expand=True)
        tree = ttk.Treeview(panel, columns=("id", "title", "period", "script", "type"), show="headings")
        for c in ("id", "title", "period", "script", "type"): tree.heading(c, text=c.title())
        tree.pack(fill="both", expand=True)
        for r in recent: tree.insert("", "end", values=(r.get("id"), r.get("title"), r.get("period_name") or r.get("period_key"), r.get("script_key"), r.get("object_type")))

    def _scanner(self) -> None:
        pan = ttk.Panedwindow(self.workspace, orient="horizontal")
        pan.pack(fill="both", expand=True)
        left = ttk.Frame(pan, padding=10); right = ttk.Frame(pan, padding=10)
        pan.add(left, weight=3); pan.add(right, weight=2)
        self.image_label = ttk.Label(left, text="Import an inscription photograph or PDF", anchor="center")
        self.image_label.pack(fill="both", expand=True)
        box = ttk.LabelFrame(right, text="Classification & Evidence", padding=12); box.pack(fill="both", expand=True)
        self.script_var = tk.StringVar(value="old_north_arabian")
        self.period_var = tk.StringVar(value="iron_age")
        self.object_type = tk.StringVar(value="inscription")
        for label, variable, values in [("Script / variety", self.script_var, sorted(SCRIPTS)), ("Historical period", self.period_var, [p["key"] for p in PERIODS])]:
            ttk.Label(box, text=label).pack(anchor="w", pady=(0, 4)); ttk.Combobox(box, textvariable=variable, values=values, state="readonly").pack(fill="x", pady=(0, 10))
        ttk.Label(box, text="Object type").pack(anchor="w"); ttk.Entry(box, textvariable=self.object_type).pack(fill="x", pady=(0, 10))
        self.translit = self._text(box, "Transliteration", 4)
        self.arabic = self._text(box, "Arabic translation / notes", 4)
        self.english = self._text(box, "English translation / notes", 4)
        self.notes = self._text(box, "Provenance / bibliography", 5)
        ttk.Button(box, text="Add reviewed evidence to catalog", command=self.add_current_object).pack(fill="x", pady=6)

    def _text(self, parent, label, height):
        ttk.Label(parent, text=label).pack(anchor="w"); widget = tk.Text(parent, height=height, wrap="word", undo=True); widget.pack(fill="x", pady=(2, 7)); return widget

    def _catalog(self) -> None:
        bar = ttk.Frame(self.workspace); bar.pack(fill="x", pady=(0, 10))
        ttk.Entry(bar, textvariable=self.search_var).pack(side="left", fill="x", expand=True)
        ttk.Button(bar, text="Search", command=self._refresh_catalog).pack(side="left", padx=6)
        tree = ttk.Treeview(self.workspace, columns=("id", "title", "period", "type", "script", "source"), show="headings")
        self.catalog_tree = tree
        for c in ("id", "title", "period", "type", "script", "source"): tree.heading(c, text=c.title())
        tree.pack(fill="both", expand=True); self._refresh_catalog()

    def _refresh_catalog(self):
        if not hasattr(self, "catalog_tree"): return
        for i in self.catalog_tree.get_children(): self.catalog_tree.delete(i)
        with self.db as db: rows = db.list_objects(query=self.search_var.get(), limit=500)
        self.db = ObjectDatabase(self.db.path)
        for r in rows: self.catalog_tree.insert("", "end", values=(r.get("id"), r.get("title"), r.get("period_name") or r.get("period_key"), r.get("object_type"), r.get("script_key"), r.get("source_name")))

    def _translator(self):
        panel = ttk.LabelFrame(self.workspace, text="Evidence-aware Translation", padding=14); panel.pack(fill="both", expand=True)
        ttk.Label(panel, text="Enter transliteration or Unicode text. Translation output is advisory and requires scholarly review.").pack(anchor="w")
        text = tk.Text(panel, height=12, wrap="word"); text.pack(fill="both", expand=True, pady=10)
        output = tk.Text(panel, height=8, wrap="word"); output.pack(fill="both", expand=True)
        def analyze():
            output.delete("1.0", "end"); output.insert("1.0", json.dumps({"input": text.get("1.0", "end-1c"), "status": "human_review_required"}, ensure_ascii=False, indent=2))
        ttk.Button(panel, text="Analyze", style="Primary.TButton", command=analyze).pack(anchor="e")

    def _scripts(self):
        tree = ttk.Treeview(self.workspace, columns=("key", "name", "description"), show="headings")
        for c in ("key", "name", "description"): tree.heading(c, text=c.title()); tree.column(c, width=250 if c != "description" else 700)
        tree.pack(fill="both", expand=True)
        for key, value in SCRIPTS.items(): tree.insert("", "end", values=(key, value.name, value.description))

    def _sources(self):
        tree = ttk.Treeview(self.workspace, columns=("name", "homepage", "rights", "image"), show="headings")
        for c in ("name", "homepage", "rights", "image"): tree.heading(c, text=c.title())
        tree.pack(fill="both", expand=True)
        for s in source_catalog(): tree.insert("", "end", values=(s.get("name"), s.get("homepage"), s.get("rights_policy"), s.get("image_policy")))

    def _database(self):
        with self.db as db: stats = db.statistics()
        self.db = ObjectDatabase(self.db.path)
        panel = ttk.LabelFrame(self.workspace, text="Canonical SQLite Database", padding=14); panel.pack(fill="both", expand=True)
        ttk.Label(panel, text="SQLite is the canonical store; JSON, CSV and SQL remain interchange/export formats.").pack(anchor="w")
        text = tk.Text(panel, wrap="word"); text.pack(fill="both", expand=True, pady=10); text.insert("1.0", json.dumps(stats, ensure_ascii=False, indent=2)); text.configure(state="disabled")

    def _research(self):
        panel = ttk.LabelFrame(self.workspace, text="Research Workflow", padding=16); panel.pack(fill="both", expand=True)
        for i, step in enumerate(["Import photograph or PDF", "Normalize and segment glyphs", "Review candidate components", "Record transliteration and translations", "Attach provenance and rights", "Persist evidence in SQLite", "Export JSON/CSV/SQL"] , 1):
            ttk.Label(panel, text=f"{i}. {step}").pack(anchor="w", pady=5)

    def import_media(self):
        path = filedialog.askopenfilename(filetypes=[("Images/PDF", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp *.pdf"), ("All files", "*.*")])
        if not path: return
        p = Path(path)
        if p.suffix.lower() == ".pdf":
            try:
                import fitz
                doc = fitz.open(p); pix = doc[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                tmp = Path(tempfile.gettempdir()) / f"{p.stem}_page1.png"; pix.save(tmp); self.current_image = tmp
            except Exception as exc:
                messagebox.showerror("PDF import", str(exc)); return
        else: self.current_image = p
        image = Image.open(self.current_image); image.thumbnail((900, 700)); self.preview = ImageTk.PhotoImage(image); self._show_section("Scanner"); self.image_label.configure(image=self.preview, text=""); self.status_var.set(f"Imported {p.name}")

    def scan(self):
        if not self.current_image: self.import_media(); return
        try:
            self.current_result = scan_image(self.current_image)
            self.status_var.set(f"Scan complete: {len(self.current_result.get('glyphs', []))} candidate components")
            if hasattr(self, "image_label"): self._show_section("Scanner")
        except Exception as exc: messagebox.showerror("Scan error", str(exc))

    def add_current_object(self):
        if not self.current_result: self.scan()
        if not self.current_result: return
        review = {"transliteration": self.translit.get("1.0", "end-1c"), "translation_ar": self.arabic.get("1.0", "end-1c"), "translation_en": self.english.get("1.0", "end-1c"), "provenance": self.notes.get("1.0", "end-1c")}
        record = {"title": Path(self.current_result.get("source_image", "inscription")).stem, "period_key": self.period_var.get(), "object_type": self.object_type.get(), "script_key": self.script_var.get(), "description": "Scanner evidence record", **review, "source_name": "Local scanner", "image_local_path": self.current_result.get("source_image", ""), "tags": ["thamudic", "ancient-script"]}
        rid = self.db.add_object(record); self.status_var.set(f"Catalog record created: {rid}")

    def export_softr(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            export_softr_csv(self.db.list_objects(), path); self.status_var.set(f"Exported: {path}")

    def export_json(self):
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if path:
            export_softr_json(self.db.list_objects(), path); self.status_var.set(f"Exported: {path}")

    def _close(self):
        self.db.close(); self.destroy()


if __name__ == "__main__":
    ThamudicDesktop().mainloop()
