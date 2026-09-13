#!/usr/bin/env python3
"""Progress-enabled launcher for the unified Thamudic/NLP workbench.

The underlying scanner remains the single all-in-one implementation. This
thin entry point adds a GUI-safe progress indicator without touching Tkinter
from the worker thread. It works for both Thamudic and NLP modes.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from thamudic_all_in_one import App as BaseApp
from thamudic_all_in_one import NLPApp as BaseNLPApp


class ProgressMixin:
    def __init__(self, *args, **kwargs):
        self._progress_after_id = None
        self._progress_bar = None
        super().__init__(*args, **kwargs)

    def _show_section(self, section):
        super()._show_section(section)
        if section == "Scanner":
            self._install_progress_bar()

    def _install_progress_bar(self):
        if not hasattr(self, "workspace"):
            return
        holder = ttk.Frame(self.workspace, style="App.TFrame")
        holder.pack(fill="x", side="top", pady=(0, 8))
        ttk.Label(holder, text="Scanning progress", style="Muted.TLabel").pack(side="left", padx=(0, 8))
        self._progress_bar = ttk.Progressbar(holder, mode="indeterminate", maximum=100)
        self._progress_bar.pack(side="left", fill="x", expand=True)
        self._progress_text = tk.StringVar(value="Ready")
        ttk.Label(holder, textvariable=self._progress_text, style="Muted.TLabel", width=34).pack(side="left", padx=(8, 0))
        self._sync_progress()

    def _start_progress(self):
        if self._progress_bar is None:
            self._show_section("Scanner")
        if self._progress_bar is not None:
            self._progress_bar.start(12)
            self._progress_text.set("Preparing image…")

    def _sync_progress(self):
        if self._progress_bar is not None:
            running = bool(getattr(self, "scan_running", False))
            if running:
                self._progress_text.set("Scanning image in local worker…")
                self._progress_bar.configure(mode="indeterminate")
                self._progress_bar.start(12)
            else:
                self._progress_bar.stop()
                self._progress_bar.configure(mode="determinate", value=100)
                if getattr(self, "current_result", None):
                    self._progress_text.set("Scan complete")
                else:
                    self._progress_text.set("Ready")
        self._progress_after_id = self.after(100, self._sync_progress)

    def scan(self):
        self._start_progress()
        try:
            return super().scan()
        except Exception:
            if self._progress_bar is not None:
                self._progress_bar.stop()
                self._progress_bar.configure(mode="determinate", value=0)
                self._progress_text.set("Scan failed")
            raise

    def clear_current(self):
        result = super().clear_current()
        if self._progress_bar is not None:
            self._progress_bar.stop()
            self._progress_bar.configure(mode="determinate", value=0)
            self._progress_text.set("Ready")
        return result

    def _close(self):
        if self._progress_after_id:
            try:
                self.after_cancel(self._progress_after_id)
            except Exception:
                pass
            self._progress_after_id = None
        return super()._close()


class ThamudicProgressApp(ProgressMixin, BaseApp):
    """Thamudic all-in-one GUI with a worker-safe scan progress bar."""
    pass


class NLPProgressApp(ProgressMixin, BaseNLPApp):
    """NLP all-in-one GUI with the identical worker-safe progress bar."""
    pass


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Progress-enabled unified Thamudic/NLP scanner")
    parser.add_argument("mode", choices=("thamudic", "nlp"), default="nlp", nargs="?")
    parser.add_argument("--db", default="ancient_objects.sqlite")
    args = parser.parse_args()
    (NLPProgressApp if args.mode == "nlp" else ThamudicProgressApp)(db_path=args.db).mainloop()


if __name__ == "__main__":
    main()
