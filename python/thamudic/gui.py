"""Tkinter desktop UI for scanning, transliteration and corpus-backed translation."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .ancient_translation import supported_targets, translate


class ThamudicTranslatorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Thamudic / Ancient North Arabian Translator")
        self.geometry("980x700")
        self._build()

    def _build(self) -> None:
        ttk.Label(self, text="Inscription / scholarly transliteration").pack(anchor="w", padx=12, pady=(12, 4))
        self.source = tk.Text(self, height=8, wrap="word")
        self.source.pack(fill="x", padx=12)

        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=12, pady=10)
        ttk.Label(controls, text="Script").pack(side="left")
        self.script = ttk.Combobox(controls, values=["Dadanitic", "Safaitic", "Hismaic", "Taymanitic", "Thamudic B"], state="readonly", width=16)
        self.script.set("Dadanitic")
        self.script.pack(side="left", padx=6)
        ttk.Label(controls, text="Target").pack(side="left", padx=(12, 0))
        self.target = ttk.Combobox(controls, values=list(supported_targets()), state="readonly", width=10)
        self.target.set("en")
        self.target.pack(side="left", padx=6)
        ttk.Button(controls, text="Translate + transliterate", command=self.translate).pack(side="left", padx=12)

        ttk.Label(self, text="Transliteration").pack(anchor="w", padx=12, pady=(4, 4))
        self.transliteration = tk.Text(self, height=5, wrap="word")
        self.transliteration.pack(fill="x", padx=12)

        ttk.Label(self, text="Translation").pack(anchor="w", padx=12, pady=(12, 4))
        self.translation = tk.Text(self, height=7, wrap="word")
        self.translation.pack(fill="both", expand=True, padx=12)
        self.status = ttk.Label(self, text="Ready")
        self.status.pack(anchor="w", padx=12, pady=8)

    def translate(self) -> None:
        text = self.source.get("1.0", "end-1c").strip()
        result = translate(text, script=self.script.get(), target_language=self.target.get())
        self.transliteration.delete("1.0", "end")
        self.transliteration.insert("1.0", result["transliteration"])
        self.translation.delete("1.0", "end")
        self.translation.insert("1.0", result["translation"] or "No corpus-backed translation is available for this reading.")
        self.status.config(text=f"{result['translation_status']} · confidence={result['confidence']} · corpus={result['corpus_id'] or 'none'}")


def main() -> None:
    ThamudicTranslatorApp().mainloop()


if __name__ == "__main__":
    main()
