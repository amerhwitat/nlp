"""Native Tkinter GUI for the Python Thamudic API."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from thamudic import extract, transliterate, utf8_bytes


class ThamudicGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Ancient North Arabian — Python GUI")
        self.geometry("900x620")
        self._build()

    def _build(self) -> None:
        root = ttk.Frame(self, padding=16)
        root.pack(fill="both", expand=True)
        ttk.Label(root, text="Thamudic / Ancient North Arabian", font=("TkDefaultFont", 18, "bold")).pack(anchor="w")
        ttk.Label(root, text="Enter Unicode text to extract, transliterate and inspect UTF-8 bytes.").pack(anchor="w", pady=(0, 10))
        self.input = tk.Text(root, height=8, wrap="word")
        self.input.pack(fill="x")
        ttk.Button(root, text="Process", command=self.process).pack(anchor="w", pady=8)
        self.output = tk.Text(root, height=18, wrap="word", state="disabled")
        self.output.pack(fill="both", expand=True)

    def process(self) -> None:
        text = self.input.get("1.0", "end-1c")
        result = "Extracted:\n" + extract(text)
        result += "\n\nTransliteration:\n" + transliterate(text)
        result += "\n\nUTF-8:\n" + " ".join(f"{b:02x}" for b in utf8_bytes(text))
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", result)
        self.output.configure(state="disabled")


if __name__ == "__main__":
    ThamudicGUI().mainloop()
