#!/usr/bin/env python3
"""Standalone cryptography research GUI.

Safe scope: hashing user-provided text only. No wallet/key searching or
transaction functionality is included.
"""
from __future__ import annotations
import hashlib
import tkinter as tk
from tkinter import ttk

ALGORITHMS = ("sha256", "sha512", "sha3_256", "sha3_512")


def digest(text: str, algorithm: str) -> str:
    if algorithm not in ALGORITHMS:
        raise ValueError("Unsupported algorithm")
    return hashlib.new(algorithm, text.encode("utf-8")).hexdigest()


def main() -> None:
    root = tk.Tk()
    root.title("Cryptography Research Lab")
    root.geometry("760x520")

    frame = ttk.Frame(root, padding=16)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Input text").pack(anchor="w")
    source = tk.Text(frame, height=9, wrap="word")
    source.pack(fill="x", pady=(4, 12))

    output = tk.Text(frame, height=14, wrap="none")
    output.pack(fill="both", expand=True, pady=(8, 8))

    def calculate() -> None:
        text = source.get("1.0", "end-1c")
        output.delete("1.0", "end")
        for algorithm in ALGORITHMS:
            output.insert("end", f"{algorithm}:\n{digest(text, algorithm)}\n\n")

    buttons = ttk.Frame(frame)
    buttons.pack(fill="x")
    ttk.Button(buttons, text="Calculate digests", command=calculate).pack(side="left")
    ttk.Button(buttons, text="Clear", command=lambda: output.delete("1.0", "end")).pack(side="left", padx=8)
    root.mainloop()


if __name__ == "__main__":
    main()
