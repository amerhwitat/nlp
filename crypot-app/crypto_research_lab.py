"""Standalone cryptography research GUI and reusable digest API."""
from __future__ import annotations
import hashlib
import tkinter as tk
from tkinter import ttk


def digest_text(text: str) -> dict[str, str]:
    data = text.encode("utf-8")
    return {name: getattr(hashlib, name)(data).hexdigest() for name in ("sha256", "sha512", "sha3_256", "sha3_512")}


def main() -> None:
    root = tk.Tk()
    root.title("Crypto Research Lab")
    root.geometry("760x520")
    ttk.Label(root, text="Input").pack(anchor="w", padx=12, pady=(12, 4))
    entry = tk.Text(root, height=8, wrap="word")
    entry.pack(fill="x", padx=12)
    output = tk.Text(root, height=18, wrap="word")
    output.pack(fill="both", expand=True, padx=12, pady=12)

    def run():
        output.delete("1.0", "end")
        for name, value in digest_text(entry.get("1.0", "end-1c")).items():
            output.insert("end", f"{name}:\n{value}\n\n")

    ttk.Button(root, text="Hash", command=run).pack(pady=(0, 12))
    root.mainloop()


if __name__ == "__main__":
    main()
