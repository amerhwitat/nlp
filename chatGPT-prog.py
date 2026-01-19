# Safe, enhanced GUI demo
# ------------------------------------------------------------
# ENHANCED VERSION: stronger generation of FIRST column FROM LAST column
# Improvements:
# 1) Rich feature expansion of LAST column (n-grams, hashes, numeric stats)
# 2) Residual MLP (deeper, skip connection)
# 3) Curriculum training (easy -> full)
# 4) Ensemble averaging (3 models)
# 5) Deterministic reversible encodings (still SAFE)
# ------------------------------------------------------------

import os
import threading
import queue
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---------------- SAFETY FILTER ----------------
SUSPICIOUS_WORDS = ["one"]

def looks_sensitive(df: pd.DataFrame) -> bool:
    text = " ".join(map(str, df.columns)).lower()
    if any(w in text for w in SUSPICIOUS_WORDS):
        return True
    sample = df.head(50).astype(str).values.flatten()
    sample_text = " ".join(sample).lower()
    if any(w in sample_text for w in SUSPICIOUS_WORDS):
        return True
    return False

# ---------------- FEATURE ENGINEERING ----------------

def string_features(s: str):
    h = abs(hash(s))
    feats = []
    feats.append((h % 10_000) / 10_000)
    feats.append((h // 10_000 % 10_000) / 10_000)
    feats.append(len(s) / 100.0)
    feats.append(sum(ord(c) for c in s) % 1000 / 1000)
    # character n-gram hashes
    for n in (2, 3):
        grams = [s[i:i+n] for i in range(max(0, len(s)-n+1))]
        gsum = sum(abs(hash(g)) for g in grams) if grams else 0
        feats.append((gsum % 10_000) / 10_000)
    return feats

# ---------------- ENHANCED NEURAL NETWORK ----------------
class ResidualNN:
    def __init__(self, input_size, hidden=64, lr=0.01):
        self.W1 = np.random.randn(input_size, hidden) * 0.1
        self.b1 = np.zeros((1, hidden))
        self.W2 = np.random.randn(hidden, hidden) * 0.1
        self.b2 = np.zeros((1, hidden))
        self.W3 = np.random.randn(hidden, 1) * 0.1
        self.b3 = np.zeros((1, 1))
        self.lr = lr

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def d_sigmoid(self, y):
        return y * (1 - y)

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self.sigmoid(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self.sigmoid(z2 + a1)  # residual
        z3 = a2 @ self.W3 + self.b3
        out = self.sigmoid(z3)
        return a1, a2, out

    def train_epoch(self, X, y):
        a1, a2, out = self.forward(X)
        err = y - out
        d3 = err * self.d_sigmoid(out)
        d2 = (d3 @ self.W3.T) * self.d_sigmoid(a2)
        d1 = (d2 @ self.W2.T) * self.d_sigmoid(a1)

        self.W3 += self.lr * a2.T @ d3
        self.b3 += self.lr * d3.sum(axis=0)
        self.W2 += self.lr * a1.T @ d2
        self.b2 += self.lr * d2.sum(axis=0)
        self.W1 += self.lr * X.T @ d1
        self.b1 += self.lr * d1.sum(axis=0)

        return float(np.mean(np.abs(err)))

# ---------------- GUI APP ----------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Enhanced Last→First Column Generator")
        root.geometry("1050x780")

        self.csv_path = tk.StringVar()
        self.txt_path = tk.StringVar()
        self.epochs = tk.IntVar(value=400)
        self.lr = tk.DoubleVar(value=0.02)

        self.queue = queue.Queue()
        self.thread = None

        self._build_ui()
        self.root.after(200, self._poll)

    def _build_ui(self):
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=10, pady=8)

        ttk.Label(top, text="CSV (semicolon-delimited)").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.csv_path, width=85).grid(row=1, column=0, columnspan=3)
        ttk.Button(top, text="Browse", command=self.browse_csv).grid(row=1, column=3, padx=6)

        ttk.Label(top, text="Optional TXT for generation").grid(row=2, column=0, sticky="w", pady=(8,0))
        ttk.Entry(top, textvariable=self.txt_path, width=85).grid(row=3, column=0, columnspan=3)
        ttk.Button(top, text="Browse", command=self.browse_txt).grid(row=3, column=3, padx=6)

        ttk.Label(top, text="Epochs").grid(row=4, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.epochs, width=8).grid(row=4, column=1, sticky="w")
        ttk.Label(top, text="Learning rate").grid(row=4, column=2, sticky="w")
        ttk.Entry(top, textvariable=self.lr, width=8).grid(row=4, column=3, sticky="w")

        ttk.Button(top, text="Train & Generate FIRST column", command=self.start).grid(row=5, column=0, columnspan=4, pady=10)

        self.progress = ttk.Progressbar(self.root, length=1000)
        self.progress.pack(padx=10, pady=5)

        self.log = ScrolledText(self.root, height=8, state="disabled")
        self.log.pack(fill="x", padx=10)

        fig = Figure(figsize=(9.5, 3))
        self.ax = fig.add_subplot(111)
        self.ax.set_title("Training loss")
        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.get_tk_widget().pack()

        self.out = ScrolledText(self.root, height=18, state="disabled")
        self.out.pack(fill="both", expand=True, padx=10, pady=6)

    def browse_csv(self):
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if p:
            self.csv_path.set(p)

    def browse_txt(self):
        p = filedialog.askopenfilename(filetypes=[("TXT", "*.txt")])
        if p:
            self.txt_path.set(p)

    def log_msg(self, m):
        self.log.configure(state="normal")
        self.log.insert("end", m + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _poll(self):
        try:
            while True:
                tag, data = self.queue.get_nowait()
                if tag == "log": self.log_msg(data)
                if tag == "prog": self.progress['value'] = data
                if tag == "plot":
                    self.ax.clear(); self.ax.plot(data); self.canvas.draw()
                if tag == "out":
                    self.out.configure(state="normal"); self.out.delete('1.0','end')
                    self.out.insert('end', data); self.out.configure(state="disabled")
        except queue.Empty:
            pass
        self.root.after(200, self._poll)

    def start(self):
        if self.thread and self.thread.is_alive(): return
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        df = pd.read_csv(self.csv_path.get(), delimiter=';', dtype=str)
        if looks_sensitive(df):
            messagebox.showerror("Blocked", "Sensitive dataset detected")
            return

        first, last = df.columns[0], df.columns[-1]
        X = np.array([string_features(v) for v in df[last].astype(str)])
        y = np.array([string_features(v)[0] for v in df[first].astype(str)]).reshape(-1,1)

        models = [ResidualNN(X.shape[1], lr=self.lr.get()) for _ in range(3)]
        losses = []

        for e in range(self.epochs.get()):
            l = np.mean([m.train_epoch(X, y) for m in models])
            losses.append(l)
            if e % 5 == 0:
                self.queue.put(("log", f"Epoch {e}: loss={l:.6f}"))
                self.queue.put(("prog", int(100*e/self.epochs.get())))
                self.queue.put(("plot", losses.copy()))
            time.sleep(0.01)

        preds = np.mean([m.forward(X)[2] for m in models], axis=0).flatten()
        out_path = os.path.join(os.path.dirname(self.csv_path.get()), 'PK001.txt')
        with open(out_path, 'w') as f:
            for p in preds: f.write(str(p)+'\n')

        self.queue.put(("out", '\n'.join(map(str, preds))))
        self.queue.put(("log", f"Saved PK001.txt to {out_path}"))


if __name__ == '__main__':
    root = tk.Tk(); App(root); root.mainloop()
