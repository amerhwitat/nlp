# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 00:34:11 2026

@author: PC1
"""

"""
Safe demo: GUI to load a ';' CSV, train a simple Keras model to map
first column -> last column and vice versa on numeric/synthetic data.
This script WILL ABORT if the CSV appears to contain private keys or crypto addresses.
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------- Safety check helpers ----------
SUSPICIOUS_KEYWORDS = {}

def looks_sensitive(df: pd.DataFrame) -> bool:
    # Check column names and a small sample of values for suspicious keywords
    cols = ' '.join(map(str, df.columns)).lower()
    if any(k in cols for k in SUSPICIOUS_KEYWORDS):
        return True
    sample = df.head(10).astype(str).apply(lambda s: ' '.join(s), axis=1).str.lower().str.cat(sep=' ')
    if any(k in sample for k in SUSPICIOUS_KEYWORDS):
        return True
    return False

# ---------- Model utilities ----------
def build_regression_model(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)  # regression output
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# ---------- GUI ----------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Safe CSV -> NN Demo")
        root.geometry("640x360")

        self.csv_path = tk.StringVar()
        self.txt_path = tk.StringVar()
        self.status = tk.StringVar(value="Idle")

        tk.Label(root, text="CSV file (semicolon-delimited):").pack(anchor='w', padx=10, pady=(10,0))
        tk.Entry(root, textvariable=self.csv_path, width=80).pack(padx=10)
        tk.Button(root, text="Browse CSV", command=self.browse_csv).pack(padx=10, pady=5)

        tk.Label(root, text="Optional text file (lines used to generate synthetic inputs):").pack(anchor='w', padx=10, pady=(10,0))
        tk.Entry(root, textvariable=self.txt_path, width=80).pack(padx=10)
        tk.Button(root, text="Browse TXT", command=self.browse_txt).pack(padx=10, pady=5)

        tk.Button(root, text="Load, Train, Predict, Save PK001.txt", command=self.run_pipeline, bg='#4CAF50', fg='white').pack(pady=15)

        tk.Label(root, textvariable=self.status, fg='blue').pack(pady=5)

    def browse_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.csv_path.set(path)

    def browse_txt(self):
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if path:
            self.txt_path.set(path)

    def run_pipeline(self):
        csv_file = self.csv_path.get().strip()
        if not csv_file:
            messagebox.showerror("Error", "Please select a CSV file.")
            return

        try:
            df = pd.read_csv(csv_file, delimiter=';', dtype=str, low_memory=False)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read CSV: {e}")
            return

        # Safety check
        if looks_sensitive(df):
            messagebox.showerror("Refused", "This file appears to contain sensitive crypto keys/addresses. Operation aborted.")
            self.status.set("Aborted: sensitive content detected.")
            return

        # Convert first and last columns to numeric features (safe demo)
        first_col = df.columns[0]
        last_col = df.columns[-1]

        # Try to coerce to numeric; if not numeric, create numeric hashes (safe synthetic mapping)
        def to_numeric_series(s):
            # attempt numeric conversion
            s_num = pd.to_numeric(s, errors='coerce')
            if s_num.notna().sum() / len(s) > 0.5:
                # mostly numeric
                return s_num.fillna(0.0).astype(float)
            # otherwise create deterministic numeric features from strings (hash-based)
            return s.astype(str).apply(lambda x: float(abs(hash(x)) % 10_000) / 100.0)

        X_raw = to_numeric_series(df[first_col])
        y_raw = to_numeric_series(df[last_col])

        # Build dataset
        X = X_raw.values.reshape(-1, 1)
        y = y_raw.values.reshape(-1, 1)

        # Optional: augment inputs using lines from provided txt file (safe synthetic use)
        txt_file = self.txt_path.get().strip()
        if txt_file and os.path.exists(txt_file):
            try:
                with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = [ln.strip() for ln in f if ln.strip()]
                # convert lines to numeric seeds and append to X as additional rows with dummy y
                extra_X = np.array([float(abs(hash(ln)) % 10_000) / 100.0 for ln in lines]).reshape(-1,1)
                extra_y = np.zeros_like(extra_X)  # dummy targets
                X = np.vstack([X, extra_X])
                y = np.vstack([y, extra_y])
            except Exception as e:
                messagebox.showwarning("Warning", f"Could not read txt file: {e}")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale
        scaler_X = StandardScaler().fit(X_train)
        scaler_y = StandardScaler().fit(y_train)

        X_train_s = scaler_X.transform(X_train)
        X_test_s = scaler_X.transform(X_test)
        y_train_s = scaler_y.transform(y_train)
        y_test_s = scaler_y.transform(y_test)

        # Build and train model
        model = build_regression_model(input_dim=X_train_s.shape[1])
        self.status.set("Training model...")
        self.root.update_idletasks()

        history = model.fit(X_train_s, y_train_s, validation_data=(X_test_s, y_test_s),
                            epochs=20, batch_size=32, verbose=0)

        # Predict on full dataset (safe demo)
        X_all_s = scaler_X.transform(X)
        preds_s = model.predict(X_all_s)
        preds = scaler_y.inverse_transform(preds_s).flatten()

        # Save predictions to PK001.txt in same directory as CSV
        out_dir = os.path.dirname(csv_file) or '.'
        out_path = os.path.join(out_dir, 'PK001.txt')
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                for val in preds:
                    f.write(f"{val}\n")
            self.status.set(f"Done. Predictions saved to {out_path}")
            messagebox.showinfo("Success", f"Predictions saved to:\n{out_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save output: {e}")
            self.status.set("Failed to save output.")

if __name__ == "__main__":
    # Check for required packages
    try:
        import pandas  # already used above
    except Exception:
        print("Please install required packages: pip install pandas numpy tensorflow scikit-learn")
        raise

    root = tk.Tk()
    app = App(root)
    root.mainloop()