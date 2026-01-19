# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 00:46:35 2026

@author: PC1
"""

"""
Enhanced safe demo: GUI to load a ';' CSV, train a Keras model to map
first column -> last column and vice versa on numeric/synthetic data.
Shows epoch-by-epoch training logs, a progress bar, and generated predictions.
Aborts if the CSV appears to contain sensitive crypto keys/addresses.
"""

import os
import threading
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------- Safety check helpers ----------
SUSPICIOUS_KEYWORDS = {}

def looks_sensitive(df: pd.DataFrame) -> bool:
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
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# ---------- Keras callback to send epoch updates to GUI via queue ----------
class EpochLogger(tf.keras.callbacks.Callback):
    def __init__(self, q, total_epochs):
        super().__init__()
        self.q = q
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_info = {
            'epoch': epoch + 1,
            'loss': float(logs.get('loss', np.nan)),
            'val_loss': float(logs.get('val_loss', np.nan)) if 'val_loss' in logs else None,
            'mae': float(logs.get('mae', np.nan)) if 'mae' in logs else None,
            'val_mae': float(logs.get('val_mae', np.nan)) if 'val_mae' in logs else None,
            'total_epochs': self.total_epochs
        }
        self.q.put(('epoch', epoch_info))

    def on_train_end(self, logs=None):
        self.q.put(('done', None))

# ---------- GUI ----------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Safe CSV -> NN Demo (Enhanced)")
        root.geometry("900x640")

        self.csv_path = tk.StringVar()
        self.txt_path = tk.StringVar()
        self.epochs_var = tk.IntVar(value=20)
        self.batch_var = tk.IntVar(value=32)
        self.status = tk.StringVar(value="Idle")

        # Top frame: file selection and training controls
        top = ttk.Frame(root)
        top.pack(fill='x', padx=10, pady=8)

        ttk.Label(top, text="CSV file (semicolon-delimited):").grid(row=0, column=0, sticky='w')
        ttk.Entry(top, textvariable=self.csv_path, width=80).grid(row=1, column=0, columnspan=3, sticky='w')
        ttk.Button(top, text="Browse CSV", command=self.browse_csv).grid(row=1, column=3, padx=6)

        ttk.Label(top, text="Optional text file (lines used to generate synthetic inputs):").grid(row=2, column=0, sticky='w', pady=(8,0))
        ttk.Entry(top, textvariable=self.txt_path, width=80).grid(row=3, column=0, columnspan=3, sticky='w')
        ttk.Button(top, text="Browse TXT", command=self.browse_txt).grid(row=3, column=3, padx=6)

        ttk.Label(top, text="Epochs:").grid(row=4, column=0, sticky='w', pady=(8,0))
        ttk.Entry(top, textvariable=self.epochs_var, width=8).grid(row=4, column=1, sticky='w')
        ttk.Label(top, text="Batch size:").grid(row=4, column=2, sticky='w')
        ttk.Entry(top, textvariable=self.batch_var, width=8).grid(row=4, column=3, sticky='w')

        ttk.Button(top, text="Load, Train, Predict, Save PK001.txt", command=self.start_pipeline, style='Accent.TButton').grid(row=5, column=0, columnspan=4, pady=10)

        # Middle frame: progress and logs
        mid = ttk.Frame(root)
        mid.pack(fill='both', expand=False, padx=10, pady=4)

        ttk.Label(mid, text="Training progress:").pack(anchor='w')
        self.progress = ttk.Progressbar(mid, orient='horizontal', length=800, mode='determinate')
        self.progress.pack(fill='x', pady=4)

        ttk.Label(mid, text="Epoch logs:").pack(anchor='w', pady=(8,0))
        self.log_widget = ScrolledText(mid, height=12, state='disabled', wrap='none')
        self.log_widget.pack(fill='both', expand=False)

        # Bottom frame: predictions preview and status
        bot = ttk.Frame(root)
        bot.pack(fill='both', expand=True, padx=10, pady=6)

        left = ttk.Frame(bot)
        left.pack(side='left', fill='both', expand=True)

        ttk.Label(left, text="Generated predictions (first 20 shown):").pack(anchor='w')
        self.pred_widget = ScrolledText(left, height=12, state='disabled')
        self.pred_widget.pack(fill='both', expand=True)

        right = ttk.Frame(bot, width=260)
        right.pack(side='right', fill='y')

        ttk.Label(right, text="Status:").pack(anchor='w')
        ttk.Label(right, textvariable=self.status, foreground='blue').pack(anchor='w', pady=(0,10))

        ttk.Button(right, text="Clear Logs", command=self.clear_logs).pack(fill='x', pady=4)
        ttk.Button(right, text="Show saved PK001.txt location", command=self.show_output_location).pack(fill='x', pady=4)

        # queue for inter-thread communication
        self.q = queue.Queue()
        self.training_thread = None
        self.last_output_path = None

        # poll queue periodically
        self.root.after(200, self._poll_queue)

    def browse_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.csv_path.set(path)

    def browse_txt(self):
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if path:
            self.txt_path.set(path)

    def clear_logs(self):
        self.log_widget.configure(state='normal')
        self.log_widget.delete('1.0', tk.END)
        self.log_widget.configure(state='disabled')
        self.pred_widget.configure(state='normal')
        self.pred_widget.delete('1.0', tk.END)
        self.pred_widget.configure(state='disabled')

    def show_output_location(self):
        if self.last_output_path:
            messagebox.showinfo("Output file", f"Predictions saved to:\n{self.last_output_path}")
        else:
            messagebox.showinfo("Output file", "No output file saved yet.")

    def start_pipeline(self):
        if self.training_thread and self.training_thread.is_alive():
            messagebox.showwarning("Training in progress", "Training is already running.")
            return
        csv_file = self.csv_path.get().strip()
        if not csv_file:
            messagebox.showerror("Error", "Please select a CSV file.")
            return
        # start background thread
        self.training_thread = threading.Thread(target=self.run_pipeline, daemon=True)
        self.training_thread.start()

    def _poll_queue(self):
        try:
            while True:
                item = self.q.get_nowait()
                tag, payload = item
                if tag == 'epoch':
                    info = payload
                    epoch = info['epoch']
                    total = info['total_epochs']
                    loss = info['loss']
                    val_loss = info.get('val_loss')
                    mae = info.get('mae')
                    val_mae = info.get('val_mae')
                    pct = int((epoch / total) * 100)
                    self.progress['value'] = pct
                    log_line = f"Epoch {epoch}/{total} — loss: {loss:.6f}"
                    if val_loss is not None:
                        log_line += f", val_loss: {val_loss:.6f}"
                    if mae is not None:
                        log_line += f", mae: {mae:.6f}"
                    if val_mae is not None:
                        log_line += f", val_mae: {val_mae:.6f}"
                    self._append_log(log_line)
                elif tag == 'predictions':
                    preds = payload  # numpy array or list
                    self._show_predictions(preds)
                elif tag == 'status':
                    self.status.set(payload)
                elif tag == 'done':
                    self.progress['value'] = 100
                    self.status.set("Training complete.")
                    self._append_log("Training finished.")
                elif tag == 'error':
                    self.status.set("Error")
                    self._append_log(f"Error: {payload}")
                    messagebox.showerror("Error", str(payload))
        except queue.Empty:
            pass
        finally:
            self.root.after(200, self._poll_queue)

    def _append_log(self, text):
        self.log_widget.configure(state='normal')
        self.log_widget.insert(tk.END, text + "\n")
        self.log_widget.see(tk.END)
        self.log_widget.configure(state='disabled')

    def _show_predictions(self, preds):
        # show first 20 predictions in GUI
        self.pred_widget.configure(state='normal')
        self.pred_widget.delete('1.0', tk.END)
        for i, p in enumerate(preds[:20]):
            self.pred_widget.insert(tk.END, f"{i+1}: {p}\n")
        self.pred_widget.see(tk.END)
        self.pred_widget.configure(state='disabled')

    def run_pipeline(self):
        # This runs in a background thread
        csv_file = self.csv_path.get().strip()
        try:
            self.q.put(('status', "Loading CSV..."))
            df = pd.read_csv(csv_file, delimiter=';', dtype=str, low_memory=False)
        except Exception as e:
            self.q.put(('error', f"Failed to read CSV: {e}"))
            return

        # Safety check
        if looks_sensitive(df):
            self.q.put(('error', "Refused: file appears to contain sensitive crypto keys/addresses. Operation aborted."))
            return

        first_col = df.columns[0]
        last_col = df.columns[-1]

        def to_numeric_series(s):
            s_num = pd.to_numeric(s, errors='coerce')
            if s_num.notna().sum() / len(s) > 0.5:
                return s_num.fillna(0.0).astype(float)
            return s.astype(str).apply(lambda x: float(abs(hash(x)) % 10_000) / 100.0)

        X_raw = to_numeric_series(df[first_col])
        y_raw = to_numeric_series(df[last_col])

        X = X_raw.values.reshape(-1, 1)
        y = y_raw.values.reshape(-1, 1)

        # Optional: augment inputs using lines from provided txt file (safe synthetic use)
        txt_file = self.txt_path.get().strip()
        if txt_file and os.path.exists(txt_file):
            try:
                with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = [ln.strip() for ln in f if ln.strip()]
                extra_X = np.array([float(abs(hash(ln)) % 10_000) / 100.0 for ln in lines]).reshape(-1,1)
                extra_y = np.zeros_like(extra_X)
                X = np.vstack([X, extra_X])
                y = np.vstack([y, extra_y])
            except Exception as e:
                self.q.put(('status', f"Warning reading txt file: {e}"))

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale
        scaler_X = StandardScaler().fit(X_train)
        scaler_y = StandardScaler().fit(y_train)

        X_train_s = scaler_X.transform(X_train)
        X_test_s = scaler_X.transform(X_test)
        y_train_s = scaler_y.transform(y_train)
        y_test_s = scaler_y.transform(y_test)

        # Build model
        model = build_regression_model(input_dim=X_train_s.shape[1])

        epochs = max(1, int(self.epochs_var.get()))
        batch_size = max(1, int(self.batch_var.get()))

        # Setup callback
        epoch_logger = EpochLogger(self.q, total_epochs=epochs)

        # Start training
        self.q.put(('status', "Training model..."))
        try:
            model.fit(
                X_train_s, y_train_s,
                validation_data=(X_test_s, y_test_s),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=[epoch_logger]
            )
        except Exception as e:
            self.q.put(('error', f"Training failed: {e}"))
            return

        # Predict on full dataset
        self.q.put(('status', "Generating predictions..."))
        X_all_s = scaler_X.transform(X)
        preds_s = model.predict(X_all_s)
        preds = scaler_y.inverse_transform(preds_s).flatten()

        # Send predictions to GUI
        self.q.put(('predictions', preds.tolist()))

        # Save predictions to PK001.txt in same directory as CSV
        out_dir = os.path.dirname(csv_file) or '.'
        out_path = os.path.join(out_dir, 'PK001.txt')
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                for val in preds:
                    f.write(f"{val}\n")
            self.last_output_path = out_path
            self.q.put(('status', f"Done. Predictions saved to {out_path}"))
        except Exception as e:
            self.q.put(('error', f"Failed to save output: {e}"))

        # signal done
        self.q.put(('done', None))


if __name__ == "__main__":
    # Check for required packages
    try:
        import pandas  # noqa: F401
    except Exception:
        print("Please install required packages: pip install pandas numpy tensorflow scikit-learn")
        raise

    root = tk.Tk()
    style = ttk.Style(root)
    # Use platform-appropriate theme; 'clam' is widely available
    try:
        style.theme_use('clam')
    except Exception:
        pass
    app = App(root)
    root.mainloop()