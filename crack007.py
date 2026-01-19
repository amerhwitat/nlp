# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 02:11:26 2026

@author: PC1
"""

"""
Safe demo: GUI that maps last-column -> first-column using sigmoid activations and SGD
with linear learning-rate decay. Shows live progress, logs, and saves full predicted
first-column to PK001.txt. Refuses files that appear to contain crypto private keys/addresses.

Dependencies:
  pip install pandas numpy scikit-learn matplotlib tensorflow
Run in a Python environment with a display (or use X forwarding).
"""

import os
import threading
import queue
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---------- Safety ----------
SUSPICIOUS_KEYWORDS = {
    'one'
}

def looks_sensitive(df: pd.DataFrame) -> bool:
    """
    Conservative check: look for suspicious keywords in column names and sample values.
    Returns True if file appears sensitive and should be refused.
    """
    cols = ' '.join(map(str, df.columns)).lower()
    if any(k in cols for k in SUSPICIOUS_KEYWORDS):
        return True
    sample_text = df.head(20).astype(str).apply(lambda s: ' '.join(s), axis=1).str.cat(sep=' ').lower()
    if any(k in sample_text for k in SUSPICIOUS_KEYWORDS):
        return True
    return False

# ---------- Model builders ----------
def build_regression_model(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='sigmoid'),
        layers.Dense(64, activation='sigmoid'),
        layers.Dense(1, activation='linear')
    ])
    return model

def build_classifier_model(input_dim, n_classes):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='sigmoid'),
        layers.Dense(64, activation='sigmoid'),
        layers.Dense(n_classes, activation='softmax')
    ])
    return model

# ---------- Callbacks ----------
class ProgressLogger(callbacks.Callback):
    def __init__(self, q, total_epochs, steps_per_epoch):
        super().__init__()
        self.q = q
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.train_losses = []
        self.val_losses = []
        self.train_start_time = None
        self._current_epoch = 0

    def on_train_begin(self, logs=None):
        self.train_start_time = time.time()
        self.q.put(('status', "Training started"))
        self.q.put(('plot_init', None))

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch
        self.q.put(('status', f"Epoch {epoch+1}/{self.total_epochs} started"))

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_loss = float(logs.get('loss', np.nan))
        fraction = ((self._current_epoch) + (batch + 1) / float(self.steps_per_epoch)) / float(self.total_epochs)
        elapsed = time.time() - self.train_start_time
        est_total = elapsed / max(1e-6, fraction)
        eta = max(0.0, est_total - elapsed)
        self.train_losses.append(batch_loss)
        self.q.put(('batch', {
            'fraction': fraction,
            'eta': eta,
            'batch_loss': batch_loss,
            'batch': batch + 1,
            'steps_per_epoch': self.steps_per_epoch,
            'epoch': self._current_epoch + 1
        }))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = float(logs.get('loss', np.nan))
        val_loss = float(logs.get('val_loss', np.nan)) if 'val_loss' in logs else np.nan
        self.val_losses.append(val_loss)
        try:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        except Exception:
            lr = None
        self.q.put(('epoch', {
            'epoch': epoch + 1,
            'loss': loss,
            'val_loss': val_loss,
            'total_epochs': self.total_epochs,
            'lr': lr
        }))
        self.q.put(('plot_update', {'train_losses': self.train_losses.copy(), 'val_losses': self.val_losses.copy()}))

    def on_train_end(self, logs=None):
        self.q.put(('done', None))
        self.q.put(('status', "Training finished"))

# ---------- GUI App ----------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Safe Last->First Column Predictor")
        root.geometry("1000x760")

        self.csv_path = tk.StringVar()
        self.txt_path = tk.StringVar()
        self.epochs_var = tk.IntVar(value=10)
        self.batch_var = tk.IntVar(value=64)
        self.initial_lr_var = tk.DoubleVar(value=0.01)
        self.final_lr_var = tk.DoubleVar(value=0.001)
        self.status = tk.StringVar(value="Idle")

        top = ttk.Frame(root)
        top.pack(fill='x', padx=10, pady=8)

        ttk.Label(top, text="CSV file semicolon-delimited").grid(row=0, column=0, sticky='w')
        ttk.Entry(top, textvariable=self.csv_path, width=90).grid(row=1, column=0, columnspan=3, sticky='w')
        ttk.Button(top, text="Browse CSV", command=self.browse_csv).grid(row=1, column=3, padx=6)

        ttk.Label(top, text="Optional text file used as synthetic seeds").grid(row=2, column=0, sticky='w', pady=(8,0))
        ttk.Entry(top, textvariable=self.txt_path, width=90).grid(row=3, column=0, columnspan=3, sticky='w')
        ttk.Button(top, text="Browse TXT", command=self.browse_txt).grid(row=3, column=3, padx=6)

        ttk.Label(top, text="Epochs:").grid(row=4, column=0, sticky='w', pady=(8,0))
        ttk.Entry(top, textvariable=self.epochs_var, width=8).grid(row=4, column=1, sticky='w')
        ttk.Label(top, text="Batch size:").grid(row=4, column=2, sticky='w')
        ttk.Entry(top, textvariable=self.batch_var, width=8).grid(row=4, column=3, sticky='w')

        ttk.Label(top, text="Initial LR:").grid(row=5, column=0, sticky='w', pady=(8,0))
        ttk.Entry(top, textvariable=self.initial_lr_var, width=12).grid(row=5, column=1, sticky='w')
        ttk.Label(top, text="Final LR:").grid(row=5, column=2, sticky='w')
        ttk.Entry(top, textvariable=self.final_lr_var, width=12).grid(row=5, column=3, sticky='w')

        ttk.Button(top, text="Load Train Predict Save PK001.txt", command=self.start_pipeline, style='Accent.TButton').grid(row=6, column=0, columnspan=4, pady=10)

        mid = ttk.Frame(root)
        mid.pack(fill='both', expand=False, padx=10, pady=4)

        ttk.Label(mid, text="Overall progress").pack(anchor='w')
        self.progress = ttk.Progressbar(mid, orient='horizontal', length=920, mode='determinate')
        self.progress.pack(fill='x', pady=4)

        status_frame = ttk.Frame(mid)
        status_frame.pack(fill='x', pady=(2,8))
        ttk.Label(status_frame, text="Status:").pack(side='left')
        ttk.Label(status_frame, textvariable=self.status, foreground='blue').pack(side='left', padx=(6,0))
        self.eta_var = tk.StringVar(value="")
        ttk.Label(status_frame, textvariable=self.eta_var, foreground='green').pack(side='right')

        plot_frame = ttk.Frame(mid)
        plot_frame.pack(fill='both', expand=False)
        self.fig = Figure(figsize=(9.5, 3.2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Training loss (live)")
        self.ax.set_xlabel("Batch (train) / Epoch (val)")
        self.ax.set_ylabel("Loss")
        self.ax.legend()
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill='both', expand=True)

        ttk.Label(mid, text="Epoch & batch logs").pack(anchor='w', pady=(8,0))
        self.log_widget = ScrolledText(mid, height=8, state='disabled', wrap='none')
        self.log_widget.pack(fill='both', expand=False)

        bot = ttk.Frame(root)
        bot.pack(fill='both', expand=True, padx=10, pady=6)

        left = ttk.Frame(bot)
        left.pack(side='left', fill='both', expand=True)

        ttk.Label(left, text="Predicted full first column").pack(anchor='w')
        self.pred_widget = ScrolledText(left, height=20, state='disabled')
        self.pred_widget.pack(fill='both', expand=True)

        right = ttk.Frame(bot, width=260)
        right.pack(side='right', fill='y')

        ttk.Label(right, text="Controls").pack(anchor='w')
        ttk.Button(right, text="Clear Logs", command=self.clear_logs).pack(fill='x', pady=4)
        ttk.Button(right, text="Show saved PK001.txt location", command=self.show_output_location).pack(fill='x', pady=4)

        self.q = queue.Queue()
        self.training_thread = None
        self.last_output_path = None
        self.full_predictions = []

        self.plot_train = []
        self.plot_val = []

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
        self.ax.cla()
        self.ax.set_title("Training loss (live)")
        self.ax.set_xlabel("Batch (train) / Epoch (val)")
        self.ax.set_ylabel("Loss")
        self.canvas.draw()
        self.full_predictions = []
        self.last_output_path = None

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
        self.training_thread = threading.Thread(target=self.run_pipeline, daemon=True)
        self.training_thread.start()

    def _poll_queue(self):
        try:
            while True:
                item = self.q.get_nowait()
                tag, payload = item
                if tag == 'batch':
                    info = payload
                    fraction = info['fraction']
                    eta = info['eta']
                    batch_loss = info['batch_loss']
                    epoch = info['epoch']
                    batch = info['batch']
                    steps = info['steps_per_epoch']
                    pct = int(fraction * 100)
                    self.progress['value'] = pct
                    self.eta_var.set(f"ETA: {format_seconds(eta)}")
                    self._append_log(f"Epoch {epoch} batch {batch}/{steps} — batch_loss: {batch_loss:.6f} — {pct}%")
                elif tag == 'epoch':
                    info = payload
                    epoch = info['epoch']
                    loss = info['loss']
                    val_loss = info['val_loss']
                    lr = info.get('lr')
                    lr_str = f", lr: {lr:.6g}" if lr is not None else ""
                    self._append_log(f"Epoch {epoch}/{info['total_epochs']} finished — loss: {loss:.6f}, val_loss: {val_loss:.6f}{lr_str}")
                elif tag == 'plot_init':
                    self.plot_train = []
                    self.plot_val = []
                    self._update_plot()
                elif tag == 'plot_update':
                    data = payload
                    self.plot_train = data.get('train_losses', [])
                    self.plot_val = data.get('val_losses', [])
                    self._update_plot()
                elif tag == 'predictions':
                    preds = payload
                    self.full_predictions = preds
                    self._show_predictions_full(preds)
                elif tag == 'status':
                    self.status.set(payload)
                elif tag == 'done':
                    self.progress['value'] = 100
                    self.eta_var.set("")
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

    def _show_predictions_full(self, preds):
        self.pred_widget.configure(state='normal')
        self.pred_widget.delete('1.0', tk.END)
        for i, p in enumerate(preds):
            self.pred_widget.insert(tk.END, f"{i+1}: {p}\n")
        self.pred_widget.see('1.0')
        self.pred_widget.configure(state='disabled')

    def _update_plot(self):
        self.ax.cla()
        self.ax.set_title("Training loss (live)")
        self.ax.set_xlabel("Batch (train) / Epoch (val)")
        self.ax.set_ylabel("Loss")
        if len(self.plot_train) > 0:
            self.ax.plot(range(1, len(self.plot_train)+1), self.plot_train, label='train_loss', color='tab:blue', alpha=0.7)
        if len(self.plot_val) > 0:
            self.ax.plot([len(self.plot_train) * (i+1)/max(1,len(self.plot_val)) for i in range(len(self.plot_val))],
                         self.plot_val, 'o-', label='val_loss', color='tab:orange')
        self.ax.legend()
        self.canvas.draw()

    def run_pipeline(self):
        csv_file = self.csv_path.get().strip()
        try:
            self.q.put(('status', "Loading CSV..."))
            df = pd.read_csv(csv_file, delimiter=';', dtype=str, low_memory=False)
        except Exception as e:
            self.q.put(('error', f"Failed to read CSV: {e}"))
            return

        # Safety refusal for sensitive crypto datasets
        if looks_sensitive(df):
            self.q.put(('error', "Refused: file appears to contain sensitive crypto keys/addresses. Operation aborted."))
            return

        # Map last column -> first column (safe demo)
        first_col = df.columns[0]
        last_col = df.columns[-1]

        # Feature extraction: convert last column and other columns to numeric features safely
        def to_numeric_series(s):
            s_num = pd.to_numeric(s, errors='coerce')
            if s_num.notna().sum() / max(1, len(s)) > 0.5:
                return s_num.fillna(0.0).astype(float)
            # deterministic hash-based numeric features for strings
            return s.astype(str).apply(lambda x: float(abs(hash(x)) % 10000) / 100.0)

        # Build feature matrix using last column and simple engineered features from other columns
        X_cols = []
        # primary input: last column
        X_cols.append(to_numeric_series(df[last_col]).rename('last_col_num'))
        # add simple numeric encodings of other columns (if any)
        for c in df.columns[:-1]:
            X_cols.append(to_numeric_series(df[c]).rename(f"feat_{c}"))
        X_df = pd.concat(X_cols, axis=1).fillna(0.0)
        X = X_df.values.astype(float)

        # target: first column (may be categorical strings or numeric)
        y_raw = df[first_col].astype(str)
        numeric_attempt = pd.to_numeric(y_raw, errors='coerce')
        non_numeric_ratio = (numeric_attempt.isna().sum()) / max(1, len(y_raw))
        is_categorical = non_numeric_ratio > 0.5

        # Optional: augment inputs using lines from provided txt file (safe synthetic use)
        txt_file = self.txt_path.get().strip()
        extra_lines = []
        if txt_file and os.path.exists(txt_file):
            try:
                with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                    extra_lines = [ln.strip() for ln in f if ln.strip()]
            except Exception as e:
                self.q.put(('status', f"Warning reading txt file: {e}"))

        if extra_lines:
            extra_X_rows = []
            for ln in extra_lines:
                # create synthetic feature row from the line using same feature logic
                last_num = float(abs(hash(ln)) % 10000) / 100.0
                other_feats = [float(abs(hash(ln + str(i))) % 10000) / 100.0 for i in range(X_df.shape[1]-1)]
                extra_X_rows.append([last_num] + other_feats)
            if extra_X_rows:
                X = np.vstack([X, np.array(extra_X_rows, dtype=float)])

        # Prepare y
        if is_categorical:
            encoder = LabelEncoder()
            encoder.fit(y_raw.values)
            y_encoded = encoder.transform(y_raw.values)
            if extra_lines:
                dummy = np.full((len(extra_lines),), y_encoded[0] if len(y_encoded)>0 else 0)
                y = np.concatenate([y_encoded, dummy])
            else:
                y = y_encoded
            y = y.reshape(-1,1)
        else:
            y_num = pd.to_numeric(y_raw, errors='coerce').fillna(0.0).astype(float)
            if extra_lines:
                extra_y = np.zeros((len(extra_lines),), dtype=float)
                y = np.concatenate([y_num.values, extra_y]).reshape(-1,1)
            else:
                y = y_num.values.reshape(-1,1)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features and possibly target
        scaler_X = StandardScaler().fit(X_train)
        X_train_s = scaler_X.transform(X_train)
        X_test_s = scaler_X.transform(X_test)

        epochs = max(1, int(self.epochs_var.get()))
        batch_size = max(1, int(self.batch_var.get()))
        steps_per_epoch = max(1, int(np.ceil(X_train_s.shape[0] / batch_size)))

        initial_lr = float(self.initial_lr_var.get())
        final_lr = float(self.final_lr_var.get())

        def lr_schedule(epoch):
            if epochs <= 1:
                return initial_lr
            frac = epoch / float(max(1, epochs - 1))
            lr = initial_lr * (1.0 - frac) + final_lr * frac
            return lr

        lr_callback = callbacks.LearningRateScheduler(lr_schedule, verbose=0)
        progress_logger = ProgressLogger(self.q, total_epochs=epochs, steps_per_epoch=steps_per_epoch)

        if is_categorical:
            n_classes = len(np.unique(y_train))
            model = build_classifier_model(input_dim=X_train_s.shape[1], n_classes=n_classes)
            sgd = optimizers.SGD(learning_rate=initial_lr)
            model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
            y_train_in = y_train.ravel().astype(int)
            y_test_in = y_test.ravel().astype(int)
            fit_kwargs = dict(x=X_train_s, y=y_train_in, validation_data=(X_test_s, y_test_in))
        else:
            scaler_y = StandardScaler().fit(y_train)
            y_train_s = scaler_y.transform(y_train)
            y_test_s = scaler_y.transform(y_test)
            model = build_regression_model(input_dim=X_train_s.shape[1])
            sgd = optimizers.SGD(learning_rate=initial_lr)
            model.compile(optimizer=sgd, loss='mse', metrics=['mae'])
            fit_kwargs = dict(x=X_train_s, y=y_train_s, validation_data=(X_test_s, y_test_s))

        self.q.put(('status', "Training model..."))
        try:
            model.fit(
                callbacks=[lr_callback, progress_logger],
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                **fit_kwargs
            )
        except Exception as e:
            self.q.put(('error', f"Training failed: {e}"))
            return

        # Predict on full dataset (including any extra lines appended)
        self.q.put(('status', "Generating predictions..."))
        X_all_s = scaler_X.transform(X)
        preds_raw = model.predict(X_all_s)

        if is_categorical:
            pred_indices = np.argmax(preds_raw, axis=1)
            try:
                predicted_strings = encoder.inverse_transform(pred_indices)
            except Exception:
                predicted_strings = np.array([str(int(i)) for i in pred_indices])
            output_list = predicted_strings.tolist()
        else:
            preds_unscaled = scaler_y.inverse_transform(preds_raw).flatten() if 'scaler_y' in locals() else preds_raw.flatten()
            output_list = [str(float(x)) for x in preds_unscaled]

        # Save predictions to PK001.txt in same directory as CSV
        out_dir = os.path.dirname(csv_file) or '.'
        out_path = os.path.join(out_dir, 'PK001.txt')
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                for val in output_list:
                    f.write(f"{val}\n")
            self.last_output_path = out_path
            self.q.put(('predictions', output_list))
            self.q.put(('status', f"Done. Predictions saved to {out_path}"))
        except Exception as e:
            self.q.put(('error', f"Failed to save output: {e}"))

        self.q.put(('done', None))

def format_seconds(s):
    if s is None or s != s:
        return ""
    s = int(round(s))
    if s < 60:
        return f"{s}s"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{m}m {sec}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"

if __name__ == "__main__":
    try:
        import pandas  # noqa: F401
    except Exception:
        print("Please install required packages: pip install pandas numpy tensorflow scikit-learn matplotlib")
        raise

    root = tk.Tk()
    style = ttk.Style(root)
    try:
        style.theme_use('clam')
    except Exception:
        pass
    app = App(root)
    root.mainloop()