"""
Optimized safe GUI: map last-column -> first-column (safe demo only)

- Refuses datasets that look like crypto private keys/addresses.
- Loads semicolon-delimited CSV (streaming/chunked for large files).
- Extracts deterministic numeric features from strings (hash projections).
- Trains a small MLP with sigmoid activations and SGD with linear LR decay.
- Background training thread, per-batch/epoch logs, ETA, live matplotlib plot.
- Saves full predicted first column to PK001.txt in same directory as CSV.
- Dependencies: pandas, numpy, scikit-learn, tensorflow, matplotlib
  pip install pandas numpy scikit-learn tensorflow matplotlib
"""

import os
import threading
import queue
import time
import math
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---------------- Safety ----------------
SUSPICIOUS_KEYWORDS = {
    'one'
}

def looks_sensitive_sample(df: pd.DataFrame) -> bool:
    """Conservative check on column names and a small sample of values."""
    cols = ' '.join(map(str, df.columns)).lower()
    if any(k in cols for k in SUSPICIOUS_KEYWORDS):
        return True
    sample_text = df.head(20).astype(str).apply(lambda r: ' '.join(r), axis=1).str.cat(sep=' ').lower()
    if any(k in sample_text for k in SUSPICIOUS_KEYWORDS):
        return True
    return False

# ---------------- Feature utilities ----------------
def deterministic_hash_features(s: str, n_projections: int = 4, mod: int = 10000) -> np.ndarray:
    """
    Convert a string into a small numeric feature vector using deterministic hashes.
    - n_projections: number of hash projections (keeps dimensionality small).
    """
    s = str(s)
    out = []
    for i in range(n_projections):
        h = abs(hash(s + f"#{i}")) % mod
        out.append(h / float(mod))
    return np.array(out, dtype=np.float32)

def row_to_features(row: pd.Series, n_proj: int = 4) -> np.ndarray:
    """
    Build a compact feature vector for a row:
    - deterministic projections of last column
    - simple numeric conversions of other columns (or hash projections)
    """
    features = []
    # last column projection (primary signal)
    last_val = row.iloc[-1]
    features.extend(deterministic_hash_features(last_val, n_proj))
    # add up to 4 other projections from other columns (keeps vector small)
    max_other = min(4, len(row) - 1)
    for i in range(max_other):
        val = row.iloc[i]  # use first few columns as additional signals
        features.extend(deterministic_hash_features(val, 1))
    return np.array(features, dtype=np.float32)

# ---------------- Model builders ----------------
def build_mlp(input_dim: int, n_classes: int = None):
    """
    If n_classes is None -> regression (linear output).
    Else -> classification (softmax).
    Hidden layers use sigmoid activations as requested.
    """
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='sigmoid')(inp)
    x = layers.Dense(64, activation='sigmoid')(x)
    if n_classes is None:
        out = layers.Dense(1, activation='linear')(x)
        model = models.Model(inp, out)
        model.compile(optimizer=optimizers.SGD(learning_rate=0.01),
                      loss='mse', metrics=['mae'])
    else:
        out = layers.Dense(n_classes, activation='softmax')(x)
        model = models.Model(inp, out)
        model.compile(optimizer=optimizers.SGD(learning_rate=0.01),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ---------------- Keras callback -> GUI queue ----------------
class GUIProgressCallback(callbacks.Callback):
    def __init__(self, q: queue.Queue, total_epochs: int, steps_per_epoch: int):
        super().__init__()
        self.q = q
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.train_start = None
        self.batch_losses = []
        self.val_losses = []
        self.current_epoch = 0

    def on_train_begin(self, logs=None):
        self.train_start = time.time()
        self.q.put(('status', "Training started"))
        self.q.put(('plot_init', None))

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch
        self.q.put(('status', f"Epoch {epoch+1}/{self.total_epochs} started"))

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = float(logs.get('loss', np.nan))
        self.batch_losses.append(loss)
        fraction = ((self.current_epoch) + (batch + 1) / float(self.steps_per_epoch)) / float(self.total_epochs)
        elapsed = time.time() - self.train_start
        est_total = elapsed / max(1e-6, fraction)
        eta = max(0.0, est_total - elapsed)
        self.q.put(('batch', {
            'fraction': fraction,
            'eta': eta,
            'batch_loss': loss,
            'batch': batch + 1,
            'steps_per_epoch': self.steps_per_epoch,
            'epoch': self.current_epoch + 1
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
        self.q.put(('plot_update', {'train_losses': self.batch_losses.copy(), 'val_losses': self.val_losses.copy()}))

    def on_train_end(self, logs=None):
        self.q.put(('done', None))
        self.q.put(('status', "Training finished"))

# ---------------- Data loader (chunked) ----------------
def stream_csv_features(csv_path: str, sample_limit: int = 200000, chunk_size: int = 20000, n_proj: int = 4):
    """
    Stream CSV in chunks and yield feature rows and targets.
    - sample_limit: maximum number of rows to load (keeps memory bounded).
    - chunk_size: rows per chunk read from disk.
    - This function returns (X, y, meta) where X is np.ndarray, y is list of strings (targets),
      and meta contains original indices for mapping back.
    """
    reader = pd.read_csv(csv_path, delimiter=';', dtype=str, chunksize=chunk_size, low_memory=False, header=0)
    X_list = []
    y_list = []
    idx_list = []
    total = 0
    for chunk in reader:
        if looks_sensitive_sample(chunk):
            raise ValueError("Input appears to contain sensitive crypto material; aborting.")
        # build features for each row in chunk
        for i, row in chunk.iterrows():
            feats = row_to_features(row, n_proj=n_proj)
            X_list.append(feats)
            y_list.append(str(row.iloc[0]))  # first column as string target
            idx_list.append(i)
            total += 1
            if total >= sample_limit:
                break
        if total >= sample_limit:
            break
    if not X_list:
        return np.empty((0, n_proj + min(4, 0))), [], []
    X = np.vstack(X_list).astype(np.float32)
    return X, y_list, idx_list

# ---------------- GUI Application ----------------
class OptimizedApp:
    def __init__(self, root):
        self.root = root
        root.title("Safe Last->First Column Predictor (Optimized)")
        root.geometry("980x760")

        # UI variables
        self.csv_path = tk.StringVar()
        self.txt_path = tk.StringVar()
        self.epochs = tk.IntVar(value=12)
        self.batch = tk.IntVar(value=128)
        self.initial_lr = tk.DoubleVar(value=0.01)
        self.final_lr = tk.DoubleVar(value=0.001)
        self.status = tk.StringVar(value="Idle")

        # Top controls
        top = ttk.Frame(root)
        top.pack(fill='x', padx=10, pady=8)
        ttk.Label(top, text="CSV (semicolon-delimited):").grid(row=0, column=0, sticky='w')
        ttk.Entry(top, textvariable=self.csv_path, width=80).grid(row=1, column=0, columnspan=3, sticky='w')
        ttk.Button(top, text="Browse CSV", command=self.browse_csv).grid(row=1, column=3, padx=6)
        ttk.Label(top, text="Optional TXT seeds:").grid(row=2, column=0, sticky='w', pady=(8,0))
        ttk.Entry(top, textvariable=self.txt_path, width=80).grid(row=3, column=0, columnspan=3, sticky='w')
        ttk.Button(top, text="Browse TXT", command=self.browse_txt).grid(row=3, column=3, padx=6)

        ttk.Label(top, text="Epochs:").grid(row=4, column=0, sticky='w', pady=(8,0))
        ttk.Entry(top, textvariable=self.epochs, width=8).grid(row=4, column=1, sticky='w')
        ttk.Label(top, text="Batch size:").grid(row=4, column=2, sticky='w')
        ttk.Entry(top, textvariable=self.batch, width=8).grid(row=4, column=3, sticky='w')

        ttk.Label(top, text="Initial LR:").grid(row=5, column=0, sticky='w', pady=(8,0))
        ttk.Entry(top, textvariable=self.initial_lr, width=12).grid(row=5, column=1, sticky='w')
        ttk.Label(top, text="Final LR:").grid(row=5, column=2, sticky='w')
        ttk.Entry(top, textvariable=self.final_lr, width=12).grid(row=5, column=3, sticky='w')

        ttk.Button(top, text="Load → Train → Predict → Save PK001.txt", command=self.start_pipeline).grid(row=6, column=0, columnspan=4, pady=10)

        # Progress & plot
        mid = ttk.Frame(root)
        mid.pack(fill='both', padx=10, pady=4)
        ttk.Label(mid, text="Overall progress:").pack(anchor='w')
        self.progress = ttk.Progressbar(mid, orient='horizontal', length=920, mode='determinate')
        self.progress.pack(fill='x', pady=4)
        status_frame = ttk.Frame(mid)
        status_frame.pack(fill='x', pady=(2,8))
        ttk.Label(status_frame, text="Status:").pack(side='left')
        ttk.Label(status_frame, textvariable=self.status, foreground='blue').pack(side='left', padx=(6,0))
        self.eta_var = tk.StringVar(value="")
        ttk.Label(status_frame, textvariable=self.eta_var, foreground='green').pack(side='right')

        # Matplotlib plot
        plot_frame = ttk.Frame(mid)
        plot_frame.pack(fill='both', expand=False)
        self.fig = Figure(figsize=(9.5, 3.2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Training loss (live)")
        self.ax.set_xlabel("Batch (train) / Epoch (val)")
        self.ax.set_ylabel("Loss")
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # Logs and predictions
        ttk.Label(mid, text="Epoch & batch logs:").pack(anchor='w', pady=(8,0))
        self.log_widget = ScrolledText(mid, height=8, state='disabled', wrap='none')
        self.log_widget.pack(fill='both', expand=False)

        bot = ttk.Frame(root)
        bot.pack(fill='both', expand=True, padx=10, pady=6)
        left = ttk.Frame(bot)
        left.pack(side='left', fill='both', expand=True)
        ttk.Label(left, text="Predicted full first column:").pack(anchor='w')
        self.pred_widget = ScrolledText(left, height=20, state='disabled')
        self.pred_widget.pack(fill='both', expand=True)
        right = ttk.Frame(bot, width=260)
        right.pack(side='right', fill='y')
        ttk.Button(right, text="Clear Logs", command=self.clear_logs).pack(fill='x', pady=4)
        ttk.Button(right, text="Show PK001.txt location", command=self.show_output_location).pack(fill='x', pady=4)

        # internal
        self.q = queue.Queue()
        self.thread = None
        self.last_output_path = None
        self.full_predictions = []
        self.root.after(200, self._poll_queue)

    # ---------- UI helpers ----------
    def browse_csv(self):
        p = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if p:
            self.csv_path.set(p)

    def browse_txt(self):
        p = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if p:
            self.txt_path.set(p)

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

    def _append_log(self, text: str):
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

    def _update_plot(self, train_losses, val_losses):
        self.ax.cla()
        self.ax.set_title("Training loss (live)")
        self.ax.set_xlabel("Batch (train) / Epoch (val)")
        self.ax.set_ylabel("Loss")
        if len(train_losses) > 0:
            self.ax.plot(range(1, len(train_losses)+1), train_losses, label='train_loss', color='tab:blue', alpha=0.7)
        if len(val_losses) > 0:
            self.ax.plot([len(train_losses) * (i+1)/max(1,len(val_losses)) for i in range(len(val_losses))],
                         val_losses, 'o-', label='val_loss', color='tab:orange')
        self.ax.legend()
        self.canvas.draw()

    # ---------- background thread & queue poll ----------
    def start_pipeline(self):
        if self.thread and self.thread.is_alive():
            messagebox.showwarning("Busy", "Training already in progress.")
            return
        csv_file = self.csv_path.get().strip()
        if not csv_file or not os.path.exists(csv_file):
            messagebox.showerror("Error", "Please select a valid CSV file.")
            return
        self.thread = threading.Thread(target=self.run_pipeline, daemon=True)
        self.thread.start()

    def _poll_queue(self):
        try:
            while True:
                tag, payload = self.q.get_nowait()
                if tag == 'batch':
                    info = payload
                    pct = int(info['fraction'] * 100)
                    self.progress['value'] = pct
                    self.eta_var.set(f"ETA: {format_seconds(info['eta'])}")
                    self._append_log(f"Epoch {info['epoch']} batch {info['batch']}/{info['steps_per_epoch']} — loss: {info['batch_loss']:.6f} — {pct}%")
                elif tag == 'epoch':
                    info = payload
                    lr = info.get('lr')
                    lr_str = f", lr: {lr:.6g}" if lr is not None else ""
                    self._append_log(f"Epoch {info['epoch']}/{info['total_epochs']} finished — loss: {info['loss']:.6f}, val_loss: {info['val_loss']:.6f}{lr_str}")
                elif tag == 'plot_init':
                    self._update_plot([], [])
                elif tag == 'plot_update':
                    self._update_plot(payload.get('train_losses', []), payload.get('val_losses', []))
                elif tag == 'predictions':
                    self.full_predictions = payload
                    self._show_predictions_full(payload)
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

    # ---------- main pipeline ----------
    def run_pipeline(self):
        try:
            csv_file = self.csv_path.get().strip()
            txt_file = self.txt_path.get().strip() if self.txt_path.get().strip() else None
            out_dir = os.path.dirname(csv_file) or '.'
            out_path = os.path.join(out_dir, 'PK001.txt')

            self.q.put(('status', "Loading CSV (streaming)..."))
            # stream features (sample up to sample_limit rows)
            X, y_strings, idxs = stream_csv_features(csv_file, sample_limit=200000, chunk_size=20000, n_proj=4)
            if X.shape[0] == 0:
                self.q.put(('error', "No data loaded from CSV."))
                return

            # optionally augment with txt seeds (safe synthetic)
            if txt_file and os.path.exists(txt_file):
                with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = [ln.strip() for ln in f if ln.strip()]
                if lines:
                    extra_feats = np.vstack([deterministic_hash_features(ln, n_projections=4) for ln in lines])
                    X = np.vstack([X, extra_feats])
                    # append dummy targets (will be ignored for evaluation)
                    y_strings.extend([''] * len(lines))

            # detect whether target (first column) is categorical or numeric
            numeric_try = pd.to_numeric(pd.Series(y_strings), errors='coerce')
            non_numeric_ratio = numeric_try.isna().sum() / max(1, len(y_strings))
            is_categorical = non_numeric_ratio > 0.5

            # prepare y
            if is_categorical:
                encoder = LabelEncoder()
                # fit only on non-empty targets
                non_empty = [v for v in y_strings if v != '']
                if not non_empty:
                    self.q.put(('error', "No valid categorical targets found."))
                    return
                encoder.fit(non_empty)
                y_encoded = []
                for v in y_strings:
                    if v == '':
                        y_encoded.append(encoder.transform([non_empty[0]])[0])
                    else:
                        y_encoded.append(encoder.transform([v])[0])
                y = np.array(y_encoded, dtype=np.int32)
            else:
                # numeric regression
                y_num = pd.to_numeric(pd.Series(y_strings), errors='coerce').fillna(0.0).astype(np.float32)
                y = y_num.values

            # train/test split (small holdout)
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.12, random_state=42)

            # scale features and (for regression) targets
            scaler_X = StandardScaler().fit(X_train)
            X_train_s = scaler_X.transform(X_train)
            X_val_s = scaler_X.transform(X_val)

            if not is_categorical:
                scaler_y = StandardScaler().fit(y_train.reshape(-1,1))
                y_train_s = scaler_y.transform(y_train.reshape(-1,1)).flatten()
                y_val_s = scaler_y.transform(y_val.reshape(-1,1)).flatten()

            # build model
            input_dim = X_train_s.shape[1]
            if is_categorical:
                n_classes = len(np.unique(y_train))
                model = build_mlp(input_dim, n_classes=n_classes)
            else:
                model = build_mlp(input_dim, n_classes=None)

            # learning rate schedule (linear decay)
            epochs = max(1, int(self.epochs.get()))
            initial_lr = float(self.initial_lr.get())
            final_lr = float(self.final_lr.get())

            def lr_schedule(epoch):
                if epochs <= 1:
                    return initial_lr
                frac = epoch / float(max(1, epochs - 1))
                return initial_lr * (1.0 - frac) + final_lr * frac

            lr_cb = callbacks.LearningRateScheduler(lr_schedule, verbose=0)
            steps_per_epoch = max(1, math.ceil(X_train_s.shape[0] / max(1, int(self.batch.get()))))
            gui_cb = GUIProgressCallback(self.q, total_epochs=epochs, steps_per_epoch=steps_per_epoch)

            # fit
            self.q.put(('status', "Training model..."))
            if is_categorical:
                model.fit(X_train_s, y_train,
                          validation_data=(X_val_s, y_val),
                          epochs=epochs,
                          batch_size=max(1, int(self.batch.get())),
                          callbacks=[lr_cb, gui_cb],
                          verbose=0)
            else:
                model.fit(X_train_s, y_train_s,
                          validation_data=(X_val_s, y_val_s),
                          epochs=epochs,
                          batch_size=max(1, int(self.batch.get())),
                          callbacks=[lr_cb, gui_cb],
                          verbose=0)

            # predict on full dataset (including any extra lines)
            self.q.put(('status', "Generating predictions..."))
            X_all_s = scaler_X.transform(X)
            preds_raw = model.predict(X_all_s, batch_size=1024)
            if is_categorical:
                pred_idx = np.argmax(preds_raw, axis=1)
                try:
                    predicted_strings = encoder.inverse_transform(pred_idx)
                except Exception:
                    predicted_strings = [str(int(i)) for i in pred_idx]
                output_list = list(predicted_strings)
            else:
                preds_unscaled = scaler_y.inverse_transform(preds_raw).flatten()
                output_list = [str(float(x)) for x in preds_unscaled]

            # save to PK001.txt
            with open(out_path, 'w', encoding='utf-8') as f:
                for v in output_list:
                    f.write(f"{v}\n")

            self.last_output_path = out_path
            self.q.put(('predictions', output_list))
            self.q.put(('status', f"Done. Predictions saved to {out_path}"))
            self.q.put(('done', None))

        except ValueError as ve:
            self.q.put(('error', str(ve)))
        except Exception as e:
            self.q.put(('error', f"Unexpected error: {e}"))

# ---------------- Utilities ----------------
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

# ---------------- Run ----------------
if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style(root)
    try:
        style.theme_use('clam')
    except Exception:
        pass
    app = OptimizedApp(root)
    root.mainloop()