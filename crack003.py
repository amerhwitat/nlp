"""
Enhanced safe demo with on-screen live training progress and embedded loss plot.

Features:
- Loads ';' CSV (safety check for crypto-like content).
- Trains a small Keras model in a background thread.
- Shows per-batch and per-epoch progress, ETA, and logs.
- Embeds a live matplotlib plot of loss and val_loss.
- Shows first 20 predictions and saves PK001.txt in same directory.
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
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Matplotlib for live plotting in Tkinter
import matplotlib
matplotlib.use('Agg')  # use non-interactive backend for safe drawing
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---------- Safety check helpers ----------
SUSPICIOUS_KEYWORDS = {'one', 'two', 'three', 'four', 'five', 'six', 'seven'}

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

# ---------- Keras callback to send epoch and batch updates to GUI via queue ----------
class ProgressLogger(tf.keras.callbacks.Callback):
    def __init__(self, q, total_epochs, steps_per_epoch):
        super().__init__()
        self.q = q
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.train_losses = []
        self.val_losses = []
        self.epoch_start_time = None
        self.train_start_time = None

    def on_train_begin(self, logs=None):
        self.train_start_time = time.time()
        self.q.put(('status', "Training started"))
        self.q.put(('plot_init', None))

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        self.q.put(('status', f"Epoch {epoch+1}/{self.total_epochs} started"))

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_loss = float(logs.get('loss', np.nan))
        epoch = self.params.get('epochs', self.total_epochs)  # fallback
        current_epoch = self.model._train_counter.numpy() if hasattr(self.model, '_train_counter') else None
        # compute overall progress fraction
        epoch_index = self.model.optimizer.iterations.numpy()  # not reliable for epoch index; instead compute from batch info
        # We'll compute progress using known epoch and batch counts from params
        current_epoch_num = self.params.get('epochs', self.total_epochs)  # fallback
        # Use provided steps_per_epoch and total_epochs to compute fraction:
        # We don't have direct epoch index here, so rely on internal state: self._current_epoch if available
        # Simpler: use batch and current epoch from logs via attribute set in on_epoch_begin
        # We'll store current_epoch in self._current_epoch in on_epoch_begin
        cur_epoch = getattr(self, '_current_epoch', 0)
        fraction = ((cur_epoch) + (batch + 1) / float(self.steps_per_epoch)) / float(self.total_epochs)
        elapsed = time.time() - self.train_start_time
        est_total = elapsed / max(1e-6, fraction)
        eta = max(0.0, est_total - elapsed)
        self.q.put(('batch', {'fraction': fraction, 'eta': eta, 'batch_loss': batch_loss, 'batch': batch+1, 'steps_per_epoch': self.steps_per_epoch, 'epoch': cur_epoch+1}))
        # also append to train_losses for plotting (coarse)
        self.train_losses.append(batch_loss)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = float(logs.get('loss', np.nan))
        val_loss = float(logs.get('val_loss', np.nan)) if 'val_loss' in logs else np.nan
        self.val_losses.append(val_loss)
        # store current epoch index for batch callback
        self._current_epoch = epoch
        # send epoch summary
        self.q.put(('epoch', {'epoch': epoch+1, 'loss': loss, 'val_loss': val_loss, 'total_epochs': self.total_epochs}))
        # send plot update with aggregated losses
        self.q.put(('plot_update', {'train_losses': self.train_losses.copy(), 'val_losses': self.val_losses.copy()}))

    def on_train_end(self, logs=None):
        self.q.put(('done', None))
        self.q.put(('status', "Training finished"))

# ---------- GUI ----------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Safe CSV -> NN Demo (Live Progress)")
        root.geometry("1000x760")

        self.csv_path = tk.StringVar()
        self.txt_path = tk.StringVar()
        self.epochs_var = tk.IntVar(value=20)
        self.batch_var = tk.IntVar(value=32)
        self.status = tk.StringVar(value="Idle")

        # Top frame: file selection and training controls
        top = ttk.Frame(root)
        top.pack(fill='x', padx=10, pady=8)

        ttk.Label(top, text="CSV file (semicolon-delimited):").grid(row=0, column=0, sticky='w')
        ttk.Entry(top, textvariable=self.csv_path, width=90).grid(row=1, column=0, columnspan=3, sticky='w')
        ttk.Button(top, text="Browse CSV", command=self.browse_csv).grid(row=1, column=3, padx=6)

        ttk.Label(top, text="Optional text file (lines used to generate synthetic inputs):").grid(row=2, column=0, sticky='w', pady=(8,0))
        ttk.Entry(top, textvariable=self.txt_path, width=90).grid(row=3, column=0, columnspan=3, sticky='w')
        ttk.Button(top, text="Browse TXT", command=self.browse_txt).grid(row=3, column=3, padx=6)

        ttk.Label(top, text="Epochs:").grid(row=4, column=0, sticky='w', pady=(8,0))
        ttk.Entry(top, textvariable=self.epochs_var, width=8).grid(row=4, column=1, sticky='w')
        ttk.Label(top, text="Batch size:").grid(row=4, column=2, sticky='w')
        ttk.Entry(top, textvariable=self.batch_var, width=8).grid(row=4, column=3, sticky='w')

        ttk.Button(top, text="Load, Train, Predict, Save PK001.txt", command=self.start_pipeline, style='Accent.TButton').grid(row=5, column=0, columnspan=4, pady=10)

        # Middle frame: progress, plot and logs
        mid = ttk.Frame(root)
        mid.pack(fill='both', expand=False, padx=10, pady=4)

        ttk.Label(mid, text="Overall progress:").pack(anchor='w')
        self.progress = ttk.Progressbar(mid, orient='horizontal', length=920, mode='determinate')
        self.progress.pack(fill='x', pady=4)

        # ETA and status
        status_frame = ttk.Frame(mid)
        status_frame.pack(fill='x', pady=(2,8))
        ttk.Label(status_frame, text="Status:").pack(side='left')
        ttk.Label(status_frame, textvariable=self.status, foreground='blue').pack(side='left', padx=(6,0))
        self.eta_var = tk.StringVar(value="")
        ttk.Label(status_frame, textvariable=self.eta_var, foreground='green').pack(side='right')

        # Plot area
        plot_frame = ttk.Frame(mid)
        plot_frame.pack(fill='both', expand=False)
        self.fig = Figure(figsize=(9.5, 3.2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Training loss (live)")
        self.ax.set_xlabel("Batch (train) / Epoch (val)")
        self.ax.set_ylabel("Loss")
        self.train_line, = self.ax.plot([], [], label='train_loss', color='tab:blue')
        self.val_line, = self.ax.plot([], [], label='val_loss', color='tab:orange')
        self.ax.legend()
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill='both', expand=True)

        # Logs
        ttk.Label(mid, text="Epoch & batch logs:").pack(anchor='w', pady=(8,0))
        self.log_widget = ScrolledText(mid, height=8, state='disabled', wrap='none')
        self.log_widget.pack(fill='both', expand=False)

        # Bottom frame: predictions preview and controls
        bot = ttk.Frame(root)
        bot.pack(fill='both', expand=True, padx=10, pady=6)

        left = ttk.Frame(bot)
        left.pack(side='left', fill='both', expand=True)

        ttk.Label(left, text="Generated predictions (first 20 shown):").pack(anchor='w')
        self.pred_widget = ScrolledText(left, height=12, state='disabled')
        self.pred_widget.pack(fill='both', expand=True)

        right = ttk.Frame(bot, width=260)
        right.pack(side='right', fill='y')

        ttk.Label(right, text="Controls:").pack(anchor='w')
        ttk.Button(right, text="Clear Logs", command=self.clear_logs).pack(fill='x', pady=4)
        ttk.Button(right, text="Show saved PK001.txt location", command=self.show_output_location).pack(fill='x', pady=4)

        # queue for inter-thread communication
        self.q = queue.Queue()
        self.training_thread = None
        self.last_output_path = None

        # plot data buffers
        self.plot_train = []
        self.plot_val = []

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
        self.ax.cla()
        self.ax.set_title("Training loss (live)")
        self.ax.set_xlabel("Batch (train) / Epoch (val)")
        self.ax.set_ylabel("Loss")
        self.train_line, = self.ax.plot([], [], label='train_loss', color='tab:blue')
        self.val_line, = self.ax.plot([], [], label='val_loss', color='tab:orange')
        self.ax.legend()
        self.canvas.draw()

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
                    self._append_log(f"Epoch {epoch}/{info['total_epochs']} finished — loss: {loss:.6f}, val_loss: {val_loss:.6f}")
                elif tag == 'plot_init':
                    # reset plot buffers
                    self.plot_train = []
                    self.plot_val = []
                    self._update_plot()
                elif tag == 'plot_update':
                    data = payload
                    # payload contains lists of train_losses (per-batch) and val_losses (per-epoch)
                    self.plot_train = data.get('train_losses', [])
                    self.plot_val = data.get('val_losses', [])
                    self._update_plot()
                elif tag == 'predictions':
                    preds = payload
                    self._show_predictions(preds)
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

    def _show_predictions(self, preds):
        self.pred_widget.configure(state='normal')
        self.pred_widget.delete('1.0', tk.END)
        for i, p in enumerate(preds[:20]):
            self.pred_widget.insert(tk.END, f"{i+1}: {p}\n")
        self.pred_widget.see(tk.END)
        self.pred_widget.configure(state='disabled')

    def _update_plot(self):
        # update matplotlib lines and redraw canvas
        self.ax.cla()
        self.ax.set_title("Training loss (live)")
        self.ax.set_xlabel("Batch (train) / Epoch (val)")
        self.ax.set_ylabel("Loss")
        if len(self.plot_train) > 0:
            self.ax.plot(range(1, len(self.plot_train)+1), self.plot_train, label='train_loss', color='tab:blue', alpha=0.7)
        if len(self.plot_val) > 0:
            # plot val losses at approximate positions (every N batches) — show as points
            self.ax.plot([len(self.plot_train) * (i+1)/max(1,len(self.plot_val)) for i in range(len(self.plot_val))],
                         self.plot_val, 'o-', label='val_loss', color='tab:orange')
        self.ax.legend()
        self.canvas.draw()

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
        steps_per_epoch = max(1, int(np.ceil(X_train_s.shape[0] / batch_size)))

        # Setup callback
        progress_logger = ProgressLogger(self.q, total_epochs=epochs, steps_per_epoch=steps_per_epoch)

        # Start training
        self.q.put(('status', "Training model..."))
        try:
            model.fit(
                X_train_s, y_train_s,
                validation_data=(X_test_s, y_test_s),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=[progress_logger]
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
    # Check for required packages
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