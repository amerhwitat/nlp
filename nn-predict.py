# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 19:50:10 2026

@author: PC1
"""

import os
import threading
import time
import math
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Filenames in same directory
TRAIN_FILENAME = "Public.txt"
TRAIN_OUTPUT = "public_keys_output.txt"
MODEL_FILENAME = "model.pth"
GENERATED_FILENAME = "Generated.txt"

# Neural net hyperparameters (tweakable)
HIDDEN_SIZES = [512, 256]
LR = 1e-3
BATCH_SIZE = 32
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Utility conversions
def hex_to_bits(hexstr, pad_bits=None):
    """Convert hex string to list of bits (MSB first). Optionally pad to pad_bits (left-pad with zeros)."""
    hexstr = hexstr.strip()
    if hexstr.startswith("0x") or hexstr.startswith("0X"):
        hexstr = hexstr[2:]
    if hexstr == "":
        bits = []
    else:
        try:
            b = bytes.fromhex(hexstr)
        except Exception:
            raise ValueError("Invalid hex string")
        bits = []
        for byte in b:
            for i in range(8)[::-1]:
                bits.append((byte >> i) & 1)
    if pad_bits is not None:
        if len(bits) < pad_bits:
            # left-pad with zeros
            bits = [0] * (pad_bits - len(bits)) + bits
        elif len(bits) > pad_bits:
            # truncate most-significant bits (left side) to fit
            bits = bits[-pad_bits:]
    return bits

def bits_to_hex(bits):
    """Convert list of bits (MSB first) to hex string (no 0x), length in hex chars = ceil(len(bits)/4)."""
    if len(bits) % 8 != 0:
        # pad right with zeros to full bytes
        pad = 8 - (len(bits) % 8)
        bits = bits + [0] * pad
    b = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | (bits[i + j] & 1)
        b.append(byte)
    return b.hex()

# Simple feedforward network
class BitNet(nn.Module):
    def __init__(self, input_size, output_size=256, hidden_sizes=None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = HIDDEN_SIZES
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # logits for BCEWithLogitsLoss

# GUI App
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bit-to-256 Neural Trainer")
        self.geometry("820x560")
        self.create_widgets()
        self.model = None
        self.input_size = None

    def create_widgets(self):
        frm = ttk.Frame(self)
        frm.pack(fill='both', expand=True, padx=10, pady=10)

        ttk.Label(frm, text="Training file (semicolon delimited)").grid(row=0, column=0, sticky='w')
        self.train_file_var = tk.StringVar(value=os.path.join(os.getcwd(), TRAIN_FILENAME))
        ttk.Entry(frm, textvariable=self.train_file_var, width=70).grid(row=0, column=1, sticky='w')
        ttk.Button(frm, text="Browse", command=self.browse_train).grid(row=0, column=2, padx=6)

        ttk.Button(frm, text="Prepare and Train", command=self.start_train).grid(row=1, column=0, pady=8)
        self.progress = ttk.Progressbar(frm, length=520, mode='determinate')
        self.progress.grid(row=1, column=1, columnspan=2, sticky='w')

        ttk.Label(frm, text="Load file to generate predictions").grid(row=2, column=0, sticky='w', pady=(10,0))
        self.predict_file_var = tk.StringVar(value=os.path.join(os.getcwd(), "Input.txt"))
        ttk.Entry(frm, textvariable=self.predict_file_var, width=70).grid(row=2, column=1, sticky='w')
        ttk.Button(frm, text="Browse", command=self.browse_predict).grid(row=2, column=2, padx=6)

        ttk.Button(frm, text="Generate Predictions", command=self.start_generate).grid(row=3, column=0, pady=8)

        ttk.Label(frm, text="Model status").grid(row=4, column=0, sticky='w', pady=(10,0))
        self.model_status = tk.StringVar(value="No model loaded")
        ttk.Label(frm, textvariable=self.model_status).grid(row=4, column=1, sticky='w')

        ttk.Label(frm, text="Log").grid(row=5, column=0, sticky='w', pady=(10,0))
        self.log_area = scrolledtext.ScrolledText(frm, height=18, state='disabled')
        self.log_area.grid(row=6, column=0, columnspan=3, sticky='nsew')

    def browse_train(self):
        f = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select training file")
        if f:
            self.train_file_var.set(f)

    def browse_predict(self):
        f = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select input file for generation")
        if f:
            self.predict_file_var.set(f)

    def log(self, text):
        self.log_area.configure(state='normal')
        self.log_area.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {text}\n")
        self.log_area.see(tk.END)
        self.log_area.configure(state='disabled')

    def start_train(self):
        t = threading.Thread(target=self.prepare_and_train, daemon=True)
        t.start()

    def prepare_and_train(self):
        path = self.train_file_var.get()
        if not os.path.isfile(path):
            messagebox.showerror("Error", f"Training file not found: {path}")
            self.log(f"Training file not found: {path}")
            return

        self.log("Reading training file")
        # Read lines
        try:
            with open(path, 'r') as fh:
                raw_lines = [ln.rstrip("\n") for ln in fh.readlines()]
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read training file: {e}")
            self.log(f"Failed to read training file: {e}")
            return

        # Parse lines: expect semicolon delimited, first field target hex, second field input hex
        samples = []
        for ln in raw_lines:
            ln = ln.strip()
            if not ln or ln.startswith('#'):
                continue
            parts = [p.strip() for p in ln.split(';')]
            if len(parts) < 2:
                self.log(f"Skipping malformed line: {ln[:80]}...")
                continue
            target_hex = parts[0]
            input_hex = parts[1]
            samples.append((target_hex, input_hex))

        if len(samples) == 0:
            messagebox.showinfo("Info", "No valid samples found in training file.")
            self.log("No valid samples found in training file.")
            return

        # Determine input size (bits) as max length among input hex fields
        input_bit_lengths = []
        for _, inp in samples:
            try:
                bits = hex_to_bits(inp)
            except Exception:
                bits = []
            input_bit_lengths.append(len(bits))
        max_input_bits = max(input_bit_lengths)
        if max_input_bits == 0:
            messagebox.showerror("Error", "No valid hex inputs found in second field.")
            self.log("No valid hex inputs found in second field.")
            return

        self.input_size = max_input_bits
        self.log(f"Detected input bit size {self.input_size}. Preparing dataset of {len(samples)} samples.")

        # Build dataset arrays
        X = np.zeros((len(samples), self.input_size), dtype=np.float32)
        Y = np.zeros((len(samples), 256), dtype=np.float32)  # targets fixed to 256 bits

        for i, (t_hex, in_hex) in enumerate(samples):
            try:
                in_bits = hex_to_bits(in_hex, pad_bits=self.input_size)
            except Exception:
                in_bits = [0] * self.input_size
            X[i, :] = np.array(in_bits, dtype=np.float32)

            # target: convert first field to 256 bits (left-pad with zeros or truncate MSBs)
            try:
                tgt_bits = hex_to_bits(t_hex, pad_bits=256)
            except Exception:
                tgt_bits = [0] * 256
            Y[i, :] = np.array(tgt_bits, dtype=np.float32)

        # Convert to torch tensors
        X_t = torch.from_numpy(X).to(DEVICE)
        Y_t = torch.from_numpy(Y).to(DEVICE)

        # Create model
        self.model = BitNet(input_size=self.input_size, output_size=256).to(DEVICE)
        self.model_status.set("Model created")
        self.log("Model created. Starting training.")

        # Training loop
        optimizer = optim.Adam(self.model.parameters(), lr=LR)
        criterion = nn.BCEWithLogitsLoss()

        dataset = torch.utils.data.TensorDataset(X_t, Y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        self.progress['maximum'] = EPOCHS
        for epoch in range(1, EPOCHS + 1):
            self.model.train()
            epoch_loss = 0.0
            batches = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batches += 1
            avg_loss = epoch_loss / max(1, batches)
            self.log(f"Epoch {epoch}/{EPOCHS} - loss {avg_loss:.6f}")
            self.progress['value'] = epoch
        self.progress['value'] = 0

        # Save model
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'input_size': self.input_size
            }, MODEL_FILENAME)
            self.log(f"Model saved to {MODEL_FILENAME}")
            self.model_status.set(f"Trained model saved")
        except Exception as e:
            self.log(f"Failed to save model: {e}")

        # Optionally write training outputs line-by-line to TRAIN_OUTPUT
        try:
            with open(TRAIN_OUTPUT, 'w') as out_f:
                out_f.write("# Format: target_hex ; input_hex ; predicted_target_hex ; status\n")
                self.model.eval()
                with torch.no_grad():
                    logits = self.model(X_t)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    for i, (t_hex, in_hex) in enumerate(samples):
                        bits_pred = (probs[i] >= 0.5).astype(int).tolist()
                        pred_hex = bits_to_hex(bits_pred)
                        out_f.write(f"{t_hex} ; {in_hex} ; {pred_hex} ; OK\n")
            self.log(f"Wrote training predictions to {TRAIN_OUTPUT}")
        except Exception as e:
            self.log(f"Failed to write training output file: {e}")

        messagebox.showinfo("Training complete", f"Training finished. Model saved to {MODEL_FILENAME}")

    def start_generate(self):
        t = threading.Thread(target=self.generate_from_file, daemon=True)
        t.start()

    def generate_from_file(self):
        if self.model is None:
            # try to load model from disk
            if os.path.isfile(MODEL_FILENAME):
                try:
                    ckpt = torch.load(MODEL_FILENAME, map_location=DEVICE)
                    self.input_size = ckpt.get('input_size', None)
                    self.model = BitNet(input_size=self.input_size, output_size=256).to(DEVICE)
                    self.model.load_state_dict(ckpt['model_state_dict'])
                    self.model.eval()
                    self.model_status.set("Model loaded from disk")
                    self.log(f"Loaded model from {MODEL_FILENAME}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load model: {e}")
                    self.log(f"Failed to load model: {e}")
                    return
            else:
                messagebox.showerror("Error", "No trained model available. Train first or place model.pth in directory.")
                self.log("No trained model available.")
                return

        path = self.predict_file_var.get()
        if not os.path.isfile(path):
            messagebox.showerror("Error", f"Input file not found: {path}")
            self.log(f"Input file not found: {path}")
            return

        # Read input lines
        try:
            with open(path, 'r') as fh:
                raw_lines = [ln.rstrip("\n") for ln in fh.readlines()]
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read input file: {e}")
            self.log(f"Failed to read input file: {e}")
            return

        # Prepare output file
        out_path = os.path.join(os.path.dirname(path), GENERATED_FILENAME)
        try:
            out_f = open(out_path, 'w')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open output file: {e}")
            self.log(f"Failed to open output file: {e}")
            return

        out_f.write("# Format: predicted_first_field_hex ; original_second_field_hex ; status\n")

        # Process each line one-by-one and write immediately
        total = sum(1 for ln in raw_lines if ln.strip() and not ln.strip().startswith('#'))
        self.progress['maximum'] = max(1, total)
        processed = 0

        for ln in raw_lines:
            ln = ln.strip()
            if not ln or ln.startswith('#'):
                continue
            parts = [p.strip() for p in ln.split(';')]
            if len(parts) < 2:
                self.log(f"Skipping malformed line: {ln[:80]}...")
                continue
            input_hex = parts[1]
            processed += 1
            self.progress['value'] = processed - 1
            self.log(f"[{processed}/{total}] Generating for input {input_hex[:40]}...")

            try:
                in_bits = hex_to_bits(input_hex, pad_bits=self.input_size)
            except Exception:
                in_bits = [0] * self.input_size

            x = torch.tensor([in_bits], dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.sigmoid(logits).cpu().numpy()[0]
                bits_pred = (probs >= 0.5).astype(int).tolist()
                pred_hex = bits_to_hex(bits_pred)

            try:
                out_f.write(f"{pred_hex} ; {input_hex} ; OK\n")
            except Exception as e:
                self.log(f"Failed to write prediction line: {e}")

            self.progress['value'] = processed

        out_f.close()
        self.progress['value'] = 0
        self.log(f"Finished generation. Wrote predictions to {out_path}")
        messagebox.showinfo("Done", f"Generated predictions written to {out_path}")

if __name__ == "__main__":
    app = App()
    app.mainloop()