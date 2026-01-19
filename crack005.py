# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 01:50:46 2026

@author: PC1
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import numpy as np
import pandas as pd
import time
import os

# ---------------- Neural Network Definition ---------------- #
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output):
        error = y - output
        d_output = error * self.sigmoid_derivative(output)
        d_hidden = np.dot(d_output, self.W2.T) * self.sigmoid_derivative(self.a1)
        self.W2 += self.learning_rate * np.dot(self.a1.T, d_output)
        self.b2 += self.learning_rate * np.sum(d_output, axis=0, keepdims=True)
        self.W1 += self.learning_rate * np.dot(X.T, d_hidden)
        self.b1 += self.learning_rate * np.sum(d_hidden, axis=0, keepdims=True)
        return np.mean(np.abs(error))

    def train(self, X, y, epochs, progress_callback):
        for epoch in range(epochs):
            output = self.forward(X)
            loss = self.backward(X, y, output)
            if epoch % 10 == 0:
                progress_callback(epoch, loss)
            time.sleep(0.01)  # Simulate computation time


# ---------------- GUI Application ---------------- #
class BTCTrainerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BTC Private Key Neural Network Trainer")
        self.root.geometry("700x500")

        ttk.Label(root, text="BTC Private Key Neural Network", font=("Arial", 16, "bold")).pack(pady=10)

        self.progress = ttk.Progressbar(root, orient="horizontal", length=600, mode="determinate")
        self.progress.pack(pady=10)

        self.log_text = tk.Text(root, height=20, width=80, state="disabled")
        self.log_text.pack(pady=10)

        ttk.Button(root, text="Start Training", command=self.start_training_thread).pack(pady=10)

    def log(self, message):
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")

    def start_training_thread(self):
        thread = threading.Thread(target=self.run_training)
        thread.start()

    def run_training(self):
        try:
            csv_file = "1-million-private-keys-by-btcleak.com.csv"
            txt_file = "burn-addresses-btc.txt"
            output_file = "PK001.txt"

            if not os.path.exists(csv_file):
                messagebox.showerror("Error", f"{csv_file} not found in current directory.")
                return
            if not os.path.exists(txt_file):
                messagebox.showerror("Error", f"{txt_file} not found in current directory.")
                return

            self.log("Loading CSV data...")
            df = pd.read_csv(csv_file, delimiter=';', header=None)
            X_raw = df.iloc[:, -1].astype(str).apply(lambda x: sum(bytearray(x.encode('utf-8'))) % 256 / 255.0).values.reshape(-1, 1)
            y_raw = df.iloc[:, 0].astype(str).apply(lambda x: sum(bytearray(x.encode('utf-8'))) % 256 / 255.0).values.reshape(-1, 1)

            self.log(f"Loaded {len(df)} rows from {csv_file}")

            input_size = 1
            hidden_size = 16
            output_size = 1
            nn = SimpleNN(input_size, hidden_size, output_size, learning_rate=0.05)

            self.progress["maximum"] = 100
            self.log("Starting training...")

            def progress_callback(epoch, loss):
                progress_value = min(100, epoch / 10)
                self.progress["value"] = progress_value
                self.log(f"Epoch {epoch}: Loss = {loss:.6f}")
                self.root.update_idletasks()

            nn.train(X_raw, y_raw, epochs=1000, progress_callback=progress_callback)

            self.log("Training complete.")
            self.progress["value"] = 100

            self.log("Generating predictions for burn addresses...")
            with open(txt_file, "r") as f:
                burn_addresses = f.read().splitlines()

            X_pred = np.array([sum(bytearray(x.encode('utf-8'))) % 256 / 255.0 for x in burn_addresses]).reshape(-1, 1)
            y_pred = nn.forward(X_pred)

            predictions = [str(int(y[0] * 255)) for y in y_pred]
            with open(output_file, "w") as f:
                f.write("\n".join(predictions))

            self.log(f"Predictions saved to {output_file}")
            messagebox.showinfo("Done", f"Process completed successfully.\nResults saved to {output_file}")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.log(f"Error: {e}")


# ---------------- Main ---------------- #
if __name__ == "__main__":
    root = tk.Tk()
    app = BTCTrainerApp(root)
    root.mainloop()
