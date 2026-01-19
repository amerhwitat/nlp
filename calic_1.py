# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 18:39:00 2026

@author: PC1
"""

# language: python
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import requests
import threading
import csv

BLOCKSTREAM_API_BASE = "https://blockstream.info/api"

class BTCAddressChecker(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BTC Address Checker - Safe Viewer")
        self.geometry("900x500")

        # Top frame with buttons
        top = tk.Frame(self)
        top.pack(fill=tk.X, padx=8, pady=6)

        self.load_btn = tk.Button(top, text="Load addresses file", command=self.load_file)
        self.load_btn.pack(side=tk.LEFT, padx=4)

        self.check_btn = tk.Button(top, text="Check addresses", command=self.check_addresses, state=tk.DISABLED)
        self.check_btn.pack(side=tk.LEFT, padx=4)

        self.export_btn = tk.Button(top, text="Export CSV", command=self.export_csv, state=tk.DISABLED)
        self.export_btn.pack(side=tk.LEFT, padx=4)

        # Progress label
        self.progress_var = tk.StringVar(value="No file loaded")
        tk.Label(top, textvariable=self.progress_var).pack(side=tk.LEFT, padx=12)

        # Treeview for results
        columns = ("address", "balance_sats", "tx_count", "status")
        self.tree = ttk.Treeview(self, columns=columns, show="headings")
        self.tree.heading("address", text="Address")
        self.tree.heading("balance_sats", text="Balance (sats)")
        self.tree.heading("tx_count", text="Tx Count")
        self.tree.heading("status", text="Status / Notes")
        self.tree.column("address", width=350)
        self.tree.column("balance_sats", width=120, anchor=tk.E)
        self.tree.column("tx_count", width=80, anchor=tk.CENTER)
        self.tree.column("status", width=280)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        # Scrollbar
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        vsb.place(relx=0.985, rely=0.12, relheight=0.76)

        self.addresses = []
        self.results = []

    def load_file(self):
        path = filedialog.askopenfilename(title="Open addresses file",
                                          filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines()]
            # filter empty lines and comments
            self.addresses = [l for l in lines if l and not l.startswith("#")]
            if not self.addresses:
                messagebox.showinfo("No addresses", "No valid addresses found in the file.")
                return
            self.tree.delete(*self.tree.get_children())
            for addr in self.addresses:
                self.tree.insert("", tk.END, values=(addr, "", "", "Queued"))
            self.progress_var.set(f"Loaded {len(self.addresses)} addresses")
            self.check_btn.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read file: {e}")

    def check_addresses(self):
        self.check_btn.config(state=tk.DISABLED)
        self.load_btn.config(state=tk.DISABLED)
        self.export_btn.config(state=tk.DISABLED)
        self.progress_var.set("Checking addresses...")
        threading.Thread(target=self._worker_check, daemon=True).start()

    def _worker_check(self):
        self.results = []
        total = len(self.addresses)
        for i, addr in enumerate(self.addresses, start=1):
            self._update_tree_status(addr, status="Checking...")
            try:
                balance_sats, tx_count = self.query_blockstream_address(addr)
                status = "OK"
                # simple heuristics
                if balance_sats == 0 and tx_count == 0:
                    status = "No funds, no txs"
                elif balance_sats == 0:
                    status = "Zero balance"
                self.results.append({
                    "address": addr,
                    "balance_sats": balance_sats,
                    "tx_count": tx_count,
                    "status": status
                })
                self._update_tree_row(addr, balance_sats, tx_count, status)
            except Exception as e:
                self.results.append({
                    "address": addr,
                    "balance_sats": None,
                    "tx_count": None,
                    "status": f"Error: {e}"
                })
                self._update_tree_row(addr, "", "", f"Error: {e}")
            self.progress_var.set(f"Checked {i}/{total}")
        self.progress_var.set(f"Done — checked {total} addresses")
        self.load_btn.config(state=tk.NORMAL)
        self.check_btn.config(state=tk.NORMAL)
        self.export_btn.config(state=tk.NORMAL)

    def query_blockstream_address(self, address):
        """
        Query Blockstream API for address info.
        Returns (balance_sats, tx_count).
        """
        # balance in satoshis: sum of utxos - spent? Blockstream provides 'chain_stats' and 'mempool_stats'
        url = f"{BLOCKSTREAM_API_BASE}/address/{address}"
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            # chain_stats contains confirmed transactions and funding/spent sums
            chain = data.get("chain_stats", {})
            funded = chain.get("funded_txo_sum", 0)
            spent = chain.get("spent_txo_sum", 0)
            # current confirmed balance
            balance = funded - spent
            tx_count = chain.get("tx_count", 0)
            return balance, tx_count
        elif r.status_code == 404:
            # address unknown -> treat as zero
            return 0, 0
        else:
            r.raise_for_status()

    def _update_tree_status(self, address, status):
        for iid in self.tree.get_children():
            vals = self.tree.item(iid, "values")
            if vals and vals[0] == address:
                new_vals = (vals[0], vals[1], vals[2], status)
                self.tree.item(iid, values=new_vals)
                break

    def _update_tree_row(self, address, balance, tx_count, status):
        for iid in self.tree.get_children():
            vals = self.tree.item(iid, "values")
            if vals and vals[0] == address:
                bal_str = "" if balance is None else str(balance)
                tx_str = "" if tx_count is None else str(tx_count)
                new_vals = (vals[0], bal_str, tx_str, status)
                self.tree.item(iid, values=new_vals)
                break

    def export_csv(self):
        if not self.results:
            messagebox.showinfo("No data", "No results to export. Run checks first.")
            return
        path = filedialog.asksaveasfilename(title="Save CSV", defaultextension=".csv",
                                            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["address", "balance_sats", "tx_count", "status"])
                writer.writeheader()
                for row in self.results:
                    writer.writerow(row)
            messagebox.showinfo("Exported", f"Results exported to {path}")
        except Exception as e:
            messagebox.showerror("Export error", f"Failed to export CSV: {e}")

if __name__ == "__main__":
    app = BTCAddressChecker()
    app.mainloop()
