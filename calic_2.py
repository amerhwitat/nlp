# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 18:41:12 2026

@author: PC1
"""

# language: python
"""
BTC Address Checker - Enhanced
Features added:
- Show last N transactions per address (click "View TXs" for a selected address)
- Rate limiting and retry logic with exponential backoff for large lists
- Local blacklist of known burn addresses (file: burn_blacklist.txt) and auto-flagging
Save as: btc_address_checker_enhanced.py
Requirements: Python 3.8+, requests
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import requests
import threading
import csv
import time
from queue import Queue, Empty
from functools import partial

# Configurable defaults
BLOCKSTREAM_API_BASE = "https://blockstream.info/api"
MAX_CONCURRENT_REQUESTS = 5        # concurrency limit (semaphore)
RATE_LIMIT_DELAY = 0.12            # minimal delay between requests per worker (seconds)
MAX_RETRIES = 4
BACKOFF_FACTOR = 1.5               # exponential backoff multiplier
DEFAULT_LAST_N_TXS = 5             # default number of txs to fetch per address
BLACKLIST_FILENAME = "burn_blacklist.txt"

# Helper HTTP request with retries + backoff
def http_get_with_retries(url, timeout=15, max_retries=MAX_RETRIES):
    last_exc = None
    delay = 0.5
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code in (200, 404):
                return r
            # treat other 5xx as retryable
            if 500 <= r.status_code < 600:
                last_exc = Exception(f"Server error {r.status_code}")
            else:
                r.raise_for_status()
        except Exception as e:
            last_exc = e
        if attempt < max_retries:
            time.sleep(delay)
            delay *= BACKOFF_FACTOR
    raise last_exc

class BTCAddressChecker(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BTC Address Checker - Enhanced")
        self.geometry("1000x600")

        # Top frame with controls
        top = tk.Frame(self)
        top.pack(fill=tk.X, padx=8, pady=6)

        self.load_btn = tk.Button(top, text="Load addresses file", command=self.load_file)
        self.load_btn.pack(side=tk.LEFT, padx=4)

        self.load_blacklist_btn = tk.Button(top, text="Load/Reload Blacklist", command=self.load_blacklist)
        self.load_blacklist_btn.pack(side=tk.LEFT, padx=4)

        self.check_btn = tk.Button(top, text="Check addresses", command=self.check_addresses, state=tk.DISABLED)
        self.check_btn.pack(side=tk.LEFT, padx=4)

        self.last_n_label = tk.Label(top, text=f"Last N TXs:")
        self.last_n_label.pack(side=tk.LEFT, padx=(12,2))
        self.last_n_var = tk.IntVar(value=DEFAULT_LAST_N_TXS)
        self.last_n_spin = tk.Spinbox(top, from_=0, to=50, width=4, textvariable=self.last_n_var)
        self.last_n_spin.pack(side=tk.LEFT)

        self.export_btn = tk.Button(top, text="Export CSV", command=self.export_csv, state=tk.DISABLED)
        self.export_btn.pack(side=tk.LEFT, padx=8)

        self.progress_var = tk.StringVar(value="No file loaded")
        tk.Label(top, textvariable=self.progress_var).pack(side=tk.LEFT, padx=12)

        # Middle: Treeview results
        columns = ("address", "balance_sats", "tx_count", "status")
        self.tree = ttk.Treeview(self, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("address", text="Address")
        self.tree.heading("balance_sats", text="Balance (sats)")
        self.tree.heading("tx_count", text="Tx Count")
        self.tree.heading("status", text="Status / Notes")
        self.tree.column("address", width=420)
        self.tree.column("balance_sats", width=120, anchor=tk.E)
        self.tree.column("tx_count", width=80, anchor=tk.CENTER)
        self.tree.column("status", width=320)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        # Right-click or button to view TXs
        bottom = tk.Frame(self)
        bottom.pack(fill=tk.X, padx=8, pady=6)
        self.view_txs_btn = tk.Button(bottom, text="View TXs for Selected", command=self.on_view_txs, state=tk.DISABLED)
        self.view_txs_btn.pack(side=tk.LEFT, padx=4)
        self.clear_btn = tk.Button(bottom, text="Clear Results", command=self.clear_results)
        self.clear_btn.pack(side=tk.LEFT, padx=4)

        # Scrollbar
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        vsb.place(relx=0.985, rely=0.14, relheight=0.73)

        # Data
        self.addresses = []
        self.results = []
        self.blacklist = set()
        self.queue = Queue()
        self.semaphore = threading.Semaphore(MAX_CONCURRENT_REQUESTS)

        # Bind selection
        self.tree.bind("<<TreeviewSelect>>", self.on_selection)

        # Auto-load blacklist file if exists
        self.load_blacklist(silent=True)

    def load_blacklist(self, silent=False):
        try:
            with open(BLACKLIST_FILENAME, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f.readlines()]
            self.blacklist = {l for l in lines if l and not l.startswith("#")}
            if not silent:
                messagebox.showinfo("Blacklist loaded", f"Loaded {len(self.blacklist)} blacklist addresses from {BLACKLIST_FILENAME}")
            self.progress_var.set(f"Blacklist: {len(self.blacklist)} addresses loaded")
        except FileNotFoundError:
            self.blacklist = set()
            if not silent:
                messagebox.showwarning("Blacklist file not found", f"{BLACKLIST_FILENAME} not found. You can create it to mark known burn addresses.")
        except Exception as e:
            messagebox.showerror("Blacklist error", f"Failed to load blacklist: {e}")

    def load_file(self):
        path = filedialog.askopenfilename(title="Open addresses file", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines()]
            self.addresses = [l for l in lines if l and not l.startswith("#")]
            if not self.addresses:
                messagebox.showinfo("No addresses", "No valid addresses found in the file.")
                return
            self.tree.delete(*self.tree.get_children())
            for addr in self.addresses:
                status = "Blacklisted" if addr in self.blacklist else "Queued"
                self.tree.insert("", tk.END, values=(addr, "", "", status))
            self.progress_var.set(f"Loaded {len(self.addresses)} addresses")
            self.check_btn.config(state=tk.NORMAL)
            self.export_btn.config(state=tk.DISABLED)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read file: {e}")

    def check_addresses(self):
        # Lock UI controls
        self.check_btn.config(state=tk.DISABLED)
        self.load_btn.config(state=tk.DISABLED)
        self.load_blacklist_btn.config(state=tk.DISABLED)
        self.export_btn.config(state=tk.DISABLED)
        self.view_txs_btn.config(state=tk.DISABLED)
        self.results = []
        last_n = max(0, int(self.last_n_var.get()))

        # Build worker threads to process queue
        for addr in self.addresses:
            self.queue.put(addr)

        total = len(self.addresses)
        self.progress_var.set(f"Starting checks for {total} addresses...")

        # Start workers
        num_workers = min(MAX_CONCURRENT_REQUESTS, total) or 1
        workers = []
        for i in range(num_workers):
            t = threading.Thread(target=self.worker_process, args=(last_n,), daemon=True)
            t.start()
            workers.append(t)

        # Monitor queue completion in background
        threading.Thread(target=self._monitor_completion, args=(total, workers), daemon=True).start()

    def _monitor_completion(self, total, workers):
        # Wait until queue empty and workers done
        while not self.queue.empty():
            time.sleep(0.5)
        # Wait small grace for workers to finish in-flight
        for w in workers:
            w.join(timeout=0.1)
        self.progress_var.set(f"Done — checked {total} addresses")
        self.load_btn.config(state=tk.NORMAL)
        self.load_blacklist_btn.config(state=tk.NORMAL)
        self.check_btn.config(state=tk.NORMAL)
        self.export_btn.config(state=tk.NORMAL)
        self.view_txs_btn.config(state=tk.NORMAL)

    def worker_process(self, last_n):
        while True:
            try:
                addr = self.queue.get_nowait()
            except Empty:
                break
            # Rate limiting: small delay
            time.sleep(RATE_LIMIT_DELAY)
            with self.semaphore:
                try:
                    self._update_tree_status(addr, "Checking...")
                    balance, tx_count = self.query_blockstream_address(addr)
                    status = "OK"
                    if addr in self.blacklist:
                        status = "Blacklisted"
                    elif balance == 0 and tx_count == 0:
                        status = "No funds, no txs"
                    elif balance == 0:
                        status = "Zero balance"
                    # Fetch last N txs if requested
                    txs = []
                    if last_n > 0:
                        txs = self.get_last_n_txs(addr, n=last_n)
                    self.results.append({
                        "address": addr,
                        "balance_sats": balance,
                        "tx_count": tx_count,
                        "status": status,
                        "txs": txs
                    })
                    self._update_tree_row(addr, balance, tx_count, status)
                except Exception as e:
                    err_msg = f"Error: {e}"
                    self.results.append({
                        "address": addr,
                        "balance_sats": None,
                        "tx_count": None,
                        "status": err_msg,
                        "txs": []
                    })
                    self._update_tree_row(addr, "", "", err_msg)
            self.queue.task_done()

    def query_blockstream_address(self, address):
        url = f"{BLOCKSTREAM_API_BASE}/address/{address}"
        r = http_get_with_retries(url)
        if r.status_code == 200:
            data = r.json()
            chain = data.get("chain_stats", {})
            funded = chain.get("funded_txo_sum", 0)
            spent = chain.get("spent_txo_sum", 0)
            balance = funded - spent
            tx_count = chain.get("tx_count", 0)
            return balance, tx_count
        elif r.status_code == 404:
            return 0, 0
        else:
            r.raise_for_status()

    def get_last_n_txs(self, address, n=5):
        """
        Uses Blockstream API: /address/:address/txs returns last confirmed txs (paginated)
        We'll fetch enough pages to collect up to n txids, and then fetch details for those txids (tx endpoint)
        """
        txs = []
        try:
            page = 1
            collected = 0
            # Blockstream returns latest confirmed txs up to 25 per page
            while collected < n:
                url = f"{BLOCKSTREAM_API_BASE}/address/{address}/txs"
                # Blockstream API supports pagination by /txs/chain? No stable page param in docs; but the endpoint above returns latest up to 25.
                # We'll call once and slice n (simplifying to one call). If more needed, could use /txs/txid etc.
                r = http_get_with_retries(url)
                if r.status_code == 200:
                    data = r.json()
                    if not data:
                        break
                    for tx in data:
                        if collected >= n:
                            break
                        # Minimal TX info: txid, status (block_height, timestamp), vsize, fee
                        txid = tx.get("txid")
                        fee = tx.get("fee")
                        status = tx.get("status", {})
                        block_height = status.get("block_height")
                        block_time = status.get("block_time")
                        # gather inputs/outputs summary counts
                        vin = len(tx.get("vin", []))
                        vout = len(tx.get("vout", []))
                        txs.append({
                            "txid": txid,
                            "fee": fee,
                            "block_height": block_height,
                            "block_time": block_time,
                            "vin": vin,
                            "vout": vout
                        })
                        collected += 1
                    break
                elif r.status_code == 404:
                    break
                else:
                    r.raise_for_status()
            return txs
        except Exception:
            # On any error return empty list rather than failing the whole run
            return []

    def on_selection(self, event):
        selection = self.tree.selection()
        if selection:
            self.view_txs_btn.config(state=tk.NORMAL)
        else:
            self.view_txs_btn.config(state=tk.DISABLED)

    def on_view_txs(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("No selection", "Select an address first.")
            return
        addr = self.tree.item(sel[0], "values")[0]
        # Find results entry
        entry = next((r for r in self.results if r["address"] == addr), None)
        last_n = int(self.last_n_var.get())
        if entry is None:
            messagebox.showinfo("Not fetched yet", "Address not yet checked or no TXs fetched.")
            return
        txs = entry.get("txs", [])
        if last_n > 0 and not txs:
            # Try fetching now synchronously (with retries/backoff)
            self.progress_var.set(f"Fetching last {last_n} txs for {addr} ...")
            self.update_idletasks()
            txs = self.get_last_n_txs(addr, n=last_n)
        # Open a new window showing tx list
        win = tk.Toplevel(self)
        win.title(f"Last {last_n} TXs for {addr}")
        win.geometry("820x400")
        cols = ("txid", "fee", "block_height", "block_time", "vin", "vout")
        tv = ttk.Treeview(win, columns=cols, show="headings")
        for c in cols:
            tv.heading(c, text=c)
            tv.column(c, width=120 if c == "txid" else 100, anchor=tk.CENTER)
        tv.column("txid", width=380)
        tv.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        for tx in txs:
            tid = tx.get("txid")
            fee = tx.get("fee")
            bh = tx.get("block_height")
            bt = tx.get("block_time")
            vin = tx.get("vin")
            vout = tx.get("vout")
            bt_str = str(bt) if bt is not None else ""
            tv.insert("", tk.END, values=(tid, fee, bh, bt_str, vin, vout))
        if not txs:
            tk.Label(win, text="No transactions found or failed to fetch.").pack(pady=8)

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
        path = filedialog.asksaveasfilename(title="Save CSV", defaultextension=".csv", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # Header
                writer.writerow(["address", "balance_sats", "tx_count", "status", "last_txs_count", "txids"])
                for row in self.results:
                    txids = ",".join([t.get("txid", "") for t in row.get("txs", [])])
                    writer.writerow([row.get("address"), row.get("balance_sats"), row.get("tx_count"), row.get("status"), len(row.get("txs", [])), txids])
            messagebox.showinfo("Exported", f"Results exported to {path}")
        except Exception as e:
            messagebox.showerror("Export error", f"Failed to export CSV: {e}")

    def clear_results(self):
        self.tree.delete(*self.tree.get_children())
        self.results = []
        self.progress_var.set("Cleared")
        self.check_btn.config(state=tk.DISABLED)
        self.export_btn.config(state=tk.DISABLED)
        self.view_txs_btn.config(state=tk.DISABLED)
        self.addresses = []

if __name__ == "__main__":
    app = BTCAddressChecker()
    app.mainloop()
