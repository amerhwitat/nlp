# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 19:59:21 2026

@author: PC1
"""

import os, threading, time, requests, binascii
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox

API_BASE = "https://blockstream.info/api"  # mainnet

INFILE = "burn-addresses-btc.txt"
OUTFILE = "burn-hex.txt"
REQUEST_TIMEOUT = 15

def log(gui, text):
    gui.log_area.configure(state='normal')
    gui.log_area.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {text}\n")
    gui.log_area.see(tk.END)
    gui.log_area.configure(state='disabled')

def get_address_info(addr):
    url = f"{API_BASE}/address/{addr}"
    r = requests.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()

def get_address_txs(addr, page=0):
    url = f"{API_BASE}/address/{addr}/txs"
    r = requests.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()

def get_tx(txid):
    url = f"{API_BASE}/tx/{txid}"
    r = requests.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()

def extract_pubkey_from_vin(vin):
    # vin may contain 'scriptsig' or 'witness'
    # witness: list of hex pushes, last element often pubkey for P2WPKH
    if 'witness' in vin and vin['witness']:
        try:
            last = vin['witness'][-1]
            b = bytes.fromhex(last)
            if len(b) in (33, 65):
                return last
        except Exception:
            pass
    if 'scriptsig' in vin and vin['scriptsig']:
        s = vin['scriptsig']
        try:
            data = bytes.fromhex(s)
            # parse pushes: simple scan for 33/65-byte push near end
            for size in (65, 33):
                if len(data) >= size and data[-size] != 0:
                    candidate = data[-size:]
                    if len(candidate) in (33,65):
                        return candidate.hex()
            # fallback: search for 33/65 byte sequences
            for i in range(len(data)-33):
                if data[i] in (33, 65):
                    ln = data[i]
                    if i+1+ln <= len(data):
                        cand = data[i+1:i+1+ln]
                        if len(cand) in (33,65):
                            return cand.hex()
        except Exception:
            pass
    return None

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Burn Address Checker")
        self.geometry("760x480")
        self.create_widgets()

    def create_widgets(self):
        frm = ttk.Frame(self); frm.pack(fill='both', expand=True, padx=8, pady=8)
        ttk.Label(frm, text="Input file").grid(row=0,column=0,sticky='w')
        self.file_var = tk.StringVar(value=os.path.join(os.getcwd(), INFILE))
        ttk.Entry(frm, textvariable=self.file_var, width=64).grid(row=0,column=1,sticky='w')
        ttk.Button(frm, text="Browse", command=self.browse).grid(row=0,column=2,padx=6)
        ttk.Button(frm, text="Start", command=self.start).grid(row=1,column=0,pady=8)
        self.progress = ttk.Progressbar(frm, length=520, mode='determinate'); self.progress.grid(row=1,column=1,columnspan=2,sticky='w')
        ttk.Label(frm, text="Log").grid(row=2,column=0,sticky='w',pady=(10,0))
        self.log_area = scrolledtext.ScrolledText(frm, height=20, state='disabled'); self.log_area.grid(row=3,column=0,columnspan=3,sticky='nsew')

    def browse(self):
        f = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select input file")
        if f: self.file_var.set(f)

    def start(self):
        t = threading.Thread(target=self.process_file, daemon=True); t.start()

    def process_file(self):
        path = self.file_var.get()
        if not os.path.isfile(path):
            messagebox.showerror("Error", f"File not found: {path}"); return
        with open(path,'r') as fh:
            lines = [ln.strip() for ln in fh.readlines()]
        addrs = [ln for ln in lines if ln and not ln.startswith('#')]
        total = len(addrs)
        if total==0:
            messagebox.showinfo("Info","No addresses found"); return
        out_path = os.path.join(os.path.dirname(path), OUTFILE)
        self.progress['maximum'] = total; self.progress['value']=0
        log(self, f"Processing {total} addresses. Output: {out_path}")
        with open(out_path,'w') as out_f:
            out_f.write("# address ; balance_sats ; pubkey_hex_or_NOT_FOUND ; status\n")
            for i, addr in enumerate(addrs, start=1):
                self.progress['value'] = i-1
                log(self, f"[{i}/{total}] Checking {addr}")
                try:
                    info = get_address_info(addr)
                    cs = info.get('chain_stats',{})
                    funded = cs.get('funded_txo_sum',0)
                    spent = cs.get('spent_txo_sum',0)
                    balance = funded - spent
                except Exception as e:
                    log(self, f"Error fetching balance for {addr}: {e}")
                    out_f.write(f"{addr} ; 0 ; ; ERROR_BALANCE\n")
                    self.progress['value'] = i
                    continue

                pubhex = None
                try:
                    txs = get_address_txs(addr)
                    # iterate txs and inspect inputs for pubkey
                    for tx in txs:
                        txid = tx.get('txid') or tx.get('id')
                        if not txid: continue
                        txfull = get_tx(txid)
                        for vin in txfull.get('vin',[]):
                            pk = extract_pubkey_from_vin(vin)
                            if pk:
                                pubhex = pk
                                break
                        if pubhex: break
                except Exception as e:
                    log(self, f"Warning: failed to fetch txs for {addr}: {e}")

                status = "OK" if pubhex else "NOT_FOUND"
                out_f.write(f"{addr} ; {balance} ; {pubhex or 'NOT_FOUND'} ; {status}\n")
                log(self, f"[{i}/{total}] balance={balance} sats pubkey={'found' if pubhex else 'NOT_FOUND'}")
                self.progress['value'] = i
                time.sleep(0.15)  # gentle pacing
        log(self, "Done processing all addresses")
        messagebox.showinfo("Done", f"Wrote results to {out_path}")

if __name__ == "__main__":
    App().mainloop()