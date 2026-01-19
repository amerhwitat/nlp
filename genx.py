# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 20:27:11 2026

@author: PC1
"""

#!/usr/bin/env python3
"""
WIF Inspector GUI

- Reads compressed WIF private keys from pk.txt (same directory)
- Reads addresses from addresses.txt (same directory)
- Derives public key, P2PKH and P2WPKH addresses for each private key
- Compares generated addresses to addresses.txt and shows relations in a GUI
- Allows exporting results to CSV
"""

import os
import csv
import hashlib
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from ecdsa import SigningKey, SECP256k1
from bech32 import bech32_encode, convertbits

# Base58 alphabet for WIF decoding
BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

# -------------------------
# Crypto helpers
# -------------------------
def sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()

def ripemd160(b: bytes) -> bytes:
    h = hashlib.new("ripemd160")
    h.update(b)
    return h.digest()

def base58_decode(s: str) -> bytes:
    num = 0
    for c in s:
        num = num * 58 + BASE58_ALPHABET.index(c)
    b = num.to_bytes((num.bit_length() + 7) // 8, "big")
    # restore leading zeros
    pad = 0
    for ch in s:
        if ch == BASE58_ALPHABET[0]:
            pad += 1
        else:
            break
    return b"\x00" * pad + b

def wif_to_hex(wif: str):
    """
    Decode WIF to hex private key and detect compressed flag.
    Raises ValueError on invalid WIF.
    """
    raw = base58_decode(wif.strip())
    if len(raw) < 5:
        raise ValueError("Invalid WIF length")
    payload, checksum = raw[:-4], raw[-4:]
    if sha256(sha256(payload))[:4] != checksum:
        raise ValueError("Invalid WIF checksum")
    if payload[0] != 0x80:
        raise ValueError("Not a mainnet WIF")
    # compressed WIF has extra 0x01 at end of payload
    if len(payload) == 34 and payload[-1] == 0x01:
        return payload[1:-1].hex(), True
    elif len(payload) == 33:
        return payload[1:].hex(), False
    else:
        raise ValueError("Unsupported WIF format")

def privkey_to_pubkey(hex_privkey: str, compressed: bool = True) -> bytes:
    priv = bytes.fromhex(hex_privkey)
    sk = SigningKey.from_string(priv, curve=SECP256k1)
    vk = sk.verifying_key
    x = vk.pubkey.point.x()
    y = vk.pubkey.point.y()
    if compressed:
        prefix = b"\x02" if y % 2 == 0 else b"\x03"
        return prefix + x.to_bytes(32, "big")
    else:
        return b"\x04" + x.to_bytes(32, "big") + y.to_bytes(32, "big")

def pubkey_to_p2pkh(pubkey_bytes: bytes) -> str:
    h160 = ripemd160(sha256(pubkey_bytes))
    versioned = b"\x00" + h160
    checksum = sha256(sha256(versioned))[:4]
    payload = versioned + checksum
    num = int.from_bytes(payload, "big")
    res = ""
    while num > 0:
        num, rem = divmod(num, 58)
        res = BASE58_ALPHABET[rem] + res
    # leading zeros
    pad = 0
    for byte in payload:
        if byte == 0:
            pad += 1
        else:
            break
    return BASE58_ALPHABET[0] * pad + res

def pubkey_to_p2wpkh(pubkey_bytes: bytes) -> str:
    h160 = ripemd160(sha256(pubkey_bytes))
    data = convertbits(h160, 8, 5)
    return bech32_encode("bc", [0] + data)

# -------------------------
# File helpers
# -------------------------
def read_wif_file(path="pk.txt"):
    """
    Read WIF keys from pk.txt, one per line. Returns list of stripped lines.
    """
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines

def read_addresses_file(path="addresses.txt"):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines

# -------------------------
# Analysis
# -------------------------
def analyze_wifs(wif_list, address_list):
    """
    For each WIF, derive pubkey and addresses, compare to address_list.
    Returns list of dicts with keys:
      wif, hex_priv, compressed, pubkey_hex, p2pkh, p2wpkh, matched (bool), matched_addresses (list)
    """
    results = []
    addr_set = set(address_list)
    for wif in wif_list:
        try:
            hex_priv, compressed = wif_to_hex(wif)
            pub = privkey_to_pubkey(hex_priv, compressed=compressed)
            p2pkh = pubkey_to_p2pkh(pub)
            p2wpkh = pubkey_to_p2wpkh(pub)
            matched = False
            matched_addresses = []
            if p2pkh in addr_set:
                matched = True
                matched_addresses.append(p2pkh)
            if p2wpkh in addr_set:
                matched = True
                matched_addresses.append(p2wpkh)
            results.append({
                "wif": wif,
                "hex_priv": hex_priv,
                "compressed": compressed,
                "pubkey_hex": pub.hex(),
                "p2pkh": p2pkh,
                "p2wpkh": p2wpkh,
                "matched": matched,
                "matched_addresses": matched_addresses
            })
        except Exception as e:
            results.append({
                "wif": wif,
                "hex_priv": None,
                "compressed": None,
                "pubkey_hex": None,
                "p2pkh": None,
                "p2wpkh": None,
                "matched": False,
                "matched_addresses": [],
                "error": str(e)
            })
    return results

# -------------------------
# GUI
# -------------------------
class WIFInspectorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("WIF Inspector")
        self.geometry("1100x640")
        self.create_widgets()
        self.wif_list = []
        self.address_list = []
        self.results = []

    def create_widgets(self):
        pad = {"padx": 6, "pady": 6}

        top = ttk.Frame(self)
        top.pack(fill=tk.X, **pad)

        ttk.Button(top, text="Load pk.txt", command=self.load_pk).grid(column=0, row=0, **pad)
        ttk.Button(top, text="Load addresses.txt", command=self.load_addresses).grid(column=1, row=0, **pad)
        ttk.Button(top, text="Analyze", command=self.run_analysis).grid(column=2, row=0, **pad)
        ttk.Button(top, text="Export CSV", command=self.export_csv).grid(column=3, row=0, **pad)
        ttk.Button(top, text="Clear", command=self.clear).grid(column=4, row=0, **pad)

        # Treeview for results
        cols = ("wif", "pubkey", "p2pkh", "p2wpkh", "matched", "matched_addresses", "error")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=24)
        self.tree.heading("wif", text="WIF")
        self.tree.heading("pubkey", text="Public Key (hex)")
        self.tree.heading("p2pkh", text="P2PKH")
        self.tree.heading("p2wpkh", text="P2WPKH")
        self.tree.heading("matched", text="Matched")
        self.tree.heading("matched_addresses", text="Matched Addresses")
        self.tree.heading("error", text="Error")
        self.tree.column("wif", width=220)
        self.tree.column("pubkey", width=260)
        self.tree.column("p2pkh", width=180)
        self.tree.column("p2wpkh", width=180)
        self.tree.column("matched", width=70, anchor=tk.CENTER)
        self.tree.column("matched_addresses", width=160)
        self.tree.column("error", width=200)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0,8))

        # Footer info
        self.status_var = tk.StringVar(value="Load pk.txt and addresses.txt then click Analyze")
        status = ttk.Label(self, textvariable=self.status_var, foreground="blue")
        status.pack(fill=tk.X, padx=8, pady=(0,8))

    def load_pk(self):
        # try default file first
        default = "pk.txt"
        if os.path.exists(default):
            path = default
        else:
            path = filedialog.askopenfilename(title="Select pk.txt", filetypes=[("Text files","*.txt"),("All files","*.*")])
            if not path:
                return
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.wif_list = [l.strip() for l in f if l.strip()]
            self.status_var.set(f"Loaded {len(self.wif_list)} WIF entries from {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def load_addresses(self):
        default = "addresses.txt"
        if os.path.exists(default):
            path = default
        else:
            path = filedialog.askopenfilename(title="Select addresses.txt", filetypes=[("Text files","*.txt"),("All files","*.*")])
            if not path:
                return
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.address_list = [l.strip() for l in f if l.strip()]
            self.status_var.set(f"Loaded {len(self.address_list)} addresses from {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def run_analysis(self):
        if not self.wif_list:
            messagebox.showinfo("No WIFs", "Load pk.txt first.")
            return
        if not self.address_list:
            messagebox.showinfo("No addresses", "Load addresses.txt first.")
            return
        self.status_var.set("Analyzing...")
        self.update_idletasks()
        self.results = analyze_wifs(self.wif_list, self.address_list)
        self.populate_tree()
        matches = sum(1 for r in self.results if r.get("matched"))
        self.status_var.set(f"Analysis complete. {matches} keys matched addresses.")

    def populate_tree(self):
        self.tree.delete(*self.tree.get_children())
        for r in self.results:
            wif = r.get("wif") or ""
            pub = r.get("pubkey_hex") or ""
            p2pkh = r.get("p2pkh") or ""
            p2wpkh = r.get("p2wpkh") or ""
            matched = "Yes" if r.get("matched") else "No"
            matched_addrs = ",".join(r.get("matched_addresses", []))
            error = r.get("error", "")
            self.tree.insert("", "end", values=(wif, pub, p2pkh, p2wpkh, matched, matched_addrs, error))

    def export_csv(self):
        if not self.results:
            messagebox.showinfo("No data", "Run analysis first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files","*.csv")])
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["wif","hex_priv","compressed","pubkey_hex","p2pkh","p2wpkh","matched","matched_addresses","error"])
                for r in self.results:
                    writer.writerow([
                        r.get("wif",""),
                        r.get("hex_priv",""),
                        r.get("compressed",""),
                        r.get("pubkey_hex",""),
                        r.get("p2pkh",""),
                        r.get("p2wpkh",""),
                        r.get("matched",False),
                        ";".join(r.get("matched_addresses",[])),
                        r.get("error","")
                    ])
            messagebox.showinfo("Exported", f"CSV exported to {path}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))

    def clear(self):
        self.wif_list = []
        self.address_list = []
        self.results = []
        self.tree.delete(*self.tree.get_children())
        self.status_var.set("Cleared data")

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    app = WIFInspectorApp()
    app.mainloop()