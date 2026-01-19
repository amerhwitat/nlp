# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 18:48:14 2026

@author: PC1
"""

import hashlib
import hmac
import struct
import requests
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from ecdsa import SigningKey, SECP256k1
from ecdsa.util import sigencode_der
from bech32 import bech32_encode, convertbits

# Base58 alphabet
BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

def sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()

def ripemd160(b: bytes) -> bytes:
    h = hashlib.new("ripemd160")
    h.update(b)
    return h.digest()

def base58_encode(b: bytes) -> str:
    num = int.from_bytes(b, "big")
    enc = ""
    while num > 0:
        num, rem = divmod(num, 58)
        enc = BASE58_ALPHABET[rem] + enc
    pad = 0
    for byte in b:
        if byte == 0:
            pad += 1
        else:
            break
    return BASE58_ALPHABET[0] * pad + enc

def base58check_encode(payload: bytes) -> str:
    checksum = sha256(sha256(payload))[:4]
    return base58_encode(payload + checksum)

def hex_to_wif(hex_privkey: str, compressed: bool = True) -> str:
    privkey_bytes = bytes.fromhex(hex_privkey)
    if len(privkey_bytes) != 32:
        raise ValueError("Private key must be 32 bytes (64 hex chars)")
    prefix = b"\x80"
    payload = prefix + privkey_bytes
    if compressed:
        payload += b"\x01"
    return base58check_encode(payload)

def privkey_to_pubkey(hex_privkey: str, compressed: bool = True) -> bytes:
    privkey_bytes = bytes.fromhex(hex_privkey)
    sk = SigningKey.from_string(privkey_bytes, curve=SECP256k1)
    vk = sk.verifying_key
    x = vk.pubkey.point.x()
    y = vk.pubkey.point.y()
    if compressed:
        prefix = b"\x02" if y % 2 == 0 else b"\x03"
        return prefix + x.to_bytes(32, "big")
    else:
        return b"\x04" + x.to_bytes(32, "big") + y.to_bytes(32, "big")

def pubkey_to_p2pkh_address(pubkey_bytes: bytes) -> str:
    h160 = ripemd160(sha256(pubkey_bytes))
    versioned = b"\x00" + h160
    return base58check_encode(versioned)

def pubkey_to_p2wpkh_address(pubkey_bytes: bytes) -> str:
    # returns bech32 (native segwit) address
    h160 = ripemd160(sha256(pubkey_bytes))
    data = convertbits(h160, 8, 5)
    return bech32_encode("bc", [0] + data)

def fetch_transactions_blockstream(address: str, limit: int = 25):
    """
    Fetch recent transactions for an address using Blockstream public API.
    This is read-only and uses a public endpoint. No private keys are sent.
    """
    try:
        url = f"https://blockstream.info/api/address/{address}/txs"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        txs = resp.json()
        return txs[:limit]
    except Exception as e:
        raise RuntimeError(f"Failed to fetch transactions: {e}")

# -------------------------
# GUI
# -------------------------
class KeyInspectorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bitcoin Key Inspector (Safe, Read-only)")
        self.geometry("900x600")
        self.create_widgets()

    def create_widgets(self):
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        # Input
        ttk.Label(frm, text="Enter your private key (hex, 64 chars):").grid(column=0, row=0, sticky=tk.W)
        self.priv_entry = ttk.Entry(frm, width=80)
        self.priv_entry.grid(column=0, row=1, columnspan=3, sticky=tk.W)

        self.compressed_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="Compressed", variable=self.compressed_var).grid(column=3, row=1, sticky=tk.W)

        ttk.Button(frm, text="Derive", command=self.on_derive).grid(column=0, row=2, pady=8, sticky=tk.W)

        # Results area
        ttk.Label(frm, text="Derived values:").grid(column=0, row=3, sticky=tk.W)
        self.results = scrolledtext.ScrolledText(frm, width=110, height=12)
        self.results.grid(column=0, row=4, columnspan=4, pady=6)

        # Address lookup
        ttk.Label(frm, text="Lookup transactions for address:").grid(column=0, row=5, sticky=tk.W)
        self.addr_entry = ttk.Entry(frm, width=60)
        self.addr_entry.grid(column=0, row=6, sticky=tk.W)
        ttk.Button(frm, text="Fetch TXs", command=self.on_fetch_txs).grid(column=1, row=6, sticky=tk.W)

        ttk.Label(frm, text="Transactions:").grid(column=0, row=7, sticky=tk.W)
        self.txs_area = scrolledtext.ScrolledText(frm, width=110, height=12)
        self.txs_area.grid(column=0, row=8, columnspan=4, pady=6)

        # Security note
        note = ("Security note: Do not paste private keys you do not own. Keep keys local. "
                "This tool does not transmit private keys anywhere. If you need production-grade "
                "security, use an offline signer or hardware wallet.")
        ttk.Label(frm, text=note, foreground="red", wraplength=800).grid(column=0, row=9, columnspan=4, pady=8, sticky=tk.W)

    def on_derive(self):
        self.results.delete("1.0", tk.END)
        hex_priv = self.priv_entry.get().strip()
        if not hex_priv:
            messagebox.showinfo("Input required", "Please enter a 64-character hex private key you own.")
            return
        try:
            if len(hex_priv) != 64:
                raise ValueError("Private key must be 64 hex characters (32 bytes).")
            compressed = self.compressed_var.get()
            wif = hex_to_wif(hex_priv, compressed=compressed)
            pubkey = privkey_to_pubkey(hex_priv, compressed=compressed)
            p2pkh = pubkey_to_p2pkh_address(pubkey)
            p2wpkh = pubkey_to_p2wpkh_address(pubkey)
            out = []
            out.append(f"Private key (hex): {hex_priv}")
            out.append(f"WIF: {wif}")
            out.append(f"Compressed: {compressed}")
            out.append(f"Public key: {pubkey.hex()}")
            out.append(f"P2PKH address (legacy): {p2pkh}")
            out.append(f"P2WPKH address (bech32): {p2wpkh}")
            self.results.insert(tk.END, "\n".join(out))
            # Pre-fill address lookup with P2WPKH
            self.addr_entry.delete(0, tk.END)
            self.addr_entry.insert(0, p2wpkh)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_fetch_txs(self):
        self.txs_area.delete("1.0", tk.END)
        addr = self.addr_entry.get().strip()
        if not addr:
            messagebox.showinfo("Input required", "Please enter an address to fetch transactions for.")
            return
        try:
            txs = fetch_transactions_blockstream(addr)
            if not txs:
                self.txs_area.insert(tk.END, "No transactions found or address not recognized by API.")
                return
            for tx in txs:
                txid = tx.get("txid")
                fee = tx.get("fee")
                status = tx.get("status", {})
                block_height = status.get("block_height")
                block_time = status.get("block_time")
                self.txs_area.insert(tk.END, f"TXID: {txid}\n")
                self.txs_area.insert(tk.END, f"  Block: {block_height}  Time: {block_time}  Fee: {fee}\n")
                self.txs_area.insert(tk.END, "-"*80 + "\n")
        except Exception as e:
            messagebox.showerror("Fetch error", str(e))

if __name__ == "__main__":
    app = KeyInspectorApp()
    app.mainloop()