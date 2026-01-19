# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 18:09:17 2026

@author: PC1
"""

import tkinter as tk
from tkinter import ttk, messagebox
import secrets, hashlib, time, random, requests, threading

from ecdsa import SigningKey, SECP256k1
from bech32 import bech32_encode, convertbits

# =====================================================
# Crypto
# =====================================================

class Hash:
    @staticmethod
    def sha256(b): return hashlib.sha256(b).digest()
    @staticmethod
    def ripemd160(b):
        h = hashlib.new("ripemd160")
        h.update(b)
        return h.digest()

# =====================================================
# Base58 (Testnet)
# =====================================================

class Base58:
    ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

    @staticmethod
    def encode_check(payload):
        checksum = Hash.sha256(Hash.sha256(payload))[:4]
        return Base58.encode(payload + checksum)

    @staticmethod
    def encode(b):
        n = int.from_bytes(b, "big")
        s = ""
        while n:
            n, r = divmod(n, 58)
            s = Base58.ALPHABET[r] + s
        pad = len(b) - len(b.lstrip(b"\x00"))
        return Base58.ALPHABET[0] * pad + s

# =====================================================
# Bitcoin Key (TESTNET)
# =====================================================

class BitcoinKey:
    def __init__(self):
        self.priv = secrets.token_bytes(32)
        self.sk = SigningKey.from_string(self.priv, curve=SECP256k1)

    def priv_hex(self):
        return self.priv.hex()

    def pubkey(self):
        p = self.sk.verifying_key.pubkey.point
        x = p.x().to_bytes(32, "big")
        y = p.y()
        return (b"\x02" if y % 2 == 0 else b"\x03") + x

    def address(self):
        h160 = Hash.ripemd160(Hash.sha256(self.pubkey()))
        return Base58.encode_check(b"\x6f" + h160)  # testnet prefix

# =====================================================
# Network (Blockstream Testnet API)
# =====================================================

class TestnetAPI:
    BASE = "https://blockstream.info/testnet/api"

    @staticmethod
    def balance(address):
        r = requests.get(f"{TestnetAPI.BASE}/address/{address}")
        if r.status_code != 200:
            return 0
        data = r.json()
        return data["chain_stats"]["funded_txo_sum"] - data["chain_stats"]["spent_txo_sum"]

# =====================================================
# GUI App
# =====================================================

class ScannerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bitcoin Testnet Scanner")
        self.geometry("750x500")
        self.running = False

        ttk.Button(self, text="Start", command=self.start).pack(pady=5)
        ttk.Button(self, text="Stop", command=self.stop).pack()

        self.log = tk.Text(self, height=25)
        self.log.pack(fill="both")

        with open("burn-addresses-btc.txt") as f:
            self.burn = set(x.strip() for x in f if x.strip())

        with open("BTC-address-recieve.txt") as f:
            self.recv = [x.strip() for x in f if x.strip()]

    def log_msg(self, msg):
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)

    def start(self):
        self.running = True
        threading.Thread(target=self.loop, daemon=True).start()

    def stop(self):
        self.running = False

    def loop(self):
        while self.running:
            key = BitcoinKey()
            addr = key.address()
            self.log_msg(f"Checking {addr}")

            if addr in self.burn:
                bal = TestnetAPI.balance(addr)
                self.log_msg(f"🔥 MATCH FOUND — balance: {bal}")
                self.write_winner(key, addr)
                self.running = False
                return

            bal = TestnetAPI.balance(addr)
            if bal > 0:
                self.log_msg(f"💰 FUNDED ADDRESS FOUND: {bal} sats (testnet)")
                # Sending omitted intentionally – requires full TX builder

            time.sleep(0.5)

    def write_winner(self, key, addr):
        with open("winner.txt", "w") as f:
            f.write(f"Private key: {key.priv_hex()}\n")
            f.write(f"Address: {addr}\n")

# =====================================================
# Run
# =====================================================

if __name__ == "__main__":
    ScannerGUI().mainloop()
