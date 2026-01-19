# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 17:39:44 2026

@author: PC1
"""

import tkinter as tk
from tkinter import ttk, messagebox
import hashlib, hmac, struct, os, base64
import ecdsa
from ecdsa import SigningKey, SECP256k1
from ecdsa.util import sigencode_der_canonize
from mnemonic import Mnemonic
from bech32 import bech32_encode, convertbits
from coincurve import PrivateKey

# ======================================================
# Utilities
# ======================================================

class Hash:
    @staticmethod
    def sha256(b): return hashlib.sha256(b).digest()

    @staticmethod
    def dsha256(b): return Hash.sha256(Hash.sha256(b))

    @staticmethod
    def ripemd160(b):
        h = hashlib.new("ripemd160")
        h.update(b)
        return h.digest()

# ======================================================
# Base58 / WIF
# ======================================================

class Base58:
    ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

    @staticmethod
    def encode(b):
        n = int.from_bytes(b, "big")
        s = ""
        while n > 0:
            n, r = divmod(n, 58)
            s = Base58.ALPHABET[r] + s
        pad = len(b) - len(b.lstrip(b"\x00"))
        return Base58.ALPHABET[0] * pad + s

    @staticmethod
    def check_encode(payload):
        checksum = Hash.dsha256(payload)[:4]
        return Base58.encode(payload + checksum)

# ======================================================
# Keys & Addresses
# ======================================================

class BitcoinKey:
    def __init__(self, priv_hex):
        self.priv = bytes.fromhex(priv_hex)
        self.sk = SigningKey.from_string(self.priv, curve=SECP256k1)
        self.vk = self.sk.verifying_key

    def pubkey(self, compressed=True):
        p = self.vk.pubkey.point
        x = p.x().to_bytes(32, "big")
        y = p.y()
        if not compressed:
            return b"\x04" + x + y.to_bytes(32, "big")
        return (b"\x02" if y % 2 == 0 else b"\x03") + x

    def address_p2pkh(self):
        h160 = Hash.ripemd160(Hash.sha256(self.pubkey()))
        return Base58.check_encode(b"\x00" + h160)

    def address_segwit(self):
        h160 = Hash.ripemd160(Hash.sha256(self.pubkey()))
        return bech32_encode("bc", [0] + convertbits(h160, 8, 5))

    def address_taproot(self):
        pk = PrivateKey(self.priv)
        xonly = pk.public_key.format(compressed=False)[1:33]
        return bech32_encode("bc", [1] + convertbits(xonly, 8, 5))

# ======================================================
# BIP39 / BIP32
# ======================================================

class HDWallet:
    def __init__(self):
        self.mnemo = Mnemonic("english")

    def generate(self):
        words = self.mnemo.generate(128)
        seed = self.mnemo.to_seed(words)
        return words, seed

# ======================================================
# Signing
# ======================================================

class Signer:
    @staticmethod
    def sign_ecdsa(privkey, msg):
        z = Hash.sha256(msg)
        sk = SigningKey.from_string(privkey, curve=SECP256k1)
        return sk.sign_digest(z, sigencode=sigencode_der_canonize)

    @staticmethod
    def sign_schnorr(privkey, msg32):
        return PrivateKey(privkey).sign_schnorr(msg32)

# ======================================================
# GUI
# ======================================================

class BitcoinGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bitcoin Dev Toolkit")
        self.geometry("820x600")
        self.resizable(False, False)

        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True)

        self._key_tab()
        self._wallet_tab()
        self._sign_tab()

    # --------------------------------------------------

    def _key_tab(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="Keys & Addresses")

        ttk.Label(tab, text="Private Key (hex):").pack(anchor="w")
        self.priv_entry = ttk.Entry(tab, width=80)
        self.priv_entry.pack()

        ttk.Button(tab, text="Generate Addresses", command=self.gen_keys).pack(pady=5)

        self.out = tk.Text(tab, height=15)
        self.out.pack(fill="x")

    def gen_keys(self):
        try:
            key = BitcoinKey(self.priv_entry.get())
            self.out.delete("1.0", tk.END)
            self.out.insert(tk.END, f"P2PKH: {key.address_p2pkh()}\n")
            self.out.insert(tk.END, f"SegWit: {key.address_segwit()}\n")
            self.out.insert(tk.END, f"Taproot: {key.address_taproot()}\n")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # --------------------------------------------------

    def _wallet_tab(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="HD Wallet")

        ttk.Button(tab, text="Generate BIP39 Wallet", command=self.gen_wallet).pack(pady=10)
        self.wallet_out = tk.Text(tab, height=20)
        self.wallet_out.pack(fill="x")

    def gen_wallet(self):
        hd = HDWallet()
        words, seed = hd.generate()
        self.wallet_out.delete("1.0", tk.END)
        self.wallet_out.insert(tk.END, "Mnemonic:\n" + words + "\n\n")
        self.wallet_out.insert(tk.END, f"Seed:\n{seed.hex()}")

    # --------------------------------------------------

    def _sign_tab(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="Signing")

        ttk.Label(tab, text="Private Key (hex):").pack(anchor="w")
        self.sign_priv = ttk.Entry(tab, width=80)
        self.sign_priv.pack()

        ttk.Label(tab, text="Message:").pack(anchor="w")
        self.sign_msg = ttk.Entry(tab, width=80)
        self.sign_msg.pack()

        ttk.Button(tab, text="ECDSA Sign", command=self.do_sign).pack(pady=5)

        self.sign_out = tk.Text(tab, height=10)
        self.sign_out.pack(fill="x")

    def do_sign(self):
        sig = Signer.sign_ecdsa(
            bytes.fromhex(self.sign_priv.get()),
            self.sign_msg.get().encode()
        )
        self.sign_out.delete("1.0", tk.END)
        self.sign_out.insert(tk.END, sig.hex())

# ======================================================
# Run
# ======================================================

if __name__ == "__main__":
    BitcoinGUI().mainloop()
