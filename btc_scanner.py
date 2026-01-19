# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 18:03:22 2026

@author: PC1
"""

import os
import hashlib
import secrets
import time

from ecdsa import SigningKey, SECP256k1
from bech32 import bech32_encode, convertbits

# ==========================================
# Crypto utilities
# ==========================================

class Hash:
    @staticmethod
    def sha256(b):
        return hashlib.sha256(b).digest()

    @staticmethod
    def ripemd160(b):
        h = hashlib.new("ripemd160")
        h.update(b)
        return h.digest()

# ==========================================
# Bitcoin key & address
# ==========================================

class BitcoinKey:
    def __init__(self):
        self.private_key = secrets.token_bytes(32)
        self.signing_key = SigningKey.from_string(
            self.private_key, curve=SECP256k1
        )

    def private_hex(self):
        return self.private_key.hex()

    def public_key(self, compressed=True):
        p = self.signing_key.verifying_key.pubkey.point
        x = p.x().to_bytes(32, "big")
        y = p.y()
        if not compressed:
            return b"\x04" + x + y.to_bytes(32, "big")
        return (b"\x02" if y % 2 == 0 else b"\x03") + x

    def address_p2pkh(self):
        h160 = Hash.ripemd160(Hash.sha256(self.public_key()))
        return Base58.encode_check(b"\x00" + h160)

    def address_segwit(self):
        h160 = Hash.ripemd160(Hash.sha256(self.public_key()))
        return bech32_encode("bc", [0] + convertbits(h160, 8, 5))

# ==========================================
# Base58
# ==========================================

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

# ==========================================
# Address Scanner
# ==========================================

class AddressScanner:
    def __init__(self, burn_file):
        with open(burn_file, "r") as f:
            self.burn_addresses = set(
                line.strip() for line in f if line.strip()
            )

    def scan(self):
        while True:
            key = BitcoinKey()
            addr = key.address_p2pkh()

            if addr in self.burn_addresses:
                self.write_winner(key, addr)
                print("🔥 MATCH FOUND 🔥")
                break

            print("Tried:", addr)
            time.sleep(0.05)

    def write_winner(self, key, address):
        with open("winner.txt", "w") as f:
            f.write(f"Private key (hex): {key.private_hex()}\n")
            f.write(f"Address: {address}\n")

# ==========================================
# Run
# ==========================================

if __name__ == "__main__":
    scanner = AddressScanner("burn-addresses-btc.txt")
    scanner.scan()
