import tkinter as tk
from tkinter import ttk
import secrets, hashlib, time, threading, requests, random, struct

from ecdsa import SigningKey, SECP256k1
from ecdsa.util import sigencode_der_canonize

# =====================================================
# Crypto helpers
# =====================================================

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

# =====================================================
# Base58 (testnet)
# =====================================================

class Base58:
    ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

    @staticmethod
    def encode(b):
        n = int.from_bytes(b, "big")
        s = ""
        while n:
            n, r = divmod(n, 58)
            s = Base58.ALPHABET[r] + s
        pad = len(b) - len(b.lstrip(b"\x00"))
        return Base58.ALPHABET[0] * pad + s

    @staticmethod
    def encode_check(payload):
        return Base58.encode(payload + Hash.dsha256(payload)[:4])

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
# Blockstream Testnet API
# =====================================================

class TestnetAPI:
    BASE = "https://blockstream.info/testnet/api"

    @staticmethod
    def balance(address):
        r = requests.get(f"{TestnetAPI.BASE}/address/{address}")
        if r.status_code != 200:
            return 0
        j = r.json()
        return j["chain_stats"]["funded_txo_sum"] - j["chain_stats"]["spent_txo_sum"]

# =====================================================
# GUI Application
# =====================================================

class ScannerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bitcoin Testnet Burn Matcher")
        self.geometry("780x520")

        self.running = False

        ttk.Button(self, text="Start", command=self.start).pack(pady=4)
        ttk.Button(self, text="Stop", command=self.stop).pack()

        self.log = tk.Text(self, height=28)
        self.log.pack(fill="both", padx=5, pady=5)

        with open("burn-addresses-btc.txt") as f:
            self.burn_addresses = set(x.strip() for x in f if x.strip())

        with open("BTC-address-recieve.txt") as f:
            self.recv_addresses = [x.strip() for x in f if x.strip()]

    def log_msg(self, msg):
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)

    def start(self):
        if not self.running:
            self.running = True
            threading.Thread(target=self.loop, daemon=True).start()

    def stop(self):
        self.running = False
        self.log_msg("Stopped.")

    def loop(self):
        while self.running:
            key = BitcoinKey()
            addr = key.address()

            self.log_msg(f"Generated: {addr}")
            self.log_msg(f"Hex Key:  {key.priv_hex()}")

            # 🔥 ONLY act if burn-address match
            if addr in self.burn_addresses:
                self.log_msg("🔥 BURN ADDRESS MATCH FOUND")
                bal = TestnetAPI.balance(addr)
                self.log_msg(f"Balance: {bal} sats")

                self.write_winner(key, addr, bal)

                if bal > 0:
                    recv = random.choice(self.recv_addresses)
                    self.log_msg(f"Would send {bal} sats → {recv}")
                    # Sending intentionally not automated beyond this point
                else:
                    self.log_msg("Balance is zero. Nothing to send.")

                self.running = False
                return

            time.sleep(0.3)

    def write_winner(self, key, addr, bal):
        with open("winner.txt", "w") as f:
            f.write(f"Private key (hex): {key.priv_hex()}\n")
            f.write(f"Address: {addr}\n")
            f.write(f"Balance (sats): {bal}\n")

# =====================================================
# Run
# =====================================================

if __name__ == "__main__":
    ScannerGUI().mainloop()
