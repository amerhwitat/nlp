# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 11:18:47 2026

@author: PC1
"""

import tkinter as tk
from tkinter import ttk, messagebox
import hashlib

# --- Base58Check decode ---
B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
def b58decode_check(s):
    num = 0
    for ch in s:
        if ch not in B58:
            raise ValueError("Invalid Base58 character")
        num = num * 58 + B58.index(ch)
    combined = num.to_bytes((num.bit_length() + 7) // 8, 'big') if num else b''
    # account for leading '1' (zero) chars
    nPad = len(s) - len(s.lstrip('1'))
    data = b'\x00' * nPad + combined
    if len(data) < 4:
        raise ValueError("Invalid Base58 data")
    payload, checksum = data[:-4], data[-4:]
    chk = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    if chk != checksum:
        raise ValueError("Invalid Base58 checksum")
    return payload  # includes version byte + payload

# --- Bech32 helpers ---
CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
GEN = [0x3b6a57b2,0x26508e6d,0x1ea119fa,0x3d4233dd,0x2a1462b3]

def bech32_polymod(values):
    chk = 1
    for v in values:
        top = chk >> 25
        chk = ((chk & 0x1ffffff) << 5) ^ v
        for i in range(5):
            if (top >> i) & 1:
                chk ^= GEN[i]
    return chk

def bech32_hrp_expand(hrp):
    return [ord(x) >> 5 for x in hrp] + [0] + [ord(x) & 31 for x in hrp]

def bech32_decode(addr):
    if (addr.lower() != addr) and (addr.upper() != addr):
        return (None, None)
    addr = addr.lower()
    if '1' not in addr:
        return (None, None)
    pos = addr.rfind('1')
    hrp = addr[:pos]
    data = [CHARSET.find(c) for c in addr[pos+1:]]
    if any(d == -1 for d in data):
        return (None, None)
    if bech32_polymod(bech32_hrp_expand(hrp) + data) != 1:
        return (None, None)
    return hrp, data[:-6]

def convertbits(data, frombits, tobits, pad=True):
    acc = 0; bits = 0; ret = []
    maxv = (1 << tobits) - 1
    for value in data:
        if value < 0 or (value >> frombits):
            return None
        acc = (acc << frombits) | value
        bits += frombits
        while bits >= tobits:
            bits -= tobits
            ret.append((acc >> bits) & maxv)
    if pad and bits:
        ret.append((acc << (tobits - bits)) & maxv)
    elif not pad and bits:
        return None
    return ret

# --- Builders ---
def scriptpubkey_from_bech32(addr):
    hrp, data = bech32_decode(addr)
    if hrp is None:
        raise ValueError("Invalid bech32")
    witver = data[0]
    prog = bytes(convertbits(data[1:], 5, 8, False))
    if witver != 0 or len(prog) != 20:
        raise ValueError("Only P2WPKH (witness v0, 20 bytes) supported for bech32 here")
    return b'\x00\x14' + prog

def scriptpubkey_from_base58(addr):
    payload = b58decode_check(addr)
    version = payload[0]
    h = payload[1:]
    if len(h) != 20:
        raise ValueError("Unexpected payload length")
    if version == 0x00:  # P2PKH
        return bytes.fromhex('76a914') + h + bytes.fromhex('88ac')
    elif version == 0x05:  # P2SH
        return bytes.fromhex('a914') + h + bytes.fromhex('87')
    else:
        raise ValueError(f"Unsupported Base58 version: {version:02x}")

def scriptpubkey_from_address(addr):
    a = addr.strip()
    if not a:
        raise ValueError("Empty address")
    if a.lower().startswith('bc1') or a.lower().startswith('tb1'):
        return scriptpubkey_from_bech32(a)
    else:
        return scriptpubkey_from_base58(a)

# --- GUI ---
class App:
    def __init__(self, root):
        self.root = root
        root.title("Address → scriptPubKey hex (Bech32 + Base58)")
        self.log = tk.Text(root, height=16, width=88)
        self.log.pack(padx=8, pady=6)
        self.progress = ttk.Progressbar(root, length=700, mode='determinate')
        self.progress.pack(padx=8, pady=6)
        ttk.Button(root, text="Run (read addr-segwit.txt)", command=self.run).pack(pady=6)

    def logmsg(self, s):
        self.log.insert('end', s + "\n"); self.log.see('end'); self.root.update()

    def run(self):
        try:
            with open('addr-segwit.txt','r') as f:
                addrs = [l.strip() for l in f if l.strip()]
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open addr-segwit.txt: {e}")
            return
        out = []
        total = len(addrs)
        self.progress['maximum'] = max(1, total)
        for i, a in enumerate(addrs, 1):
            try:
                spk = scriptpubkey_from_address(a)
                hexv = spk.hex()
                out.append(hexv)
                self.logmsg(f"[{i}/{total}] OK: {a} -> {hexv}")
            except Exception as e:
                out.append("")
                self.logmsg(f"[{i}/{total}] ERROR: {a} : {e}")
            self.progress['value'] = i
        try:
            with open('addr-ready.txt','w') as fo:
                for line in out:
                    fo.write(line + "\n")
            messagebox.showinfo("Done", "Wrote addr-ready.txt")
        except Exception as e:
            messagebox.showerror("Write error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()