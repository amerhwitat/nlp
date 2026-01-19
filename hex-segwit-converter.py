# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 10:35:37 2026

@author: PC1
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import hashlib, binascii, struct

# --- Base58Check decode (no external libs) ---
B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
def b58decode_check(s):
    num = 0
    for ch in s:
        num = num * 58 + B58.index(ch)
    combined = num.to_bytes((num.bit_length()+7)//8, 'big')
    # leading zeros
    nPad = len(s) - len(s.lstrip('1'))
    data = b'\x00'*nPad + combined
    if len(data) < 4:
        raise ValueError("Invalid base58")
    payload, checksum = data[:-4], data[-4:]
    if hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4] != checksum:
        raise ValueError("Invalid checksum")
    return payload

# --- Bech32 decode (minimal) ---
CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
GEN = [0x3b6a57b2,0x26508e6d,0x1ea119fa,0x3d4233dd,0x2a1462b3]
def bech32_polymod(values):
    chk=1
    for v in values:
        b=chk>>25
        chk=((chk&0x1ffffff)<<5)^v
        for i in range(5):
            if (b>>i)&1:
                chk ^= GEN[i]
    return chk
def bech32_hrp_expand(hrp):
    return [ord(x)>>5 for x in hrp] + [0] + [ord(x)&31 for x in hrp]
def bech32_decode(bech):
    bech = bech.lower()
    if ('1' not in bech) or (bech.rfind('1')==0):
        return (None,None)
    hrp = bech[:bech.rfind('1')]
    data = [CHARSET.find(c) for c in bech[bech.rfind('1')+1:]]
    if -1 in data:
        return (None,None)
    if bech32_polymod(bech32_hrp_expand(hrp)+data) != 1:
        return (None,None)
    return hrp, data[:-6]
def convertbits(data, frombits, tobits, pad=True):
    acc=0; bits=0; ret=[]
    maxv=(1<<tobits)-1
    for value in data:
        acc = (acc<<frombits) | value
        bits += frombits
        while bits>=tobits:
            bits -= tobits
            ret.append((acc>>bits)&maxv)
    if pad:
        if bits:
            ret.append((acc<<(tobits-bits))&maxv)
    elif bits>=frombits or ((acc<<(tobits-bits))&maxv):
        return None
    return ret

# --- scriptPubKey builders ---
def scriptpubkey_from_address(addr):
    addr = addr.strip()
    if addr.lower().startswith('bc1'):
        hrp, data = bech32_decode(addr)
        if hrp is None:
            raise ValueError("Invalid bech32")
        witver = data[0]
        prog = bytes(convertbits(data[1:],5,8,False))
        if witver == 0:
            if len(prog)==20:
                # P2WPKH
                return b'\x00\x14' + prog
            elif len(prog)==32:
                # P2WSH
                return b'\x00\x20' + prog
        # other witness versions
        return bytes([0x50 + witver]) + bytes([len(prog)]) + prog
    else:
        payload = b58decode_check(addr)
        version = payload[0]
        h = payload[1:]
        if version == 0x00:  # P2PKH
            return bytes.fromhex('76a914') + h + bytes.fromhex('88ac')
        elif version == 0x05:  # P2SH
            return bytes.fromhex('a914') + h + bytes.fromhex('87')
        else:
            raise ValueError("Unknown base58 version: %02x" % version)

# --- GUI ---
class App:
    def __init__(self, root):
        self.root = root
        root.title("Address → scriptPubKey hex")
        self.log = tk.Text(root, height=15, width=80)
        self.log.pack(padx=8, pady=6)
        self.progress = ttk.Progressbar(root, length=600, mode='determinate')
        self.progress.pack(padx=8, pady=6)
        frm = tk.Frame(root)
        frm.pack(pady=6)
        tk.Button(frm, text="Run", command=self.run).pack(side='left', padx=6)
        tk.Button(frm, text="Choose input file", command=self.choose).pack(side='left', padx=6)
        self.infile = "addr-hex.txt"

    def logmsg(self, s):
        self.log.insert('end', s+"\n"); self.log.see('end'); self.root.update()

    def choose(self):
        f = filedialog.askopenfilename(initialdir='.', title='Select addr-hex.txt')
        if f: self.infile = f; self.logmsg("Selected: "+f)

    def run(self):
        try:
            with open(self.infile,'r') as f:
                addrs = [l.strip() for l in f if l.strip()]
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open {self.infile}: {e}"); return
        out = []
        total = len(addrs)
        self.progress['maximum'] = total
        for i, a in enumerate(addrs,1):
            try:
                spk = scriptpubkey_from_address(a)
                hexv = spk.hex()
                out.append(hexv)
                self.logmsg(f"[{i}/{total}] OK: {a} → {hexv}")
            except Exception as e:
                out.append("") 
                self.logmsg(f"[{i}/{total}] ERROR: {a} : {e}")
            self.progress['value'] = i
        with open('addr-ready.txt','w') as fo:
            for line in out:
                fo.write(line+"\n")
        messagebox.showinfo("Done", "Wrote addr-ready.txt")

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()