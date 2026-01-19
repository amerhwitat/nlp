# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 18:52:14 2026

@author: PC1
"""

import os
import re
import csv
import json
import hashlib
import requests
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from ecdsa import SigningKey, SECP256k1
from bech32 import bech32_encode, convertbits
import networkx as nx
import matplotlib.pyplot as plt

# -------------------------
# Utilities: crypto & base58
# -------------------------
BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

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
    pad = 0
    for c in s:
        if c == BASE58_ALPHABET[0]:
            pad += 1
        else:
            break
    return b"\x00" * pad + b

def wif_to_hex(wif: str):
    raw = base58_decode(wif)
    payload, checksum = raw[:-4], raw[-4:]
    if sha256(sha256(payload))[:4] != checksum:
        raise ValueError("Invalid WIF checksum")
    if payload[0] != 0x80:
        raise ValueError("Not a mainnet WIF")
    if len(payload) == 34 and payload[-1] == 0x01:
        return payload[1:-1].hex(), True
    elif len(payload) == 33:
        return payload[1:].hex(), False
    else:
        raise ValueError("Invalid WIF format")

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

def pubkey_to_p2pkh_address(pubkey_bytes: bytes) -> str:
    h160 = ripemd160(sha256(pubkey_bytes))
    versioned = b"\x00" + h160
    checksum = sha256(sha256(versioned))[:4]
    payload = versioned + checksum
    num = int.from_bytes(payload, "big")
    res = ""
    while num > 0:
        num, rem = divmod(num, 58)
        res = BASE58_ALPHABET[rem] + res
    pad = 0
    for byte in payload:
        if byte == 0:
            pad += 1
        else:
            break
    return BASE58_ALPHABET[0] * pad + res

def pubkey_to_p2wpkh_address(pubkey_bytes: bytes) -> str:
    h160 = ripemd160(sha256(pubkey_bytes))
    data = convertbits(h160, 8, 5)
    return bech32_encode("bc", [0] + data)

# -------------------------
# Blockchain fetch helpers
# -------------------------
BLOCKSTREAM_API = "https://blockstream.info/api"

def fetch_address_txs(address: str, limit: int = 50):
    """
    Fetch transactions for an address using Blockstream API.
    Returns list of tx JSON objects (as provided by the API).
    """
    try:
        url = f"{BLOCKSTREAM_API}/address/{address}/txs"
        resp = requests.get(url, timeout=12)
        resp.raise_for_status()
        return resp.json()[:limit]
    except Exception as e:
        raise RuntimeError(f"Failed to fetch txs for {address}: {e}")

# -------------------------
# Parsing: extract pubkeys from inputs
# -------------------------
def extract_pubkeys_from_input(input_obj):
    """
    Try to extract public keys from a tx input.
    Returns list of hex pubkeys found (compressed or uncompressed).
    Handles:
      - P2PKH scriptSig: <sig> <pubkey>
      - P2PK scriptSig: <sig> (pubkey may be in scriptPubKey instead)
      - P2WPKH / P2SH-P2WPKH: witness[1] often contains pubkey
    """
    found = []
    # scriptSig parsing (hex)
    script_sig = input_obj.get("scriptSig", {}).get("hex")
    if script_sig:
        # naive parse: look for 33 or 65 byte push data patterns
        # scriptSig is hex; scan for 66-char (33 bytes) or 130-char (65 bytes) sequences
        for m in re.finditer(r'([0-9a-fA-F]{66}|[0-9a-fA-F]{130})', script_sig):
            pk_hex = m.group(1)
            # basic validation: starts with 02/03/04
            if pk_hex.startswith(("02", "03", "04")):
                found.append(pk_hex)
    # witness parsing (list of hex strings)
    witness = input_obj.get("witness")
    if witness and isinstance(witness, list):
        # witness[1] is often pubkey for P2WPKH
        for item in witness:
            if isinstance(item, str) and len(item) in (66, 130) and item.startswith(("02", "03", "04")):
                found.append(item)
    # scriptSig asm may also contain pubkey
    asm = input_obj.get("scriptSig", {}).get("asm", "")
    for m in re.finditer(r'(02|03|04)[0-9a-fA-F]{64,128}', asm):
        found.append(m.group(0))
    # dedupe
    return list(dict.fromkeys(found))

def extract_pubkeys_from_tx(tx):
    """
    For a transaction JSON (Blockstream format), extract pubkeys from inputs.
    Returns dict: {txid: [pubkey_hex, ...]}
    """
    txid = tx.get("txid")
    res = []
    for vin in tx.get("vin", []):
        res.extend(extract_pubkeys_from_input(vin))
    return txid, list(dict.fromkeys(res))

# -------------------------
# File & relation processing
# -------------------------
def load_addresses_from_file(path="burn-addresses-btc.txt"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found in current folder.")
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    # basic validation: keep lines that look like addresses
    addrs = [l for l in lines if re.match(r'^[13mn][A-HJ-NP-Za-km-z1-9]{25,39}$', l) or l.startswith("bc1")]
    return addrs

def analyze_addresses(addresses, tx_limit=25):
    """
    For each address, fetch txs and extract revealed pubkeys.
    Returns:
      - addr_to_txs: {address: [txid,...]}
      - tx_to_pubkeys: {txid: [pubkey_hex,...]}
      - pubkey_to_addresses: {pubkey_hex: [address,...]} (addresses where pubkey was revealed)
    """
    addr_to_txs = {}
    tx_to_pubkeys = {}
    pubkey_to_addresses = {}
    for addr in addresses:
        try:
            txs = fetch_address_txs(addr, limit=tx_limit)
        except Exception as e:
            txs = []
        addr_to_txs[addr] = [t.get("txid") for t in txs]
        for tx in txs:
            txid, pks = extract_pubkeys_from_tx(tx)
            if pks:
                tx_to_pubkeys.setdefault(txid, []).extend(pks)
                for pk in pks:
                    pubkey_to_addresses.setdefault(pk, []).append(addr)
    # dedupe lists
    for k in tx_to_pubkeys:
        tx_to_pubkeys[k] = list(dict.fromkeys(tx_to_pubkeys[k]))
    for k in pubkey_to_addresses:
        pubkey_to_addresses[k] = list(dict.fromkeys(pubkey_to_addresses[k]))
    return addr_to_txs, tx_to_pubkeys, pubkey_to_addresses

# -------------------------
# GUI Application
# -------------------------
class RelationInspector(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Address-Pubkey Relation Inspector (read-only)")
        self.geometry("1000x700")
        self.create_widgets()
        self.addr_list = []
        self.addr_to_txs = {}
        self.tx_to_pubkeys = {}
        self.pubkey_to_addresses = {}
        self.local_privkeys = {}  # hex -> (wif, compressed, derived_pub_hex)

    def create_widgets(self):
        frm = ttk.Frame(self, padding=8)
        frm.pack(fill=tk.BOTH, expand=True)

        # Top controls
        ttk.Button(frm, text="Load burn-addresses-btc.txt", command=self.load_burn_file).grid(column=0, row=0, sticky=tk.W)
        ttk.Button(frm, text="Analyze loaded addresses", command=self.run_analysis).grid(column=1, row=0, sticky=tk.W)
        ttk.Button(frm, text="Load local private keys (optional)", command=self.load_local_privkeys).grid(column=2, row=0, sticky=tk.W)
        ttk.Button(frm, text="Export CSV summary", command=self.export_csv).grid(column=3, row=0, sticky=tk.W)
        ttk.Button(frm, text="Visualize graph", command=self.visualize_graph).grid(column=4, row=0, sticky=tk.W)

        # Results panes
        ttk.Label(frm, text="Loaded addresses:").grid(column=0, row=1, sticky=tk.W, pady=(8,0))
        self.addr_box = scrolledtext.ScrolledText(frm, width=60, height=8)
        self.addr_box.grid(column=0, row=2, columnspan=3, sticky=tk.W)

        ttk.Label(frm, text="Analysis log:").grid(column=0, row=3, sticky=tk.W, pady=(8,0))
        self.log_box = scrolledtext.ScrolledText(frm, width=120, height=18)
        self.log_box.grid(column=0, row=4, columnspan=5, pady=(0,8))

    def log(self, *parts):
        self.log_box.insert(tk.END, " ".join(str(p) for p in parts) + "\n")
        self.log_box.see(tk.END)

    def load_burn_file(self):
        try:
            addrs = load_addresses_from_file()
            self.addr_list = addrs
            self.addr_box.delete("1.0", tk.END)
            for a in addrs:
                self.addr_box.insert(tk.END, a + "\n")
            self.log(f"Loaded {len(addrs)} addresses from burn-addresses-btc.txt")
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def run_analysis(self):
        if not self.addr_list:
            messagebox.showinfo("No addresses", "Load burn-addresses-btc.txt first.")
            return
        self.log_box.delete("1.0", tk.END)
        self.log("Starting analysis (this may take a while)...")
        try:
            addr_to_txs, tx_to_pubkeys, pubkey_to_addresses = analyze_addresses(self.addr_list, tx_limit=50)
            self.addr_to_txs = addr_to_txs
            self.tx_to_pubkeys = tx_to_pubkeys
            self.pubkey_to_addresses = pubkey_to_addresses
            # Summarize
            total_revealed_pubkeys = len(pubkey_to_addresses)
            self.log(f"Analysis complete. Addresses: {len(self.addr_list)}; revealed pubkeys: {total_revealed_pubkeys}")
            for pk, addrs in pubkey_to_addresses.items():
                self.log(f"Pubkey {pk} appears in addresses: {', '.join(addrs)}")
            # Show txs with pubkeys
            for txid, pks in tx_to_pubkeys.items():
                self.log(f"TX {txid} reveals pubkeys: {', '.join(pks)}")
            # If local privkeys loaded, compare
            if self.local_privkeys:
                self.log("Comparing revealed pubkeys to locally loaded private keys...")
                for priv_hex, info in self.local_privkeys.items():
                    derived = info.get("derived_pub")
                    if not derived:
                        continue
                    if derived in pubkey_to_addresses:
                        self.log(f"Local private key {priv_hex} DERIVED pubkey {derived} matches revealed pubkey in addresses: {pubkey_to_addresses[derived]}")
                    else:
                        self.log(f"Local private key {priv_hex} derived pubkey not found among revealed pubkeys.")
            self.log("Done.")
        except Exception as e:
            messagebox.showerror("Analysis error", str(e))

    def load_local_privkeys(self):
        """
        Load a local file of private keys (one per line). Accepts hex (64 chars) or WIF.
        This file is read locally and not transmitted anywhere.
        """
        path = filedialog.askopenfilename(title="Select private keys file (local only)", filetypes=[("Text files","*.txt"),("All files","*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            self.local_privkeys = {}
            for line in lines:
                try:
                    if re.fullmatch(r'[0-9a-fA-F]{64}', line):
                        hex_priv = line
                        # assume compressed by default
                        pub = privkey_to_pubkey(hex_priv, compressed=True).hex()
                        self.local_privkeys[hex_priv] = {"wif": None, "compressed": True, "derived_pub": pub}
                    else:
                        # try WIF
                        try:
                            hex_priv, compressed = wif_to_hex(line)
                            pub = privkey_to_pubkey(hex_priv, compressed=compressed).hex()
                            self.local_privkeys[hex_priv] = {"wif": line, "compressed": compressed, "derived_pub": pub}
                        except Exception:
                            # skip invalid lines
                            continue
                except Exception:
                    continue
            self.log(f"Loaded {len(self.local_privkeys)} local private keys from {os.path.basename(path)} (kept local only).")
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def export_csv(self):
        if not (self.addr_to_txs or self.tx_to_pubkeys or self.pubkey_to_addresses):
            messagebox.showinfo("Nothing to export", "Run analysis first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv")])
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["address","txid_list","revealed_pubkeys"])
                for addr in self.addr_list:
                    txs = self.addr_to_txs.get(addr, [])
                    # collect pubkeys for this address
                    pks = []
                    for txid in txs:
                        pks.extend(self.tx_to_pubkeys.get(txid, []))
                    writer.writerow([addr, ";".join(txs), ";".join(list(dict.fromkeys(pks)))])
            self.log(f"Exported CSV to {path}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))

    def visualize_graph(self):
        """
        Build a bipartite graph: addresses <-> pubkeys. Draw with networkx.
        """
        if not self.pubkey_to_addresses:
            messagebox.showinfo("No data", "Run analysis first and ensure some pubkeys were revealed.")
            return
        G = nx.Graph()
        # add address nodes
        for addr in self.addr_list:
            G.add_node(addr, bipartite=0)
        # add pubkey nodes and edges
        for pk, addrs in self.pubkey_to_addresses.items():
            G.add_node(pk, bipartite=1)
            for a in addrs:
                G.add_edge(a, pk)
        # layout
        pos = nx.spring_layout(G, k=0.5, iterations=100)
        plt.figure(figsize=(12, 8))
        # color nodes by type
        addr_nodes = [n for n in G.nodes() if n in self.addr_list]
        pk_nodes = [n for n in G.nodes() if n not in self.addr_list]
        nx.draw_networkx_nodes(G, pos, nodelist=addr_nodes, node_color="orange", node_size=300, label="addresses")
        nx.draw_networkx_nodes(G, pos, nodelist=pk_nodes, node_color="lightblue", node_size=200, label="pubkeys")
        nx.draw_networkx_edges(G, pos, alpha=0.6)
        # labels: shorten pubkey labels for readability
        labels = {n: (n if n in self.addr_list else n[:12]+"...") for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        plt.title("Address ↔ Revealed Pubkey Graph")
        plt.axis("off")
        plt.legend(scatterpoints=1)
        plt.show()

if __name__ == "__main__":
    app = RelationInspector()
    app.mainloop()