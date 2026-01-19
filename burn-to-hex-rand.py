# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 21:29:49 2026

@author: PC1
"""

import os, threading, time, requests, binascii
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox

# Default filenames
INFILE = "burn-addresses-btc.txt"
OUTFILE = "burn-hex.txt"
TIMEOUT = 15
PAUSE = 0.12  # gentle pacing

# Provider endpoints (mainnet)
PROVIDERS = {
    "Blockstream": {
        "base": "https://blockstream.info/api",
        "balance": lambda a: f"https://blockstream.info/api/address/{a}",
        "txs": lambda a: f"https://blockstream.info/api/address/{a}/txs",
        "tx": lambda txid: f"https://blockstream.info/api/tx/{txid}"
    },
    "Blockchair": {
        "base": "https://api.blockchair.com/bitcoin",
        "address": lambda a: f"https://api.blockchair.com/bitcoin/dashboards/address/{a}",
        "tx_raw": lambda txid: f"https://api.blockchair.com/bitcoin/raw/transaction/{txid}"
    },
    "BlockCypher": {
        "base": "https://api.blockcypher.com/v1/btc/main",
        "balance": lambda a: f"https://api.blockcypher.com/v1/btc/main/addrs/{a}/balance",
        "full": lambda a: f"https://api.blockcypher.com/v1/btc/main/addrs/{a}/full"
    },
    "SoChain": {
        "base": "https://sochain.com/api/v2",
        "balance": lambda a: f"https://sochain.com/api/v2/get_address_balance/BTC/{a}",
        "txs": lambda a: f"https://sochain.com/api/v2/get_tx_received/BTC/{a}"
    },
    "Blockchain.info": {
        "base": "https://blockchain.info",
        "rawaddr": lambda a: f"https://blockchain.info/rawaddr/{a}"
    }
}

# helpers
def safe_get(url):
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        return r
    except Exception as e:
        return None

def extract_pubkey_from_vin(vin):
    # vin may contain 'witness' (list) or 'scriptSig'/'scriptsig'
    try:
        if isinstance(vin, dict):
            # Blockstream style: vin['witness'] may be list of hex strings
            w = vin.get('witness') or vin.get('witnesses') or vin.get('witness_hex')
            if w:
                # if list, take last element
                if isinstance(w, list) and len(w):
                    last = w[-1]
                else:
                    last = w
                try:
                    b = bytes.fromhex(last)
                    if len(b) in (33,65):
                        return last
                except Exception:
                    pass
            ss = vin.get('scriptsig') or vin.get('scriptSig') or vin.get('script_sig')
            if ss:
                try:
                    data = bytes.fromhex(ss)
                    # search for 33/65 byte sequences
                    for i in range(len(data)-32):
                        cand = data[i:i+33]
                        if len(cand)==33 and cand[0] in (0x02,0x03,0x04):
                            return cand.hex()
                    for i in range(len(data)-64):
                        cand = data[i:i+65]
                        if len(cand)==65 and cand[0] in (0x02,0x03,0x04):
                            return cand.hex()
                except Exception:
                    pass
    except Exception:
        pass
    return None

# GUI
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Multi-Provider Burn Address Checker")
        self.geometry("820x520")
        self.create_widgets()

    def create_widgets(self):
        frm = ttk.Frame(self); frm.pack(fill='both', expand=True, padx=8, pady=8)
        ttk.Label(frm, text="Input file").grid(row=0,column=0,sticky='w')
        self.file_var = tk.StringVar(value=os.path.join(os.getcwd(), INFILE))
        ttk.Entry(frm, textvariable=self.file_var, width=64).grid(row=0,column=1,sticky='w')
        ttk.Button(frm, text="Browse", command=self.browse).grid(row=0,column=2,padx=6)

        # provider checkboxes
        ttk.Label(frm, text="Providers").grid(row=1,column=0,sticky='w', pady=(8,0))
        self.provider_vars = {}
        col = 1
        for name in PROVIDERS.keys():
            v = tk.BooleanVar(value=True)
            self.provider_vars[name] = v
            ttk.Checkbutton(frm, text=name, variable=v).grid(row=1, column=col, sticky='w', padx=4)
            col += 1

        ttk.Button(frm, text="Start", command=self.start).grid(row=2,column=0,pady=8)
        self.progress = ttk.Progressbar(frm, length=560, mode='determinate'); self.progress.grid(row=2,column=1,columnspan=2,sticky='w')

        ttk.Label(frm, text="Log").grid(row=3,column=0,sticky='w',pady=(10,0))
        self.log_area = scrolledtext.ScrolledText(frm, height=22, state='disabled'); self.log_area.grid(row=4,column=0,columnspan=3,sticky='nsew')

    def browse(self):
        f = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select input file")
        if f: self.file_var.set(f)

    def log(self, text):
        self.log_area.configure(state='normal')
        self.log_area.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {text}\n")
        self.log_area.see(tk.END)
        self.log_area.configure(state='disabled')

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
        self.log(f"Processing {total} addresses. Output: {out_path}")

        selected = [n for n,v in self.provider_vars.items() if v.get()]
        if not selected:
            messagebox.showerror("Error","Select at least one provider"); return

        with open(out_path,'w') as out_f:
            out_f.write("# address ; balance_sats ; pubkey_hex_or_NOT_FOUND ; provider_used ; status\n")
            for i, addr in enumerate(addrs, start=1):
                self.progress['value'] = i-1
                self.log(f"[{i}/{total}] {addr}")
                balance = None
                pubhex = None
                used_provider = None
                status = "OK"

                for prov in selected:
                    try:
                        if prov == "Blockstream":
                            r = safe_get(PROVIDERS[prov]['balance'](addr))
                            if r:
                                info = r.json()
                                cs = info.get('chain_stats',{})
                                balance = cs.get('funded_txo_sum',0) - cs.get('spent_txo_sum',0)
                                used_provider = prov
                                # try txs -> tx details
                                rtxs = safe_get(PROVIDERS[prov]['tx'](addr))  # not all endpoints same; fallback to /txs
                                txs = None
                                rtxs = safe_get(PROVIDERS[prov]['txs'](addr))
                                if rtxs:
                                    txs = rtxs.json()
                                if txs:
                                    for tx in txs:
                                        txid = tx.get('txid') or tx.get('id')
                                        if not txid: continue
                                        txfull_r = safe_get(PROVIDERS[prov]['tx'](txid))
                                        if not txfull_r: continue
                                        txfull = txfull_r.json()
                                        for vin in txfull.get('vin',[]):
                                            pk = extract_pubkey_from_vin(vin)
                                            if pk:
                                                pubhex = pk; break
                                        if pubhex: break
                        elif prov == "Blockchair":
                            r = safe_get(PROVIDERS[prov]['address'](addr))
                            if r:
                                j = r.json()
                                data = j.get('data',{}).get(addr,{})
                                balance = data.get('address',{}).get('balance', None)
                                used_provider = prov
                                # try raw tx via blockchair raw endpoint for txs if available
                                txs = data.get('transactions',[])
                                for txid in txs[:10]:
                                    rraw = safe_get(PROVIDERS[prov]['tx_raw'](txid))
                                    if rraw:
                                        raw = rraw.json().get('data',{}).get(txid,{}).get('raw_transaction')
                                        if raw:
                                            # naive parse: look for pubkey hex patterns
                                            if len(raw) > 100:
                                                # search for 66/130 hex chars sequences
                                                import re
                                                m = re.search(r'([0-9a-fA-F]{66}|[0-9a-fA-F]{130})', raw)
                                                if m:
                                                    pubhex = m.group(1); break
                                    if pubhex: break
                        elif prov == "BlockCypher":
                            r = safe_get(PROVIDERS[prov]['balance'](addr))
                            if r:
                                j = r.json()
                                balance = j.get('final_balance', None)
                                used_provider = prov
                                # try full txs
                                rfull = safe_get(PROVIDERS[prov]['full'](addr))
                                if rfull:
                                    jfull = rfull.json()
                                    txs = jfull.get('txs',[])
                                    for tx in txs:
                                        for vin in tx.get('inputs',[]):
                                            pk = extract_pubkey_from_vin(vin)
                                            if pk:
                                                pubhex = pk; break
                                        if pubhex: break
                        elif prov == "SoChain":
                            r = safe_get(PROVIDERS[prov]['balance'](addr))
                            if r:
                                j = r.json()
                                if j.get('status') == 'success':
                                    balance = int(float(j['data'].get('confirmed_balance',0)) * 1e8)
                                    used_provider = prov
                        elif prov == "Blockchain.info":
                            r = safe_get(PROVIDERS[prov]['rawaddr'](addr))
                            if r:
                                j = r.json()
                                balance = j.get('final_balance', None)
                                used_provider = prov
                                # try txs
                                for tx in j.get('txs',[]):
                                    for vin in tx.get('inputs',[]):
                                        pk = extract_pubkey_from_vin(vin)
                                        if pk:
                                            pubhex = pk; break
                                    if pubhex: break
                    except Exception as e:
                        self.log(f"Provider {prov} error for {addr}: {e}")
                    if pubhex and balance is not None:
                        break

                if balance is None:
                    status = "NO_BALANCE_INFO"
                    balance = 0
                if not pubhex:
                    pubhex = "NOT_FOUND"
                    status = status if status!="NO_BALANCE_INFO" else "NO_BALANCE_AND_PUBKEY"

                out_f.write(f"{addr} ; {balance} ; {pubhex} ; {used_provider or 'none'} ; {status}\n")
                self.log(f"[{i}/{total}] balance={balance} sats pubkey={'found' if pubhex!='NOT_FOUND' else 'NOT_FOUND'} via {used_provider or 'none'}")
                self.progress['value'] = i
                time.sleep(PAUSE)
        self.log("Done processing all addresses")
        messagebox.showinfo("Done", f"Wrote results to {out_path}")

if __name__ == "__main__":
    App().mainloop()