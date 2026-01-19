import os
import requests
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from bs4 import BeautifulSoup
from bitcoinlib.encoding import addr_bech32_to_pubkeyhash

BASE_ADDR_URL = "https://blockchair.com/bitcoin/address/{}"
BASE_TX_URL = "https://blockchair.com/bitcoin/transaction/{}"

HEADERS = {"User-Agent": "Mozilla/5.0"}

class SegWitScannerGUI:
    def __init__(self, root):
        root.title("SegWit Balance & Public Key Scanner (Blockchain)")
        root.geometry("1000x650")

        self.input_path = tk.StringVar()

        ttk.Label(root, text="addr-segwit.txt (bc1q...)").pack(pady=5)
        ttk.Entry(root, textvariable=self.input_path, width=110).pack()
        ttk.Button(root, text="Browse", command=self.browse).pack(pady=5)
        ttk.Button(root, text="Scan Blockchain", command=self.run).pack(pady=10)

        self.log = tk.Text(root, height=30)
        self.log.pack(fill="both", expand=True, padx=10, pady=10)

    def browse(self):
        p = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if p:
            self.input_path.set(p)

    def get_balance_and_txs(self, address):
        r = requests.get(BASE_ADDR_URL.format(address), headers=HEADERS, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")

        balance = 0
        txids = []

        for span in soup.find_all("span"):
            if span.get("data-balance"):
                balance = int(span["data-balance"])

        for a in soup.find_all("a", href=True):
            if "/transaction/" in a["href"]:
                txids.append(a["href"].split("/")[-1])

        return balance, list(set(txids))

    def extract_pubkey_from_tx(self, txid):
        r = requests.get(BASE_TX_URL.format(txid), headers=HEADERS, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")

        # Witness data is displayed as hex blobs
        for code in soup.find_all("code"):
            text = code.text.strip().lower()
            if len(text) == 66 and text.startswith(("02", "03")):
                return text

        return None

    def run(self):
        infile = self.input_path.get()
        if not os.path.exists(infile):
            messagebox.showerror("Error", "Input file not found")
            return

        outfile = os.path.join(os.path.dirname(infile), "hex-segwit.txt")

        with open(infile, "r", encoding="utf-8", errors="ignore") as f:
            addresses = [l.strip() for l in f if l.strip()]

        found = 0
        with open(outfile, "w") as out:
            for addr in addresses:
                try:
                    balance, txids = self.get_balance_and_txs(addr)
                    pubkey = None

                    for txid in txids:
                        pubkey = self.extract_pubkey_from_tx(txid)
                        if pubkey:
                            break

                    if not pubkey:
                        pubkey = "NOT_AVAILABLE"

                    line = f"{addr} | {balance} sats | {pubkey}"
                    out.write(line + "\n")
                    self.log.insert("end", line + "\n")

                    if pubkey != "NOT_AVAILABLE":
                        found += 1

                except Exception as e:
                    self.log.insert("end", f"FAILED {addr}: {e}\n")

        self.log.insert(
            "end",
            f"\nDone.\nPublic keys found: {found}\nSaved to hex-segwit.txt\n"
        )

if __name__ == "__main__":
    root = tk.Tk()
    SegWitScannerGUI(root)
    root.mainloop()
