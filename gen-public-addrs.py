import threading
import os
import hashlib
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from ecdsa import SigningKey, SECP256k1

DEFAULT_INPUT = "private_key.txt"
OUTPUT_FILENAME = "public_keys.txt"

def log(gui, text):
    gui.log_area.configure(state='normal')
    gui.log_area.insert(tk.END, text + "\n")
    gui.log_area.see(tk.END)
    gui.log_area.configure(state='disabled')

def sha256(b): return hashlib.sha256(b).digest()
def ripemd160(b): return hashlib.new('ripemd160', b).digest()

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Line-by-Line Bitcoin Public Key Hex Generator")
        self.geometry("780x520")
        self.create_widgets()

    def create_widgets(self):
        frm = ttk.Frame(self)
        frm.pack(fill='both', expand=True, padx=10, pady=10)

        ttk.Label(frm, text="Private key file").grid(row=0, column=0, sticky='w')
        self.file_var = tk.StringVar(value=os.path.join(os.getcwd(), DEFAULT_INPUT))
        ttk.Entry(frm, textvariable=self.file_var, width=68).grid(row=0, column=1, sticky='w')
        ttk.Button(frm, text="Browse", command=self.browse).grid(row=0, column=2, padx=6)

        ttk.Button(frm, text="Generate (line-by-line)", command=self.start_generate).grid(row=1, column=0, pady=8)
        self.progress = ttk.Progressbar(frm, length=520, mode='determinate')
        self.progress.grid(row=1, column=1, columnspan=2, sticky='w')

        ttk.Label(frm, text="Last processed uncompressed public key hex").grid(row=2, column=0, sticky='w', pady=(10,0))
        self.uncomp = tk.Text(frm, height=2, width=95)
        self.uncomp.grid(row=3, column=0, columnspan=3)

        ttk.Label(frm, text="Last processed compressed public key hex").grid(row=4, column=0, sticky='w', pady=(10,0))
        self.comp = tk.Text(frm, height=2, width=95)
        self.comp.grid(row=5, column=0, columnspan=3)

        ttk.Label(frm, text="Last processed pubkey hash (RIPEMD160(SHA256)) hex").grid(row=6, column=0, sticky='w', pady=(10,0))
        self.pkhash = tk.Text(frm, height=1, width=95)
        self.pkhash.grid(row=7, column=0, columnspan=3)

        ttk.Label(frm, text="Log").grid(row=8, column=0, sticky='w', pady=(10,0))
        self.log_area = scrolledtext.ScrolledText(frm, height=10, state='disabled')
        self.log_area.grid(row=9, column=0, columnspan=3, sticky='nsew')

    def browse(self):
        f = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select private key file")
        if f:
            self.file_var.set(f)

    def start_generate(self):
        t = threading.Thread(target=self.generate_line_by_line, daemon=True)
        t.start()

    def generate_line_by_line(self):
        # Reset UI
        self.progress['value'] = 0
        self.uncomp.delete('1.0', tk.END)
        self.comp.delete('1.0', tk.END)
        self.pkhash.delete('1.0', tk.END)
        path = self.file_var.get()
        if not os.path.isfile(path):
            messagebox.showerror("Error", f"File not found: {path}")
            log(self, f"Error: input file not found: {path}")
            return

        # First pass: count non-empty, non-comment lines to set progress maximum
        try:
            with open(path, 'r') as fh:
                raw_lines = fh.readlines()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read private key file: {e}")
            log(self, f"Error reading file: {e}")
            return

        candidate_lines = [ln.rstrip("\n") for ln in raw_lines]
        keys_to_process = [ln for ln in candidate_lines if ln and not ln.strip().startswith('#')]
        total = len(keys_to_process)
        if total == 0:
            messagebox.showinfo("Info", "No private keys found in the file.")
            log(self, "No private keys to process.")
            return

        self.progress['maximum'] = total
        out_path = os.path.join(os.path.dirname(path), OUTPUT_FILENAME)
        log(self, f"Processing {total} private key(s). Writing line-by-line to {out_path}")

        # Open output file once and write header (overwrite existing file)
        try:
            out_f = open(out_path, 'w', buffering=1)  # line buffered
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open output file for writing: {e}")
            log(self, f"Error opening output file: {e}")
            return

        out_f.write("# Format: private_hex ; uncompressed_pub_hex ; compressed_pub_hex ; pubkey_hash_hex ; status\n")

        processed = 0
        # Second pass: iterate original lines and process those that qualify
        for idx, raw in enumerate(candidate_lines, start=1):
            line = raw.strip()
            if not line or line.startswith('#'):
                continue

            processed += 1
            self.progress['value'] = processed - 1
            display_line = line[:40] + ("..." if len(line) > 40 else "")
            log(self, f"[{processed}/{total}] Processing: {display_line}")

            priv_hex = ""
            pub_hex_uncompressed = ""
            pub_hex_compressed = ""
            h_hex = ""
            status = "OK"

            try:
                priv_hex = line.split()[0]
                priv_bytes = bytes.fromhex(priv_hex)
                if len(priv_bytes) != 32:
                    raise ValueError("private key must be 32 bytes (64 hex chars)")
            except Exception as e:
                status = f"ERROR: invalid private key - {e}"
                log(self, f"[{processed}/{total}] {status}")
                out_f.write(f"{line} ; {pub_hex_uncompressed} ; {pub_hex_compressed} ; {h_hex} ; {status}\n")
                self.progress['value'] = processed
                continue

            try:
                sk = SigningKey.from_string(priv_bytes, curve=SECP256k1)
                vk = sk.verifying_key
                raw_pub = vk.to_string()  # 64 bytes X||Y

                # Uncompressed
                pub_uncompressed = b'\x04' + raw_pub
                pub_hex_uncompressed = pub_uncompressed.hex()

                # Compressed
                px = raw_pub[:32]
                py = raw_pub[32:]
                parity = int.from_bytes(py, 'big') & 1
                prefix = b'\x03' if parity else b'\x02'
                pub_compressed = prefix + px
                pub_hex_compressed = pub_compressed.hex()

                # pubkey hash
                h = ripemd160(sha256(pub_compressed))
                h_hex = h.hex()

                # Update UI with last processed
                self.uncomp.delete('1.0', tk.END); self.uncomp.insert(tk.END, pub_hex_uncompressed)
                self.comp.delete('1.0', tk.END); self.comp.insert(tk.END, pub_hex_compressed)
                self.pkhash.delete('1.0', tk.END); self.pkhash.insert(tk.END, h_hex)

                log(self, f"[{processed}/{total}] OK - compressed: {pub_hex_compressed[:40]}...")
            except Exception as e:
                status = f"ERROR: derive failed - {e}"
                log(self, f"[{processed}/{total}] {status}")

            # Write the result line-by-line immediately
            try:
                out_f.write(f"{priv_hex} ; {pub_hex_uncompressed} ; {pub_hex_compressed} ; {h_hex} ; {status}\n")
            except Exception as e:
                log(self, f"Failed to write line {processed} to output: {e}")

            self.progress['value'] = processed

        out_f.close()
        log(self, f"Finished processing. Wrote {processed} entries to {out_path}")
        messagebox.showinfo("Done", f"Processed {processed} keys. Output: {out_path}")

if __name__ == "__main__":
    App().mainloop()