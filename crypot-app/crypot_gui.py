"""Unified Tkinter GUI for the cooperative Crypto Research App.

The GUI exposes the local hashing lab, public-address analysis, cooperative
P2P messaging, local client/server jobs, and the RNN/LLM research facade.
No private keys are requested, generated, stored, or transmitted.
"""
from __future__ import annotations

import asyncio
import json
import queue
import threading
import tkinter as tk
from datetime import datetime, timezone
from tkinter import filedialog, messagebox, ttk

from client_server import ResearchHandler, submit
from http.server import ThreadingHTTPServer
from gui_support import (
    build_analysis_report,
    build_digest_report,
    build_peer_message,
    format_json,
    validate_public_address,
)
from p2p_transport import P2PPeer


class CryptoResearchGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Chimera Crypto Research App — Unified GUI")
        self.geometry("1180x820")
        self.minsize(980, 680)
        self.events: queue.Queue[str] = queue.Queue()
        self.p2p_thread: threading.Thread | None = None
        self.p2p_loop: asyncio.AbstractEventLoop | None = None
        self.p2p_peer: P2PPeer | None = None
        self.p2p_server = None
        self.http_server: ThreadingHTTPServer | None = None
        self.http_thread: threading.Thread | None = None
        self._build_style()
        self._build_ui()
        self.after(100, self._drain_events)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.log("GUI initialized; all local research modules are ready.")

    def _build_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("TkDefaultFont", 18, "bold"))
        style.configure("Status.TLabel", font=("TkDefaultFont", 10, "bold"))

    def _build_ui(self) -> None:
        header = ttk.Frame(self, padding=12)
        header.pack(fill="x")
        ttk.Label(header, text="Chimera Crypto Research App", style="Title.TLabel").pack(side="left")
        self.status_var = tk.StringVar(value="READY")
        ttk.Label(header, textvariable=self.status_var, style="Status.TLabel").pack(side="right")

        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self._build_dashboard()
        self._build_crypto_tab()
        self._build_address_tab()
        self._build_p2p_tab()
        self._build_server_tab()
        self._build_rnn_tab()
        self._build_logs_tab()

    def _text_area(self, parent, height=10):
        frame = ttk.Frame(parent)
        text = tk.Text(frame, height=height, wrap="word", undo=True)
        scroll = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        return frame, text

    def _build_dashboard(self) -> None:
        tab = ttk.Frame(self.tabs, padding=14)
        self.tabs.add(tab, text="Dashboard")
        ttk.Label(tab, text="Integrated modules", style="Title.TLabel").pack(anchor="w")
        modules = [
            ("Cryptography", "SHA-256 / SHA-512 / SHA3-256 / SHA3-512"),
            ("Public-address research", "Local shape validation; no private-key operations"),
            ("P2P", "Cooperative asyncio peer messaging with content verification"),
            ("Client / Server", "Local HTTP research-job API"),
            ("RNN / LLM", "16-dimensional recurrent state + optional injected LLM"),
            ("Logging", "Live activity log with JSON/text export"),
        ]
        for name, detail in modules:
            row = ttk.Frame(tab, padding=8)
            row.pack(fill="x", pady=3)
            ttk.Label(row, text="●", width=3).pack(side="left")
            ttk.Label(row, text=name, width=25, font=("TkDefaultFont", 10, "bold")).pack(side="left")
            ttk.Label(row, text=detail).pack(side="left", fill="x")
        ttk.Separator(tab).pack(fill="x", pady=15)
        ttk.Label(tab, text="Safety boundary", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        ttk.Label(tab, text="Only public-address analysis is supported. Private keys, seed phrases, brute-force discovery,\nwallet draining, and automatic transfers are not part of this application.").pack(anchor="w", pady=6)

    def _build_crypto_tab(self) -> None:
        tab = ttk.Frame(self.tabs, padding=12)
        self.tabs.add(tab, text="Crypto Lab")
        ttk.Label(tab, text="Input text", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        self.crypto_input = tk.Text(tab, height=7, wrap="word")
        self.crypto_input.pack(fill="x", pady=6)
        self.crypto_input.insert("1.0", "crypto research")
        ttk.Button(tab, text="Run all hashes", command=self.run_hashes).pack(anchor="w", pady=4)
        frame, self.crypto_output = self._text_area(tab, 18)
        frame.pack(fill="both", expand=True, pady=6)

    def run_hashes(self) -> None:
        text = self.crypto_input.get("1.0", "end-1c")
        report = build_digest_report(text)
        self.crypto_output.delete("1.0", "end")
        self.crypto_output.insert("end", format_json(report))
        self.status_var.set("HASH COMPLETE")
        self.log("Cryptographic digest set calculated.")

    def _build_address_tab(self) -> None:
        tab = ttk.Frame(self.tabs, padding=12)
        self.tabs.add(tab, text="Address Research")
        ttk.Label(tab, text="Public Bitcoin / EVM address", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        self.address_var = tk.StringVar()
        ttk.Entry(tab, textvariable=self.address_var).pack(fill="x", pady=6)
        ttk.Button(tab, text="Validate / Analyze", command=self.run_address).pack(anchor="w")
        frame, self.address_output = self._text_area(tab, 15)
        frame.pack(fill="both", expand=True, pady=8)
        ttk.Label(tab, text="Analysis is offline and shape-based; it does not query balances or discover credentials.").pack(anchor="w")

    def run_address(self) -> None:
        result = validate_public_address(self.address_var.get())
        self.address_output.delete("1.0", "end")
        self.address_output.insert("end", format_json(result))
        self.status_var.set("ADDRESS ANALYZED")
        self.log("Public address analyzed locally.")

    def _build_p2p_tab(self) -> None:
        tab = ttk.Frame(self.tabs, padding=12)
        self.tabs.add(tab, text="P2P Network")
        form = ttk.Frame(tab)
        form.pack(fill="x")
        self.node_var = tk.StringVar(value="chimera-gui-node")
        self.p2p_host_var = tk.StringVar(value="127.0.0.1")
        self.p2p_port_var = tk.StringVar(value="8765")
        self.peer_host_var = tk.StringVar(value="127.0.0.1")
        self.peer_port_var = tk.StringVar(value="8765")
        self.peer_kind_var = tk.StringVar(value="research")
        fields = [("Node", self.node_var), ("Listen host", self.p2p_host_var), ("Listen port", self.p2p_port_var), ("Peer host", self.peer_host_var), ("Peer port", self.peer_port_var), ("Kind", self.peer_kind_var)]
        for i, (label, var) in enumerate(fields):
            ttk.Label(form, text=label).grid(row=i // 3 * 2, column=i % 3, sticky="w", padx=4, pady=(3, 0))
            ttk.Entry(form, textvariable=var, width=24).grid(row=i // 3 * 2 + 1, column=i % 3, sticky="ew", padx=4, pady=(0, 6))
        for c in range(3): form.columnconfigure(c, weight=1)
        ttk.Label(tab, text="Payload JSON").pack(anchor="w")
        self.p2p_payload = tk.Text(tab, height=6)
        self.p2p_payload.pack(fill="x", pady=5)
        self.p2p_payload.insert("1.0", '{"message": "hello from Chimera GUI"}')
        buttons = ttk.Frame(tab); buttons.pack(fill="x", pady=5)
        ttk.Button(buttons, text="Start peer", command=self.start_p2p).pack(side="left", padx=3)
        ttk.Button(buttons, text="Stop peer", command=self.stop_p2p).pack(side="left", padx=3)
        ttk.Button(buttons, text="Send message", command=self.send_p2p).pack(side="left", padx=3)
        frame, self.p2p_output = self._text_area(tab, 13); frame.pack(fill="both", expand=True, pady=6)

    def start_p2p(self) -> None:
        if self.p2p_thread and self.p2p_thread.is_alive():
            self.log("P2P peer is already running."); return
        try: port = int(self.p2p_port_var.get())
        except ValueError: messagebox.showerror("P2P", "Listen port must be an integer."); return
        self.p2p_loop = asyncio.new_event_loop()
        self.p2p_peer = P2PPeer(self.node_var.get(), self._on_peer_message)
        async def runner():
            async def handler(reader, writer):
                await self.p2p_peer._handle_connection(reader, writer)  # type: ignore[attr-defined]
            self.p2p_server = await asyncio.start_server(handler, self.p2p_host_var.get(), port)
            self.log(f"P2P listening on {self.p2p_host_var.get()}:{port}")
            async with self.p2p_server: await self.p2p_server.serve_forever()
        def run():
            asyncio.set_event_loop(self.p2p_loop)
            try: self.p2p_loop.run_until_complete(runner())
            except (asyncio.CancelledError, RuntimeError, OSError) as exc: self.log(f"P2P stopped: {exc}")
        self.p2p_thread = threading.Thread(target=run, daemon=True); self.p2p_thread.start()
        self.status_var.set("P2P ACTIVE")

    async def _on_peer_message(self, msg) -> None:
        self.events.put(f"P2P message from {msg.sender}: {format_json(msg.payload)}")

    def stop_p2p(self) -> None:
        if self.p2p_loop and self.p2p_server:
            self.p2p_loop.call_soon_threadsafe(self.p2p_server.close)
            self.p2p_loop.call_soon_threadsafe(self.p2p_loop.stop)
        self.status_var.set("P2P STOPPED")
        self.log("P2P listener stop requested.")

    def send_p2p(self) -> None:
        try:
            payload = json.loads(self.p2p_payload.get("1.0", "end-1c") or "{}")
            port = int(self.peer_port_var.get())
        except (ValueError, json.JSONDecodeError) as exc:
            messagebox.showerror("P2P", f"Invalid input: {exc}"); return
        msg = build_peer_message(self.node_var.get(), self.peer_kind_var.get(), payload)
        def worker():
            try: asyncio.run(P2PPeer(self.node_var.get()).send(self.peer_host_var.get(), port, msg)); self.log("P2P message sent and content digest attached.")
            except Exception as exc: self.log(f"P2P send failed: {exc}")
        threading.Thread(target=worker, daemon=True).start()

    def _build_server_tab(self) -> None:
        tab = ttk.Frame(self.tabs, padding=12); self.tabs.add(tab, text="Client / Server")
        self.server_host_var = tk.StringVar(value="127.0.0.1"); self.server_port_var = tk.StringVar(value="8787")
        row = ttk.Frame(tab); row.pack(fill="x")
        ttk.Label(row, text="Host").pack(side="left"); ttk.Entry(row, textvariable=self.server_host_var, width=20).pack(side="left", padx=5)
        ttk.Label(row, text="Port").pack(side="left"); ttk.Entry(row, textvariable=self.server_port_var, width=10).pack(side="left", padx=5)
        ttk.Button(row, text="Start server", command=self.start_server).pack(side="left", padx=5)
        ttk.Button(row, text="Stop server", command=self.stop_server).pack(side="left", padx=5)
        ttk.Separator(tab).pack(fill="x", pady=12)
        self.operation_var = tk.StringVar(value="hash")
        ttk.Label(tab, text="Research operation").pack(anchor="w")
        ttk.Combobox(tab, textvariable=self.operation_var, values=("hash", "address_validate", "model_inference"), state="readonly").pack(anchor="w", pady=4)
        self.server_payload = tk.Text(tab, height=7); self.server_payload.pack(fill="x", pady=5); self.server_payload.insert("1.0", '{"text": "client server research"}')
        ttk.Button(tab, text="Submit job", command=self.submit_job).pack(anchor="w")
        frame, self.server_output = self._text_area(tab, 15); frame.pack(fill="both", expand=True, pady=7)

    def start_server(self) -> None:
        if self.http_server is not None: self.log("HTTP server is already running."); return
        try: port = int(self.server_port_var.get())
        except ValueError: messagebox.showerror("Server", "Port must be an integer."); return
        self.http_server = ThreadingHTTPServer((self.server_host_var.get(), port), ResearchHandler)
        self.http_thread = threading.Thread(target=self.http_server.serve_forever, daemon=True); self.http_thread.start()
        self.status_var.set("SERVER ACTIVE"); self.log(f"Research server listening on {self.server_host_var.get()}:{port}")

    def stop_server(self) -> None:
        if self.http_server:
            self.http_server.shutdown(); self.http_server.server_close(); self.http_server = None
            self.log("Research server stopped.")
        self.status_var.set("SERVER STOPPED")

    def submit_job(self) -> None:
        try: payload = json.loads(self.server_payload.get("1.0", "end-1c") or "{}")
        except json.JSONDecodeError as exc: messagebox.showerror("Client", str(exc)); return
        if self.http_server is None: self.start_server()
        base = f"http://{self.server_host_var.get()}:{self.server_port_var.get()}"
        operation = self.operation_var.get()
        def worker():
            try:
                result = submit(base, operation, payload); self.events.put("SERVER RESULT\n" + format_json(result))
            except Exception as exc: self.events.put(f"CLIENT ERROR: {exc}")
        threading.Thread(target=worker, daemon=True).start()

    def _build_rnn_tab(self) -> None:
        tab = ttk.Frame(self.tabs, padding=12); self.tabs.add(tab, text="RNN / LLM")
        ttk.Label(tab, text="Analysis prompt / text").pack(anchor="w")
        self.rnn_input = tk.Text(tab, height=8); self.rnn_input.pack(fill="x", pady=6); self.rnn_input.insert("1.0", "Analyze this cryptographic research message")
        ttk.Button(tab, text="Run RNN + LLM facade", command=self.run_rnn).pack(anchor="w")
        frame, self.rnn_output = self._text_area(tab, 18); frame.pack(fill="both", expand=True, pady=7)
        ttk.Label(tab, text="The default adapter is local-only. An application may inject its own LLM callable.").pack(anchor="w")

    def run_rnn(self) -> None:
        text = self.rnn_input.get("1.0", "end-1c")
        self.rnn_output.delete("1.0", "end"); self.rnn_output.insert("end", format_json(build_analysis_report(text)))
        self.status_var.set("RNN ANALYSIS COMPLETE"); self.log("RNN/LLM facade analysis completed locally.")

    def _build_logs_tab(self) -> None:
        tab = ttk.Frame(self.tabs, padding=12); self.tabs.add(tab, text="Logs / Export")
        frame, self.log_output = self._text_area(tab, 25); frame.pack(fill="both", expand=True)
        row = ttk.Frame(tab); row.pack(fill="x", pady=7)
        ttk.Button(row, text="Export text log", command=self.export_log).pack(side="left", padx=3)
        ttk.Button(row, text="Clear log", command=lambda: self.log_output.delete("1.0", "end")).pack(side="left", padx=3)

    def log(self, message: str) -> None:
        self.events.put(message)

    def _drain_events(self) -> None:
        while True:
            try: message = self.events.get_nowait()
            except queue.Empty: break
            stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
            self.log_output.insert("end", f"[{stamp}] {message}\n")
            self.log_output.see("end")
            if hasattr(self, "p2p_output") and message.startswith("P2P message"):
                self.p2p_output.insert("end", message + "\n")
            if hasattr(self, "server_output") and (message.startswith("SERVER RESULT") or message.startswith("CLIENT ERROR")):
                self.server_output.insert("end", message + "\n")
        self.after(100, self._drain_events)

    def export_log(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".log", filetypes=[("Log", "*.log"), ("Text", "*.txt")])
        if path:
            with open(path, "w", encoding="utf-8") as fh: fh.write(self.log_output.get("1.0", "end-1c"))
            self.log(f"Log exported to {path}")

    def _close(self) -> None:
        try: self.stop_server(); self.stop_p2p()
        finally: self.destroy()


def main() -> None:
    CryptoResearchGUI().mainloop()


if __name__ == "__main__":
    main()
