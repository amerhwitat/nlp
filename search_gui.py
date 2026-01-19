# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 19:03:31 2026

@author: PC1
search_engines.py
"""
"""
search_gui.py

Tkinter GUI wrapper for web search functions:
- Bing Web Search API (requires key)
- Google Custom Search API (requires key + cx)
- DuckDuckGo HTML fallback (lightweight)
"""

import time
import csv
import webbrowser
import requests
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from bs4 import BeautifulSoup
from typing import List, Dict, Optional

# -------------------------
# Search implementations
# -------------------------
BASE_HEADERS = {"User-Agent": "python-requests/2.x (+https://example.com)"}

def safe_get(url: str, params: dict = None, headers: dict = None, timeout: int = 10):
    r = requests.get(url, params=params, headers=headers or BASE_HEADERS, timeout=timeout)
    r.raise_for_status()
    return r

def search_bing(query: str, api_key: str, count: int = 10) -> List[Dict]:
    endpoint = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {"q": query, "count": count, "mkt": "en-US"}
    r = safe_get(endpoint, params=params, headers=headers)
    data = r.json()
    results = []
    for item in data.get("webPages", {}).get("value", []):
        results.append({"title": item.get("name"), "snippet": item.get("snippet"), "url": item.get("url")})
    return results

def search_google(query: str, api_key: str, cx: str, num: int = 10) -> List[Dict]:
    endpoint = "https://www.googleapis.com/customsearch/v1"
    params = {"key": api_key, "cx": cx, "q": query, "num": min(num, 10)}
    r = safe_get(endpoint, params=params)
    data = r.json()
    results = []
    for item in data.get("items", []):
        results.append({"title": item.get("title"), "snippet": item.get("snippet"), "url": item.get("link")})
    return results

def search_duckduckgo_html(query: str, max_results: int = 10) -> List[Dict]:
    url = "https://html.duckduckgo.com/html/"
    params = {"q": query}
    r = safe_get(url, params=params)
    soup = BeautifulSoup(r.text, "html.parser")
    results = []
    anchors = soup.select("a.result__a, a.result-link")
    for a in anchors[:max_results]:
        title = a.get_text(strip=True)
        href = a.get("href")
        snippet_tag = a.find_parent().select_one(".result__snippet, .result-snippet")
        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
        results.append({"title": title, "snippet": snippet, "url": href})
    return results

def unified_search(query: str,
                   engine: str = "bing",
                   bing_key: Optional[str] = None,
                   google_key: Optional[str] = None,
                   google_cx: Optional[str] = None,
                   limit: int = 10) -> List[Dict]:
    """
    Try preferred engine first, fall back to others. Use API keys when provided.
    """
    engine = engine.lower()
    # Try preferred
    if engine == "bing" and bing_key:
        try:
            return search_bing(query, api_key=bing_key, count=limit)
        except Exception:
            pass
    if engine == "google" and google_key and google_cx:
        try:
            return search_google(query, api_key=google_key, cx=google_cx, num=limit)
        except Exception:
            pass
    # Try other APIs
    if bing_key:
        try:
            return search_bing(query, api_key=bing_key, count=limit)
        except Exception:
            pass
    if google_key and google_cx:
        try:
            return search_google(query, api_key=google_key, cx=google_cx, num=limit)
        except Exception:
            pass
    # Final fallback: DuckDuckGo HTML
    time.sleep(0.4)  # polite pause
    return search_duckduckgo_html(query, max_results=limit)

# -------------------------
# GUI
# -------------------------
class SearchGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Web Search GUI")
        self.geometry("980x640")
        self.create_widgets()
        self.results = []  # list of dicts

    def create_widgets(self):
        pad = {"padx": 6, "pady": 6}

        top = ttk.Frame(self)
        top.pack(fill=tk.X, **pad)

        ttk.Label(top, text="Query").grid(column=0, row=0, sticky=tk.W)
        self.query_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.query_var, width=60).grid(column=1, row=0, columnspan=3, sticky=tk.W)

        ttk.Label(top, text="Engine").grid(column=0, row=1, sticky=tk.W)
        self.engine_var = tk.StringVar(value="bing")
        engine_menu = ttk.Combobox(top, textvariable=self.engine_var, values=["bing", "google", "duckduckgo"], width=12, state="readonly")
        engine_menu.grid(column=1, row=1, sticky=tk.W)

        ttk.Label(top, text="Limit").grid(column=2, row=1, sticky=tk.W)
        self.limit_var = tk.IntVar(value=10)
        ttk.Spinbox(top, from_=1, to=50, textvariable=self.limit_var, width=6).grid(column=3, row=1, sticky=tk.W)

        # API keys area
        keys = ttk.LabelFrame(self, text="API Keys (optional, stored only in memory)")
        keys.pack(fill=tk.X, **pad)

        ttk.Label(keys, text="Bing Key").grid(column=0, row=0, sticky=tk.W)
        self.bing_key_var = tk.StringVar()
        ttk.Entry(keys, textvariable=self.bing_key_var, width=80, show="*").grid(column=1, row=0, columnspan=3, sticky=tk.W)

        ttk.Label(keys, text="Google Key").grid(column=0, row=1, sticky=tk.W)
        self.google_key_var = tk.StringVar()
        ttk.Entry(keys, textvariable=self.google_key_var, width=50, show="*").grid(column=1, row=1, sticky=tk.W)

        ttk.Label(keys, text="Google CX").grid(column=2, row=1, sticky=tk.W)
        self.google_cx_var = tk.StringVar()
        ttk.Entry(keys, textvariable=self.google_cx_var, width=28).grid(column=3, row=1, sticky=tk.W)

        # Buttons
        btns = ttk.Frame(self)
        btns.pack(fill=tk.X, **pad)
        ttk.Button(btns, text="Search", command=self.on_search).grid(column=0, row=0, **pad)
        ttk.Button(btns, text="Open Selected", command=self.open_selected).grid(column=1, row=0, **pad)
        ttk.Button(btns, text="Export CSV", command=self.export_csv).grid(column=2, row=0, **pad)
        ttk.Button(btns, text="Clear", command=self.clear_results).grid(column=3, row=0, **pad)

        # Results list
        results_frame = ttk.Frame(self)
        results_frame.pack(fill=tk.BOTH, expand=True, **pad)
        self.tree = ttk.Treeview(results_frame, columns=("title", "snippet", "url"), show="headings", selectmode="browse")
        self.tree.heading("title", text="Title")
        self.tree.heading("snippet", text="Snippet")
        self.tree.heading("url", text="URL")
        self.tree.column("title", width=320)
        self.tree.column("snippet", width=420)
        self.tree.column("url", width=220)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Double-click opens URL
        self.tree.bind("<Double-1>", lambda e: self.open_selected())

    def on_search(self):
        query = self.query_var.get().strip()
        if not query:
            messagebox.showinfo("Input required", "Please enter a search query.")
            return
        engine = self.engine_var.get()
        limit = max(1, min(50, self.limit_var.get()))
        #bing_key = self.bing_key_var.get().strip() #or None
        #google_key = self.google_key_var.get().strip()# or None
        #google_cx = self.google_cx_var.get().strip()# or None
        USER_AGENT = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}
        #assert isinstance(tk.StringVar(), str), 'Search term must be a string'
        #assert isinstance(self.limit_var, int), 'Number of results must be an integer'
        escaped_search_term = tk.StringVar()
        google_url = 'https://www.google.com/search?q={}&num={}&hl={}'.format(escaped_search_term, tk.StringVar(), 'en')
        response = requests.get(google_url, headers=USER_AGENT)
        # Disable UI while searching
        self._set_ui_state("disabled")
        try:
            self.results = response#unified_search(query, engine=engine, bing_key=bing_key, google_key=google_key, google_cx=google_cx, limit=limit)
            self._populate_results()
        except Exception as e:
            messagebox.showerror("Search error", str(e))
        finally:
            self._set_ui_state("normal")

    def _populate_results(self):
        self.tree.delete(*self.tree.get_children())
        for i, r in enumerate(self.results):
            title = r.get("title") or ""
            snippet = (r.get("snippet") or "").replace("\n", " ")
            url = r.get("url") or ""
            self.tree.insert("", "end", iid=str(i), values=(title, snippet, url))

    def open_selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Select result", "Select a result to open.")
            return
        idx = int(sel[0])
        url = self.results[idx].get("url")
        if not url:
            messagebox.showinfo("No URL", "Selected result has no URL.")
            return
        webbrowser.open(url)

    def export_csv(self):
        if not self.results:
            messagebox.showinfo("No results", "Run a search first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files","*.csv")])
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["title", "snippet", "url"])
                for r in self.results:
                    writer.writerow([r.get("title",""), r.get("snippet",""), r.get("url","")])
            messagebox.showinfo("Exported", f"Results exported to {path}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))

    def clear_results(self):
        self.results = []
        self.tree.delete(*self.tree.get_children())

    def _set_ui_state(self, state: str):
        # enable/disable inputs while searching
        for child in self.winfo_children():
            try:
                child.configure(state=state)
            except Exception:
                pass
        # keep tree enabled for viewing
        if state == "disabled":
            self.tree.configure(selectmode="none")
        else:
            self.tree.configure(selectmode="browse")

if __name__ == "__main__":
    app = SearchGUI()
    app.mainloop()