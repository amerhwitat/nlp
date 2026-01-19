# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 19:39:27 2026

@author: PC1
"""

"""
search_gui_requests.py

Search GUI that uses requests + BeautifulSoup to fetch and parse HTML search results.
Supports DuckDuckGo (html.duckduckgo.com) and Bing (bing.com/search) HTML parsing.
Use responsibly and avoid high-frequency scraping.
"""

import time
import csv
import webbrowser
import requests
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlencode

# Default headers (change User-Agent if needed)
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0 Safari/537.36"
}

# -------------------------
# HTML search implementations
# -------------------------
def safe_get(url: str, params: dict = None, headers: dict = None, timeout: int = 12):
    hdrs = headers or DEFAULT_HEADERS
    resp = requests.get(url, params=params, headers=hdrs, timeout=timeout)
    resp.raise_for_status()
    return resp

def parse_duckduckgo_html(query: str, max_results: int = 10) -> List[Dict]:
    """
    Uses DuckDuckGo's lightweight HTML endpoint. Good for small, personal queries.
    """
    url = "https://html.duckduckgo.com/html/"
    params = {"q": query}
    r = safe_get(url, params=params)
    soup = BeautifulSoup(r.text, "html.parser")
    results = []
    # DuckDuckGo HTML uses result links in <a class="result__a"> or <a.result-link>
    anchors = soup.select("a.result__a, a.result-link")
    for a in anchors[:max_results]:
        title = a.get_text(strip=True)
        href = a.get("href")
        # snippet may be in sibling element
        snippet_tag = a.find_parent().select_one(".result__snippet, .result-snippet")
        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
        results.append({"title": title, "snippet": snippet, "url": href})
    return results

def parse_bing_html(query: str, max_results: int = 10) -> List[Dict]:
    """
    Parse Bing search results page. HTML structure can change; this parser is best-effort.
    """
    url = "https://www.bing.com/search"
    params = {"q": query, "count": max_results}
    r = safe_get(url, params=params)
    soup = BeautifulSoup(r.text, "html.parser")
    results = []
    # Bing result blocks often use <li class="b_algo">
    for li in soup.select("li.b_algo")[:max_results]:
        h2 = li.find("h2")
        a = h2.find("a") if h2 else None
        title = a.get_text(strip=True) if a else ""
        href = a.get("href") if a else ""
        snippet_tag = li.select_one("p")
        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
        results.append({"title": title, "snippet": snippet, "url": href})
    # Fallback: look for generic anchors if none found
    if not results:
        anchors = soup.select("a")
        for a in anchors[:max_results]:
            title = a.get_text(strip=True)
            href = a.get("href")
            if href and title:
                results.append({"title": title, "snippet": "", "url": href})
    return results

def unified_html_search(query: str, engine: str = "duckduckgo", limit: int = 10) -> List[Dict]:
    """
    Unified HTML-based search. engine: 'duckduckgo' or 'bing'.
    """
    engine = engine.lower()
    # polite pause to avoid rapid-fire requests
    time.sleep(0.3)
    if engine == "bing":
        return parse_bing_html(query, max_results=limit)
    else:
        return parse_duckduckgo_html(query, max_results=limit)

# -------------------------
# GUI
# -------------------------
class SearchGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HTML Search GUI (requests + BeautifulSoup)")
        self.geometry("980x640")
        self.results = []
        self.create_widgets()

    def create_widgets(self):
        pad = {"padx": 6, "pady": 6}

        top = ttk.Frame(self)
        top.pack(fill=tk.X, **pad)

        ttk.Label(top, text="Query").grid(column=0, row=0, sticky=tk.W)
        self.query_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.query_var, width=60).grid(column=1, row=0, columnspan=3, sticky=tk.W)

        ttk.Label(top, text="Engine").grid(column=0, row=1, sticky=tk.W)
        self.engine_var = tk.StringVar(value="duckduckgo")
        engine_menu = ttk.Combobox(top, textvariable=self.engine_var, values=["duckduckgo", "bing"], width=14, state="readonly")
        engine_menu.grid(column=1, row=1, sticky=tk.W)

        ttk.Label(top, text="Limit").grid(column=2, row=1, sticky=tk.W)
        self.limit_var = tk.IntVar(value=10)
        ttk.Spinbox(top, from_=1, to=50, textvariable=self.limit_var, width=6).grid(column=3, row=1, sticky=tk.W)

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

        # Footer note
        note = ("Note: This tool fetches and parses public HTML search result pages using requests. "
                "Use responsibly and avoid high-frequency scraping.")
        ttk.Label(self, text=note, foreground="darkred", wraplength=920).pack(padx=8, pady=(0,8), anchor=tk.W)

    def on_search(self):
        query = self.query_var.get().strip()
        if not query:
            messagebox.showinfo("Input required", "Please enter a search query.")
            return
        engine = self.engine_var.get()
        limit = max(1, min(50, self.limit_var.get()))
        # disable UI while searching
        self._set_ui_state("disabled")
        try:
            self.results = unified_html_search(query, engine=engine, limit=limit)
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
        # enable/disable top-level widgets while searching
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