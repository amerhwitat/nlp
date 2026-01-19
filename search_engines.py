# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 19:01:41 2026

@author: PC1
"""

"""
search_engines.py

Simple, reusable functions to search Bing, Google Custom Search, and DuckDuckGo (HTML).
Prefer official APIs (Bing/Google). DuckDuckGo fallback parses HTML and is for light use.
"""

import requests
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import time

# -------------------------
# Helpers
# -------------------------
def safe_get(url: str, params: dict = None, headers: dict = None, timeout: int = 10):
    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp

# -------------------------
# Bing Web Search API
# -------------------------
def search_bing(query: str, api_key: str, count: int = 10, offset: int = 0) -> List[Dict]:
    """
    Uses Bing Web Search API v7. Returns list of results with title, snippet, url.
    Requires an Azure Bing Search subscription key.
    """
    endpoint = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {"q": query, "count": count, "offset": offset, "mkt": "en-US"}
    r = safe_get(endpoint, params=params, headers=headers)
    data = r.json()
    results = []
    web_pages = data.get("webPages", {}).get("value", [])
    for item in web_pages:
        results.append({
            "title": item.get("name"),
            "snippet": item.get("snippet"),
            "url": item.get("url")
        })
    return results

# -------------------------
# Google Custom Search API
# -------------------------
def search_google(query: str, api_key: str, cx: str, num: int = 10, start: int = 1) -> List[Dict]:
    """
    Uses Google Custom Search JSON API. Requires API key and cx (search engine id).
    Returns list of results with title, snippet, url.
    """
    endpoint = "https://www.googleapis.com/customsearch/v1"
    params = {"key": api_key, "cx": cx, "q": query, "num": min(num, 10), "start": start}
    r = safe_get(endpoint, params=params)
    data = r.json()
    results = []
    for item in data.get("items", []):
        results.append({
            "title": item.get("title"),
            "snippet": item.get("snippet"),
            "url": item.get("link")
        })
    return results

# -------------------------
# DuckDuckGo HTML fallback
# -------------------------
def search_duckduckgo_html(query: str, max_results: int = 10) -> List[Dict]:
    """
    Lightweight DuckDuckGo HTML search fallback. Parses the HTML results page.
    Intended for small-scale personal use only.
    """
    url = "https://html.duckduckgo.com/html/"
    headers = {"User-Agent": "python-requests/2.x (+https://example.com)"}
    params = {"q": query}
    r = safe_get(url, params=params, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")
    results = []
    # DuckDuckGo HTML uses result links in <a class="result__a"> or <a class="result-link">
    anchors = soup.select("a.result__a, a.result-link")
    for a in anchors[:max_results]:
        title = a.get_text(strip=True)
        href = a.get("href")
        # snippet may be in sibling element
        snippet_tag = a.find_parent().select_one(".result__snippet, .result-snippet")
        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
        results.append({"title": title, "snippet": snippet, "url": href})
    return results

# -------------------------
# Unified search function
# -------------------------
def search_web(query: str,
               bing_key: Optional[str] = None,
               google_key: Optional[str] = None,
               google_cx: Optional[str] = None,
               prefer: str = "bing",
               limit: int = 10) -> List[Dict]:
    """
    Unified search that tries preferred API and falls back.
    - prefer: "bing", "google", or "duckduckgo"
    - Provide API keys for Bing or Google to use those services.
    """
    # Try preferred API first
    if prefer == "bing" and bing_key:
        try:
            return search_bing(query, api_key=bing_key, count=limit)
        except Exception as e:
            # fall through to other options
            print("Bing search failed:", e)
    if prefer == "google" and google_key and google_cx:
        try:
            return search_google(query, api_key=google_key, cx=google_cx, num=limit)
        except Exception as e:
            print("Google search failed:", e)

    # Try the other API if available
    if bing_key:
        try:
            return search_bing(query, api_key=bing_key, count=limit)
        except Exception as e:
            print("Bing fallback failed:", e)
    if google_key and google_cx:
        try:
            return search_google(query, api_key=google_key, cx=google_cx, num=limit)
        except Exception as e:
            print("Google fallback failed:", e)

    # Final fallback: DuckDuckGo HTML
    try:
        # polite pause to avoid rapid requests
        time.sleep(0.5)
        return search_duckduckgo_html(query, max_results=limit)
    except Exception as e:
        print("DuckDuckGo fallback failed:", e)
        return []

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    q = "python web scraping best practices"
    # Example: use environment variables or config to store keys
    BING_KEY = None   # set your Bing key here
    GOOGLE_KEY = None # set your Google API key here
    GOOGLE_CX = None  # set your Google custom search engine id here

    results = search_web(q, bing_key=BING_KEY, google_key=GOOGLE_KEY, google_cx=GOOGLE_CX, prefer="bing", limit=5)
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}\n")