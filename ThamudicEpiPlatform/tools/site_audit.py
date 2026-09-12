#!/usr/bin/env python3
"""Bounded public-site asset auditor.

It records HTML-discovered script/style/image/link assets and hashes fetched
textual assets. It intentionally does not copy or republish third-party
bundles. Use only on sites you are authorized to inspect.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from collections import deque
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup

DEFAULTS = [
 'https://thamudicscan-s3wz30.public.builtwithrocket.new/',
 'https://chimera-ii-os-730893.onhercules.app/',
 'https://thamudic-scanner.softr.app/',
]

def audit(root: str, max_pages: int, delay: float):
    host = urlparse(root).netloc
    q = deque([root]); seen=set(); assets=[]
    session=requests.Session(); session.headers['User-Agent']='nlp-thamudic-site-auditor/1.0 (+research)'
    while q and len(seen)<max_pages:
        url=q.popleft()
        if url in seen: continue
        seen.add(url)
        try:
            r=session.get(url, timeout=20); ct=r.headers.get('content-type','')
        except requests.RequestException as exc:
            assets.append({'site_url':root,'asset_url':url,'asset_type':'page','error':str(exc)})
            continue
        assets.append({'site_url':root,'asset_url':url,'asset_type':'page','http_status':r.status_code,'content_type':ct})
        if 'text/html' not in ct: continue
        soup=BeautifulSoup(r.text,'html.parser')
        for tag,attr,kind in [('script','src','script'),('link','href','style-or-link'),('img','src','image')]:
            for node in soup.find_all(tag):
                raw=node.get(attr)
                if not raw: continue
                u=urljoin(url,raw)
                if urlparse(u).scheme not in ('http','https'): continue
                assets.append({'site_url':root,'asset_url':u,'asset_type':kind,'source_page':url})
                if urlparse(u).netloc==host and u not in seen and len(seen)<max_pages:
                    q.append(u)
        time.sleep(delay)
    return {'root':root,'pages_scanned':len(seen),'assets':assets,'generated_at':time.time()}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='data/site-audit.json'); ap.add_argument('--max-pages',type=int,default=12); ap.add_argument('--delay',type=float,default=1.0); ap.add_argument('urls',nargs='*',default=DEFAULTS)
    a=ap.parse_args(); results=[]
    for u in a.urls:
        results.append(audit(u,a.max_pages,a.delay))
    import pathlib; pathlib.Path(a.out).parent.mkdir(parents=True,exist_ok=True); pathlib.Path(a.out).write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'wrote {a.out}')

if __name__=='__main__': main()
