#!/usr/bin/env python3
"""Bounded, robots-aware public-site asset auditor.

Records HTML-discovered assets and hashes fetched text assets. It does not
republish third-party bundles. Use only on sites you are authorized to inspect.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import requests
from bs4 import BeautifulSoup

DEFAULTS = [
    'https://thamudicscan-s3wz30.public.builtwithrocket.new/',
    'https://chimera-ii-os-730893.onhercules.app/',
    'https://thamudic-scanner.softr.app/',
]
UA='nlp-thamudic-site-auditor/1.1 (+research)'

def audit(root: str, max_pages: int, delay: float):
    host=urlparse(root).netloc; q=deque([root]); seen=set(); assets=[]
    session=requests.Session(); session.headers['User-Agent']=UA
    robots=RobotFileParser(urljoin(root,'/robots.txt'))
    try:
        rr=session.get(urljoin(root,'/robots.txt'),timeout=10)
        if rr.ok: robots.parse(rr.text.splitlines())
        else: robots=RobotFileParser(); robots.parse(['User-agent: *','Disallow: /'])
    except requests.RequestException:
        robots=RobotFileParser(); robots.parse(['User-agent: *','Disallow: /'])
    while q and len(seen)<max_pages:
        url=q.popleft()
        if url in seen or not robots.can_fetch(UA,url): continue
        seen.add(url)
        try:
            r=session.get(url,timeout=20); ct=r.headers.get('content-type','')
        except requests.RequestException as exc:
            assets.append({'site_url':root,'asset_url':url,'asset_type':'page','error':str(exc)}); continue
        digest=hashlib.sha256(r.content).hexdigest()
        assets.append({'site_url':root,'asset_url':url,'asset_type':'page','http_status':r.status_code,'content_type':ct,'sha256':digest})
        if 'text/html' not in ct: continue
        soup=BeautifulSoup(r.text,'html.parser')
        for tag,attr,kind in [('script','src','script'),('link','href','style-or-link'),('img','src','image')]:
            for node in soup.find_all(tag):
                raw=node.get(attr)
                if not raw: continue
                u=urljoin(url,raw)
                if urlparse(u).scheme not in ('http','https'): continue
                allowed=robots.can_fetch(UA,u)
                rec={'site_url':root,'asset_url':u,'asset_type':kind,'source_page':url,'robots_allowed':allowed}
                if allowed:
                    try:
                        ar=session.get(u,timeout=15); rec.update({'http_status':ar.status_code,'content_type':ar.headers.get('content-type',''),'sha256':hashlib.sha256(ar.content).hexdigest()})
                    except requests.RequestException as exc: rec['error']=str(exc)
                assets.append(rec)
                if urlparse(u).netloc==host and u not in seen and len(seen)<max_pages and allowed: q.append(u)
        time.sleep(delay)
    return {'root':root,'pages_scanned':len(seen),'robots_checked':True,'assets':assets,'generated_at':time.time()}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='data/site-audit.json'); ap.add_argument('--max-pages',type=int,default=12); ap.add_argument('--delay',type=float,default=1.0); ap.add_argument('urls',nargs='*',default=DEFAULTS)
    a=ap.parse_args(); results=[audit(u,a.max_pages,a.delay) for u in a.urls]
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8'); print(f'wrote {a.out}')
if __name__=='__main__': main()
