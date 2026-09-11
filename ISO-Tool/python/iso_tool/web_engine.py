from __future__ import annotations
from dataclasses import dataclass, asdict
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse, quote_plus
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser
import hashlib, json, time
USER_AGENT='ISO-Tool-Crawler/1.0'
@dataclass
class WebRecord: url:str; title:str; text:str; retrieved_at:str; sha256:str; source_type:str='web'; status:int=200
class _HTML(HTMLParser):
 def __init__(self): super().__init__();self.links=[];self.text=[];self.title=[];self.in_title=False
 def handle_starttag(self,tag,attrs):
  d=dict(attrs)
  if tag.lower()=='a' and d.get('href'):self.links.append(d['href'])
  if tag.lower()=='title':self.in_title=True
 def handle_endtag(self,tag):
  if tag.lower()=='title':self.in_title=False
 def handle_data(self,data):(self.title if self.in_title else self.text).append(data)
class WebCrawler:
 def __init__(self,cache_dir,max_pages=25,max_depth=2,delay=.25):
  self.cache_dir=Path(cache_dir);self.cache_dir.mkdir(parents=True,exist_ok=True);self.max_pages=max(1,int(max_pages));self.max_depth=max(0,int(max_depth));self.delay=max(0,float(delay));self._robots_cache={};self._seen=set()
 def _robots_for(self,url):
  p=urlparse(url);origin=f'{p.scheme}://{p.netloc}';now=time.time();c=self._robots_cache.get(origin)
  if c and now-c[0]<86400:return c[1]
  rp=RobotFileParser(origin+'/robots.txt')
  try:
   with urlopen(Request(rp.url,headers={'User-Agent':USER_AGENT}),timeout=10) as r:rp.parse(r.read().decode('utf-8','replace').splitlines())
  except Exception:rp.parse(['User-agent: *','Disallow: /'])
  self._robots_cache[origin]=(now,rp);return rp
 def fetch(self,url):
  with urlopen(Request(url,headers={'User-Agent':USER_AGENT}),timeout=20) as r:return getattr(r,'status',200),r.read(2000000).decode(r.headers.get_content_charset() or 'utf-8','replace')
 def crawl(self,start_url):
  p=urlparse(start_url)
  if p.scheme not in {'http','https'}:raise ValueError('Only HTTP(S) URLs are supported')
  q=[(start_url,0)];out=[];host=p.netloc
  while q and len(out)<self.max_pages:
   url,depth=q.pop(0)
   if url in self._seen or depth>self.max_depth:continue
   self._seen.add(url)
   if not self._robots_for(url).can_fetch(USER_AGENT,url):continue
   try:status,html=self.fetch(url)
   except Exception:continue
   parser=_HTML();parser.feed(html);text=' '.join(' '.join(parser.text).split());title=' '.join(' '.join(parser.title).split());digest=hashlib.sha256(html.encode('utf-8','replace')).hexdigest();out.append(WebRecord(url,title,text,time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),digest,status=status))
   if depth<self.max_depth:
    for href in parser.links:
     nxt=urljoin(url,href).split('#',1)[0]
     if urlparse(nxt).netloc==host and nxt not in self._seen:q.append((nxt,depth+1))
   time.sleep(self.delay)
  return out
def search_web(query,endpoint='https://html.duckduckgo.com/html/?q=',limit=10):
 with urlopen(Request(endpoint+quote_plus(query),headers={'User-Agent':USER_AGENT}),timeout=20) as r:html=r.read().decode('utf-8','replace')
 p=_HTML();p.feed(html);urls=[]
 for u in p.links:
  if u.startswith('http') and u not in urls:urls.append(u)
  if len(urls)>=limit:break
 return [{'url':u,'query':query,'source_type':'search'} for u in urls]
def write_records(path,records):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text('\n'.join(json.dumps(asdict(r) if hasattr(r,'__dataclass_fields__') else r,ensure_ascii=False) for r in records),encoding='utf-8');return path
