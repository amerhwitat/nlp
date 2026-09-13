#!/usr/bin/env python3
"""Self-contained all-in-one NLP scanner."""
from __future__ import annotations
import argparse,json,os,re,subprocess,sys,tempfile
from pathlib import Path
from typing import Any

def runtime():
    if sys.platform.startswith("win") and os.environ.get("NLP_ALLOW_DUPLICATE_OPENMP","1")!="0":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
    for n in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(n,"1")
runtime()

FIRST,LAST=0x10A80,0x10A9F
TRANS=dict(zip(range(FIRST,FIRST+29),["h","l","ḥ","m","q","w","s2","r","b","t","s1","k","n","ḫ","ṣ","s3","f","ʼ","ʽ","ḍ","g","d","ġ","ṭ","z","ḏ","y","ṯ","ẓ"]))
ONA={chr(k):v for k,v in TRANS.items()}
VARIANTS=("Dadanitic","Safaitic","Hismaic","Taymanitic","Minaic","Thamudic B","Thamudic C","Thamudic D")
RANGES={
"ancient-north-arabian":((0x10A80,0x10A9F),"Old North Arabian","rtl"),
"ancient-egyptian":((0x13000,0x1342F),"Egyptian Hieroglyphs","ltr"),
"ugaritic":((0x10380,0x1039F),"Ugaritic","ltr"),"phoenician":((0x10900,0x1091F),"Phoenician","rtl"),
"old-south-arabian":((0x10A60,0x10A7F),"Old South Arabian","rtl"),"old-persian":((0x103A0,0x103DF),"Old Persian","ltr"),
"coptic":((0x2C80,0x2CFF),"Coptic","ltr"),"old-turkic":((0x10C00,0x10C4F),"Old Turkic","rtl-or-ltr")}

CORPUS=[
("TIJ 503","Safaitic","ytm bn ʿbny w wgm ʿl- ḫll -h","Ytm son ʿbny and he grieved for his friend","يتم بن عبني وحزن على صديقه","https://ociana.osu.edu/inscriptions/2400"),
("AH 311","Dadanitic","bḏkrh wdd ḏ{h}k","Bḏkrh loves {Ḏhk}","بذَكرَه يحب {ذهك}","https://ociana.osu.edu/inscriptions/13954"),
("Is.H 806","Thamudic B","l ḍtm h- s¹fr w h- frs¹","By Ḍtm are the inscription and the horse","لِضَتم النقش والحصان","https://ociana.osu.edu/inscriptions/5826"),
("GETham 2","Thamudic B","l (l)hn wdd ns²l ḏ ʿtq","By Ḏ son of . (Llhn) greets Ns²l who was freed","من Ḏ بن . (للهن) يحيّي نس²ل الذي أُعتق","https://ociana.osu.edu/inscriptions/44105")]

def transliterate(s): return "".join(ONA.get(c,c) for c in s)
def scan(s,script="ancient-north-arabian"):
    hits=[]
    for key,(rng,name,direction) in RANGES.items():
        n=sum(rng[0]<=ord(c)<=rng[1] for c in s)
        if n:hits.append({"id":key,"script":name,"direction":direction,"matched_character_count":n,"unicode_range":f"U+{rng[0]:04X}-U+{rng[1]:04X}"})
    return {"language":script,"scripts_detected":hits,"matched_character_count":sum(x["matched_character_count"] for x in hits),"variant_forms":VARIANTS,"evidence_note":"Unicode/script detection is not itself linguistic translation or palaeographic identification."}
def norm(s): return re.sub(r"\s+"," ",s.replace("–","-").replace("—","-").casefold().strip())
def translate(s,script,target):
    tr=transliterate(s); lang="ar" if target.lower().startswith("ar") else "en"
    m=[x for x in CORPUS if norm(x[2])==norm(tr) and x[1].casefold()==script.casefold()] or [x for x in CORPUS if norm(x[2])==norm(tr)]
    e=m[0] if len(m)==1 else None
    return {"source_text":s,"script":script,"transliteration":tr,"target_language":lang,"translation":e[4 if lang=="ar" else 3] if e else None,"translation_status":"corpus_match" if e else "not_available","confidence":"scholarly" if e else "unknown","corpus_id":e[0] if e else None,"provenance":e[5] if e else None}

def worker_ocr(path,langs):
    import easyocr
    rows=easyocr.Reader(langs,gpu=False,verbose=False).readtext(str(path),detail=1,paragraph=False)
    rows.sort(key=lambda x:(min(p[1] for p in x[0]),min(p[0] for p in x[0])))
    return {"text":"\n".join(str(x[1]) for x in rows).strip(),"provider":"easyocr-subprocess","detections":len(rows),"ocr_confidence":round(sum(float(x[2]) for x in rows)/len(rows),6) if rows else 0.0}
def worker_pdf(path,page,out):
    import pypdfium2 as pdfium
    doc=pdfium.PdfDocument(str(path))
    if page<0 or page>=len(doc): raise IndexError(f"PDF page index out of range: {page}")
    doc[page].render(scale=2.0).to_pil().save(out); return {"output":str(out),"provider":"pypdfium2-subprocess","page":page+1}
def run_worker(args,timeout):
    env=os.environ.copy();env.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
    for n in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):env.setdefault(n,"1")
    try:p=subprocess.run([sys.executable,str(Path(__file__).resolve()),*args],capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=timeout,env=env)
    except subprocess.TimeoutExpired as e:raise RuntimeError(f"Native worker timed out after {timeout}s") from e
    lines=(p.stdout or "").strip().splitlines()
    try:r=json.loads(lines[-1]) if lines else None
    except json.JSONDecodeError:r=None
    if not r:raise RuntimeError(f"Native worker exited without valid JSON (exit code {p.returncode})")
    if not r.get("ok"):raise RuntimeError(f"Native worker unavailable: {r.get('error','unknown error')}")
    return r
def ocr(path,langs=None):return run_worker(["--worker-ocr",str(path),"--languages",",".join(langs or ["en","ar"])],max(30,int(os.environ.get("NLP_OCR_TIMEOUT","180"))))
def pdf_page(path,page,out):return run_worker(["--worker-pdf",str(path),"--page",str(page),"--output",str(out)],max(30,int(os.environ.get("NLP_PDF_TIMEOUT","120"))))

def extract(path):
    p=Path(path).expanduser();s=p.suffix.casefold()
    if not p.is_file():raise FileNotFoundError(p)
    if s in {".txt",".md",".csv"}:return p.read_text(encoding="utf-8"),{"provider":"utf8","media_type":"text","source_file":str(p)}
    if s in {".png",".jpg",".jpeg",".webp",".bmp",".tif",".tiff"}:
        try:r=ocr(p);return r.get("text",""),{"provider":"easyocr-subprocess","media_type":"image","source_file":str(p),"ocr_available":True,**{k:v for k,v in r.items() if k not in {"ok","text"}}}
        except Exception as e:return "",{"provider":"easyocr-subprocess","media_type":"image","source_file":str(p),"ocr_available":False,"ocr_error":f"{type(e).__name__}: {e}"}
    if s==".pdf":
        from pypdf import PdfReader
        pages=[x.extract_text() or "" for x in PdfReader(str(p)).pages];text="\n".join(pages).strip()
        if text:return text,{"provider":"pypdf","media_type":"pdf","source_file":str(p),"pages":len(pages),"scanned_pages":False}
        out=[];errors=[]
        with tempfile.TemporaryDirectory(prefix="scanner-pdf-") as tmp:
            for i in range(len(pages)):
                img=Path(tmp)/f"page-{i+1}.png"
                try:pdf_page(p,i,img);r=ocr(img);out.append(r.get("text",""))
                except Exception as e:errors.append(f"page {i+1}: {type(e).__name__}: {e}")
        return "\n".join(x for x in out if x).strip(),{"provider":"pypdfium2+easyocr-subprocess","media_type":"pdf","source_file":str(p),"pages":len(pages),"scanned_pages":True,"ocr_errors":errors,"ocr_available":bool(out)}
    raise ValueError(f"unsupported media type: {s}")

def history_path():return Path.home()/".scanner_history.json"
def save_history(r):
    p=history_path()
    try:rows=json.loads(p.read_text(encoding="utf-8"))
    except Exception:rows=[]
    rows.append(r);p.write_text(json.dumps(rows[-500:],ensure_ascii=False,indent=2),encoding="utf-8")
def export_history_pdf(path,title):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    p=history_path();rows=json.loads(p.read_text(encoding="utf-8")) if p.exists() else [];c=canvas.Canvas(str(path),pagesize=A4);_,h=A4;y=h-40;c.setFont("Helvetica",9);c.drawString(40,y,title);y-=20
    for r in rows:
        line=f"{r.get('corpus_id') or '-'} | {r.get('translation_status')} | {r.get('translation') or 'unavailable'}"
        for part in (line[i:i+105] for i in range(0,len(line),105)):
            if y<40:c.showPage();c.setFont("Helvetica",9);y=h-40
            c.drawString(40,y,part);y-=13
    c.save()

def cli(args,app_name):
    p=argparse.ArgumentParser(description=f"Self-contained {app_name} scanner")
    p.add_argument("file",nargs="?");p.add_argument("--text",default="");p.add_argument("--script",default="Dadanitic");p.add_argument("--target",default="en",choices=["en","ar"])
    p.add_argument("--scan",action="store_true");p.add_argument("--translate",action="store_true");p.add_argument("--gui",action="store_true")
    p.add_argument("--worker-ocr");p.add_argument("--worker-pdf");p.add_argument("--page",type=int,default=0);p.add_argument("--output");p.add_argument("--languages",default="en,ar");a=p.parse_args(args)
    if a.worker_ocr:
        r=worker_ocr(Path(a.worker_ocr),[x.strip() for x in a.languages.split(",") if x.strip()]);print(json.dumps({"ok":True,**r},ensure_ascii=False));return 0
    if a.worker_pdf:
        if not a.output:raise ValueError("--output is required")
        r=worker_pdf(Path(a.worker_pdf),a.page,Path(a.output));print(json.dumps({"ok":True,**r},ensure_ascii=False));return 0
    if a.file and not a.gui:
        text,meta=extract(a.file);r=translate(text,a.script,a.target);r["media"]=meta;print(json.dumps(r,ensure_ascii=False,indent=2) if a.translate else text);return 0
    if a.text and a.scan:print(json.dumps(scan(a.text,a.script),ensure_ascii=False,indent=2));return 0
    if a.text and a.translate:print(json.dumps(translate(a.text,a.script,a.target),ensure_ascii=False,indent=2));return 0
    return gui(a,app_name)
def gui(a,app_name):
    import tkinter as tk
    from tkinter import filedialog,messagebox,ttk
    root=tk.Tk();root.title(f"{app_name} — All-in-One Scanner");root.geometry("1100x800")
    source=tk.Text(root,height=12,wrap="word");source.pack(fill="both",expand=True,padx=10,pady=10);output=tk.Text(root,height=12,wrap="word");output.pack(fill="both",expand=True,padx=10,pady=10);status=ttk.Label(root,text="Ready");status.pack(anchor="w",padx=10)
    def process():
        try:
            path=filedialog.askopenfilename(filetypes=[("Supported","*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.pdf *.txt *.md *.csv"),("All files","*.*")])
            if not path:return
            text,meta=extract(path);source.delete("1.0","end");source.insert("1.0",text);r=translate(text,a.script,a.target);r["media"]=meta;output.delete("1.0","end");output.insert("1.0",json.dumps(r,ensure_ascii=False,indent=2));status.config(text=f"Processed {Path(path).name}")
        except Exception as e:messagebox.showerror("Scanner error",f"{type(e).__name__}: {e}");status.config(text="Import failed safely")
    ttk.Button(root,text="Import and process",command=process).pack(pady=6);root.mainloop();return 0

if __name__=="__main__":
    try:raise SystemExit(cli(sys.argv[1:],"NLP / Ancient Language"))
    except Exception as e:
        print(json.dumps({"ok":False,"error_type":type(e).__name__,"error":str(e)},ensure_ascii=False),file=sys.stderr);raise SystemExit(2)
