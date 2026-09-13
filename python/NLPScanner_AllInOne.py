#!/usr/bin/env python3
"""General NLP / ancient-language all-in-one scanner.

Everything required by the desktop application is contained in this file:
runtime safeguards, Unicode/script detection, OCR/PDF isolation, media import,
Thamudic transliteration and evidence-backed examples, voice, PDF export, CLI
and GUI. The same file acts as the native OCR/PDF child worker, so no local
package import is required for the application entry point.
"""
from __future__ import annotations
import argparse,json,os,re,subprocess,sys,tempfile,unicodedata
from pathlib import Path
from dataclasses import dataclass
from typing import Any

def configure_runtime():
    if sys.platform.startswith("win") and os.environ.get("NLP_ALLOW_DUPLICATE_OPENMP","1")!="0": os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
    for n in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(n,"1")
configure_runtime()

# Unicode/script detection is deliberately evidence-based: detection does not imply language identification.
SCRIPT_RANGES={
 "ancient-north-arabian":((0x10A80,0x10A9F),"Old North Arabian","rtl"),
 "ancient-egyptian":((0x13000,0x1342F),"Egyptian Hieroglyphs","ltr"),
 "ugaritic":((0x10380,0x1039F),"Ugaritic","ltr"),
 "phoenician":((0x10900,0x1091F),"Phoenician","rtl"),
 "old-south-arabian":((0x10A60,0x10A7F),"Old South Arabian","rtl"),
 "old-persian":((0x103A0,0x103DF),"Old Persian","ltr"),
 "coptic":((0x2C80,0x2CFF),"Coptic","ltr"),
 "gothic":((0x10330,0x1034F),"Gothic","ltr"),
 "old-turkic":((0x10C00,0x10C4F),"Old Turkic","rtl-or-ltr"),
 "linear-b":((0x10000,0x1007F),"Linear B","ltr"),
}
ONA_FIRST=0x10A80; ONA_LAST=0x10A9F
ONA_TRANS=dict(zip((chr(x) for x in range(ONA_FIRST,0x10A9D)),["h","l","ḥ","m","q","w","s2","r","b","t","s1","k","n","ḫ","ṣ","s3","f","ʼ","ʽ","ḍ","g","d","ġ","ṭ","z","ḏ","y","ṯ","ẓ"]))
THAMUDIC_VARIANTS=("Dadanitic","Safaitic","Hismaic","Taymanitic","Minaic","Thamudic B","Thamudic C","Thamudic D")

def scan_text(text,requested=None):
    hits=[]
    for name,(rng,script,direction) in SCRIPT_RANGES.items():
        count=sum(1 for ch in text if rng[0]<=ord(ch)<=rng[1])
        if count:hits.append({"id":name,"script":script,"direction":direction,"matched_character_count":count,"unicode_range":f"U+{rng[0]:04X}-U+{rng[1]:04X}"})
    if requested and requested not in {x["id"] for x in hits}: hits=[]
    return {"requested_script":requested,"scripts_detected":hits,"matched_character_count":sum(x["matched_character_count"] for x in hits),"note":"Unicode/script detection is not itself a linguistic translation or palaeographic identification."}

def transliterate_ona(text):return "".join(ONA_TRANS.get(ch,ch) for ch in text)
@dataclass(frozen=True)
class Entry:
    id:str; script:str; transliteration:str; en:str; ar:str; source_url:str
CORPUS=(
 Entry("TIJ 503","Safaitic","ytm bn ʿbny w wgm ʿl- ḫll -h","Ytm son ʿbny and he grieved for his friend","يتم بن عبني وحزن على صديقه","https://ociana.osu.edu/inscriptions/2400"),
 Entry("AH 311","Dadanitic","bḏkrh wdd ḏ{h}k","Bḏkrh loves {Ḏhk}","بذَكرَه يحب {ذهك}","https://ociana.osu.edu/inscriptions/13954"),
 Entry("Is.H 806","Thamudic B","l ḍtm h- s¹fr w h- frs¹","By Ḍtm are the inscription and the horse","لِضَتم النقش والحصان","https://ociana.osu.edu/inscriptions/5826"),
 Entry("GETham 2","Thamudic B","l (l)hn wdd ns²l ḏ ʿtq","By Ḏ son of . (Llhn) greets Ns²l who was freed","من Ḏ بن . (للهن) يحيّي نس²ل الذي أُعتق","https://ociana.osu.edu/inscriptions/44105"),
)
def norm(s):return re.sub(r"\s+"," ",s.replace("–","-").replace("—","-").casefold().strip())
def translate_ancient(text,script,target):
    tr=transliterate_ona(text); matches=[e for e in CORPUS if norm(e.transliteration)==norm(tr) and e.script.casefold()==script.casefold()]
    if not matches:matches=[e for e in CORPUS if norm(e.transliteration)==norm(tr)]
    e=matches[0] if len(matches)==1 else None; lang="ar" if target.lower().startswith("ar") else "en"
    return {"source_text":text,"script":script,"transliteration":tr,"target_language":lang,"translation":(e.ar if lang=="ar" else e.en) if e else None,"translation_status":"corpus_match" if e else "not_available","confidence":"scholarly" if e else "unknown","corpus_id":e.id if e else None,"provenance":e.source_url if e else None}

def worker_ocr(path,languages):
    import easyocr
    reader=easyocr.Reader(languages,gpu=False,verbose=False); rows=reader.readtext(str(path),detail=1,paragraph=False); rows=sorted(rows,key=lambda x:(min(p[1] for p in x[0]),min(p[0] for p in x[0])))
    return {"text":"\n".join(str(x[1]) for x in rows).strip(),"provider":"easyocr-subprocess","detections":len(rows),"ocr_confidence":round(sum(float(x[2]) for x in rows)/len(rows),6) if rows else 0.0}
def worker_pdf(path,page,out):
    import pypdfium2 as pdfium
    doc=pdfium.PdfDocument(str(path));
    if page<0 or page>=len(doc):raise IndexError(f"PDF page index out of range: {page}")
    doc[page].render(scale=2.0).to_pil().save(out); return {"output":str(out),"provider":"pypdfium2-subprocess","page":page+1}
def run_worker(args,timeout):
    cmd=[sys.executable,str(Path(__file__).resolve()),*args]
    try:p=subprocess.run(cmd,capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=timeout,env=os.environ.copy(),check=False)
    except subprocess.TimeoutExpired as e:raise RuntimeError(f"Native worker timed out after {timeout}s; the GUI remains protected.") from e
    out=(p.stdout or "").strip()
    try:r=json.loads(out.splitlines()[-1]) if out else None
    except Exception:r=None
    if not r:raise RuntimeError("Native worker exited without valid JSON"+(f": {(p.stderr or '')[-1000:]}" if p.stderr else f" (exit code {p.returncode})"))
    if not r.get("ok"):raise RuntimeError(f"Native worker unavailable: {r.get('error_type','Error')}: {r.get('error','unknown error')}")
    return r
def ocr(path,languages=None):return run_worker(["--worker-ocr",str(path),"--languages",",".join(languages or ["en","ar"])],max(30,int(os.environ.get("NLP_OCR_TIMEOUT","180"))))
def render_pdf(path,page,out):return run_worker(["--worker-pdf",str(path),"--page",str(page),"--output",str(out)],max(30,int(os.environ.get("NLP_PDF_RENDER_TIMEOUT","120"))))

def extract(path):
    p=Path(path).expanduser();
    if not p.is_file():raise FileNotFoundError(p)
    s=p.suffix.casefold()
    if s in {".txt",".md",".csv"}:return p.read_text(encoding="utf-8"),{"provider":"utf8","media_type":"text","source_file":str(p)}
    if s in {".png",".jpg",".jpeg",".webp",".bmp",".tif",".tiff"}:
        try:r=ocr(p);return r.get("text",""),{"provider":"easyocr-subprocess","media_type":"image","source_file":str(p),"ocr_available":True,"ocr_confidence":r.get("ocr_confidence",0.0),"detections":r.get("detections",0)}
        except Exception as e:return "",{"provider":"easyocr-subprocess","media_type":"image","source_file":str(p),"ocr_available":False,"ocr_error":f"{type(e).__name__}: {e}"}
    if s==".pdf":
        from pypdf import PdfReader
        reader=PdfReader(str(p)); pages=[x.extract_text() or "" for x in reader.pages]; text="\n".join(pages).strip()
        if text:return text,{"provider":"pypdf","media_type":"pdf","source_file":str(p),"pages":len(pages),"scanned_pages":False}
        all_text=[];errors=[]
        with tempfile.TemporaryDirectory(prefix="nlp-pdf-") as tmp:
            for i in range(len(pages)):
                img=Path(tmp)/f"page-{i+1}.png"
                try:render_pdf(p,i,img);r=ocr(img);all_text.append(r.get("text",""))
                except Exception as e:errors.append(f"page {i+1}: {type(e).__name__}: {e}")
        return "\n".join(x for x in all_text if x).strip(),{"provider":"pypdfium2+easyocr-subprocess","media_type":"pdf","source_file":str(p),"pages":len(pages),"scanned_pages":True,"ocr_errors":errors,"ocr_available":bool(all_text) or not errors}
    raise ValueError(f"unsupported media type: {s}")

def history_path():return Path.home()/".nlp_scanner_history.json"
def save(r):
    p=history_path(); rows=[]
    try:rows=json.loads(p.read_text(encoding="utf-8"))
    except Exception:pass
    rows.append(r);p.write_text(json.dumps(rows[-500:],ensure_ascii=False,indent=2),encoding="utf-8")
def export_pdf(path):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    rows=json.loads(history_path().read_text(encoding="utf-8")) if history_path().exists() else [];c=canvas.Canvas(str(path),pagesize=A4);w,h=A4;y=h-40;c.setFont("Helvetica",9);c.drawString(40,y,"NLP Scanner History");y-=20
    for r in rows:
        line=f"{r.get('corpus_id') or '-'} | {r.get('translation_status')} | {r.get('translation') or 'unavailable'}"
        for x in [line[i:i+105] for i in range(0,len(line),105)]:
            if y<40:c.showPage();c.setFont("Helvetica",9);y=h-40
            c.drawString(40,y,x);y-=13
    c.save()
class Voice:
    def __init__(self):
        self.engine=None
        try:import pyttsx3;self.engine=pyttsx3.init()
        except Exception:pass
    def speak(self,text,rate=160):
        if not self.engine:raise RuntimeError("pyttsx3 is not installed")
        self.engine.setProperty("rate",rate);self.engine.say(text);self.engine.runAndWait()
    def stop(self):
        if self.engine:
            try:self.engine.stop()
            except Exception:pass

def jd(x):return json.dumps(x,ensure_ascii=False,indent=2,default=str)
def put(w,x):w.delete("1.0","end");w.insert("1.0",x)
class App:
    def __init__(self):
        import tkinter as tk
        from tkinter import ttk
        self.tk=tk;self.ttk=ttk;self.root=tk.Tk();self.root.title("NLP / Thamudic — All-in-One Scanner");self.root.geometry("1180x900");self.root.minsize(960,720);self.path=None;self.voice=Voice();self.build()
    def build(self):
        tk,ttk=self.tk,self.ttk;o=ttk.Frame(self.root,padding=16);o.pack(fill="both",expand=True);ttk.Label(o,text="NLP / Ancient Language Scanner",font=("TkDefaultFont",18,"bold")).pack(anchor="w");ttk.Label(o,text="One executable: import media, safely OCR, detect scripts, transliterate, translate with evidence, export and use voice.").pack(anchor="w",pady=(2,10));m=ttk.LabelFrame(o,text="Import / media");m.pack(fill="x",pady=(0,8));ttk.Button(m,text="Import image / PDF",command=self.import_file).pack(side="left",padx=5,pady=6);ttk.Button(m,text="Import text",command=self.import_file).pack(side="left",padx=5,pady=6);ttk.Button(m,text="Process",command=self.process).pack(side="left",padx=5,pady=6);self.media=ttk.Label(m,text="Media: ready");self.media.pack(side="left",padx=12)
        c=ttk.Frame(o);c.pack(fill="x",pady=(0,8));ttk.Label(c,text="Ancient variant").pack(side="left");self.script=ttk.Combobox(c,values=THAMUDIC_VARIANTS,state="readonly",width=18);self.script.set("Dadanitic");self.script.pack(side="left",padx=5);ttk.Label(c,text="Target").pack(side="left",padx=(10,4));self.target=ttk.Combobox(c,values=("en","ar"),state="readonly",width=8);self.target.set("en");self.target.pack(side="left");ttk.Button(c,text="Scan",command=self.scan).pack(side="left",padx=5);ttk.Button(c,text="Scan + translate",command=self.process).pack(side="left",padx=5);ttk.Button(c,text="Export history PDF",command=self.hist).pack(side="left",padx=5)
        s=ttk.LabelFrame(o,text="Source / OCR text");s.pack(fill="x",pady=(0,8));self.source=tk.Text(s,height=8,wrap="word");self.source.pack(fill="x",padx=6,pady=6);pan=ttk.Panedwindow(o,orient="vertical");pan.pack(fill="both",expand=True);f1=ttk.LabelFrame(pan,text="Script detection / transliteration");f2=ttk.LabelFrame(pan,text="Translation / evidence / diagnostics");pan.add(f1,weight=1);pan.add(f2,weight=2);self.trans=tk.Text(f1,wrap="word");self.trans.pack(fill="both",expand=True,padx=6,pady=6);self.out=tk.Text(f2,wrap="word");self.out.pack(fill="both",expand=True,padx=6,pady=6);v=ttk.LabelFrame(o,text="Voice");v.pack(fill="x",pady=(8,4));ttk.Button(v,text="▶ Read",command=self.speak).pack(side="left",padx=4,pady=7);ttk.Button(v,text="■ Stop",command=self.stop).pack(side="left",padx=4,pady=7);ttk.Label(v,text="Rate").pack(side="left",padx=(18,4));self.rate=tk.IntVar(value=160);ttk.Scale(v,from_=60,to=300,variable=self.rate,orient="horizontal",length=190).pack(side="left");self.status=ttk.Label(o,text="Ready");self.status.pack(anchor="w",pady=(5,0))
    def import_file(self):
        from tkinter import filedialog
        p=filedialog.askopenfilename(title="Import image / PDF / text",filetypes=[("Images/PDF/text","*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.pdf *.txt *.md *.csv"),("All files","*.*")]);
        if p:self.path=Path(p);self.process()
    def process(self):
        from tkinter import messagebox
        try:
            if not self.path:self.import_file();return
            text,meta=extract(self.path);put(self.source,text);self.media.config(text=f"Media: {self.path.name} · {meta.get('provider','unknown')}");self.scan();r=translate_ancient(text,self.script.get(),self.target.get());put(self.out,jd({"translation":r,"media":meta}));save(r);self.status.config(text=f"{r['translation_status']} · media handled safely")
        except Exception as e:messagebox.showerror("Import/processing error",f"{type(e).__name__}: {e}");self.status.config(text="Processing failed safely; application remains running")
    def scan(self):
        text=self.source.get("1.0","end-1c").strip();r=scan_text(text);put(self.trans,jd(r));self.status.config(text=f"Detected scripts: {len(r['scripts_detected'])}")
    def hist(self):
        from tkinter import filedialog,messagebox
        p=filedialog.asksaveasfilename(defaultextension=".pdf",filetypes=[("PDF","*.pdf")],initialfile="nlp-history.pdf")
        if p:
            try:export_pdf(p);self.status.config(text=f"PDF written: {p}")
            except Exception as e:messagebox.showerror("PDF export",str(e))
    def speak(self):
        try:self.voice.speak(self.trans.get("1.0","end-1c"),int(self.rate.get()));self.status.config(text="Voice playback complete")
        except Exception as e:self.status.config(text=f"Voice error: {e}")
    def stop(self):self.voice.stop();self.status.config(text="Voice stopped")

def main(argv=None):
    p=argparse.ArgumentParser(description="Self-contained NLP scanner");p.add_argument("file",nargs="?");p.add_argument("--text",default="");p.add_argument("--script",default="Dadanitic");p.add_argument("--target",default="en",choices=["en","ar"]);p.add_argument("--scan",action="store_true");p.add_argument("--translate",action="store_true");p.add_argument("--worker-ocr");p.add_argument("--worker-pdf");p.add_argument("--page",type=int,default=0);p.add_argument("--output");p.add_argument("--languages",default="en,ar");a=p.parse_args(argv)
    try:
        if a.worker_ocr:print(json.dumps({"ok":True,**worker_ocr(Path(a.worker_ocr),[x for x in a.languages.split(",") if x]}),ensure_ascii=False));return 0
        if a.worker_pdf:
            if not a.output:raise ValueError("--output is required")
            print(json.dumps({"ok":True,**worker_pdf(Path(a.worker_pdf),a.page,Path(a.output))},ensure_ascii=False));return 0
        if a.file and not a.scan and not a.translate and not a.text:
            text,meta=extract(a.file);print(text);return 0
        if a.file:
            text,meta=extract(a.file);r=translate_ancient(text,a.script,a.target);r["media"]=meta;print(jd(r));return 0
        if a.text and a.scan:print(jd(scan_text(a.text,a.script)));return 0
        if a.text and a.translate:print(jd(translate_ancient(a.text,a.script,a.target)));return 0
        app=App();app.script.set(a.script);app.target.set(a.target);app.mainloop();return 0
    except Exception as e:print(json.dumps({"ok":False,"error_type":type(e).__name__,"error":str(e)},ensure_ascii=False),file=sys.stderr);return 2
if __name__=="__main__":raise SystemExit(main())
