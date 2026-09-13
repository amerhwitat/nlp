#!/usr/bin/env python3
"""Thamudic / Ancient North Arabian all-in-one desktop scanner.

This file is intentionally self-contained: runtime protection, Old North Arabian
Unicode registry, OCR/PDF isolation, media import, scanning, transliteration,
evidence-backed translation, PDF history, metadata, voice, CLI and Tkinter GUI
live here so the application can run without importing the local ``thamudic``
package. Native OCR/PDF work is launched by this same file in a child process.
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

# Native-runtime safety is configured before any optional scientific/OCR import.
def configure_runtime() -> None:
    if sys.platform.startswith("win") and os.environ.get("THAMUDIC_ALLOW_DUPLICATE_OPENMP", "1") != "0":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "1")
configure_runtime()

FIRST, LAST = 0x10A80, 0x10A9F
_NAMES = ["HEH","LAM","HAH","MEEM","QAF","WAW","ES-2","REH","BEH","TEH","ES-1","KAF","NOON","KHAH","SAD","ES-3","FEH","ALEF","AIN","DAD","GEEM","DAL","GHAIN","TAH","ZAIN","THAL","YEH","THEH","ZAH"]
_TRANS = ["h","l","ḥ","m","q","w","s2","r","b","t","s1","k","n","ḫ","ṣ","s3","f","ʼ","ʽ","ḍ","g","d","ġ","ṭ","z","ḏ","y","ṯ","ẓ"]
ONA = {chr(cp): tr for cp, tr in zip(range(FIRST, 0x10A9D), _TRANS)}
VARIANTS = ("Dadanitic", "Safaitic", "Hismaic", "Taymanitic", "Minaic", "Thamudic B", "Thamudic C", "Thamudic D")

def transliterate(text: str) -> str:
    return "".join(ONA.get(ch, ch) for ch in text)

def scan(text: str, script: str = "ancient-north-arabian") -> dict[str, Any]:
    chars = [(ch, f"U+{ord(ch):04X}", ONA.get(ch)) for ch in text if ch in ONA]
    return {"language": script, "matched_character_count": len(chars), "characters": chars,
            "unicode_range": f"U+{FIRST:04X}-U+{LAST:04X}", "variant_forms": VARIANTS,
            "direction": "rtl", "evidence_note": "Unicode identifies the Old North Arabian repertoire; variant inscription traditions require palaeographic evidence."}

@dataclass(frozen=True)
class CorpusEntry:
    identifier: str; script: str; transliteration: str; translations: dict[str,str]; source_url: str; confidence: str = "scholarly"
CORPUS = (
 CorpusEntry("TIJ 503","Safaitic","ytm bn ʿbny w wgm ʿl- ḫll -h",{"en":"Ytm son ʿbny and he grieved for his friend","ar":"يتم بن عبني وحزن على صديقه"},"https://ociana.osu.edu/inscriptions/2400"),
 CorpusEntry("AH 311","Dadanitic","bḏkrh wdd ḏ{h}k",{"en":"Bḏkrh loves {Ḏhk}","ar":"بذَكرَه يحب {ذهك}"},"https://ociana.osu.edu/inscriptions/13954"),
 CorpusEntry("Is.H 806","Thamudic B","l ḍtm h- s¹fr w h- frs¹",{"en":"By Ḍtm are the inscription and the horse","ar":"لِضَتم النقش والحصان"},"https://ociana.osu.edu/inscriptions/5826"),
 CorpusEntry("GETham 2","Thamudic B","l (l)hn wdd ns²l ḏ ʿtq",{"en":"By Ḏ son of . (Llhn) greets Ns²l who was freed","ar":"من Ḏ بن . (للهن) يحيّي نس²ل الذي أُعتق"},"https://ociana.osu.edu/inscriptions/44105"),
)
def norm(s: str) -> str: return re.sub(r"\s+", " ", s.replace("–","-").replace("—","-").casefold().strip())
def translate(text: str, script: str, target: str) -> dict[str,Any]:
    tr = transliterate(text); target = "ar" if target.lower().startswith("ar") else "en"
    matches = [e for e in CORPUS if norm(e.transliteration)==norm(tr) and e.script.casefold()==script.casefold()]
    if not matches: matches = [e for e in CORPUS if norm(e.transliteration)==norm(tr)]
    e = matches[0] if len(matches)==1 else None
    return {"source_text":text,"script":script,"transliteration":tr,"target_language":target,
            "translation":e.translations.get(target) if e else None,
            "translation_status":"corpus_match" if e else "not_available",
            "confidence":e.confidence if e else "unknown","corpus_id":e.identifier if e else None,
            "provenance":e.source_url if e else None}

def _worker_env():
    env=os.environ.copy(); env.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
    for n in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): env.setdefault(n,"1")
    return env

def worker_ocr(path: Path, languages: list[str]) -> dict[str,Any]:
    import easyocr
    reader=easyocr.Reader(languages,gpu=False,verbose=False)
    rows=reader.readtext(str(path),detail=1,paragraph=False)
    rows=sorted(rows,key=lambda x:(min(p[1] for p in x[0]),min(p[0] for p in x[0])))
    text="\n".join(str(x[1]) for x in rows).strip(); conf=sum(float(x[2]) for x in rows)/len(rows) if rows else 0.0
    return {"text":text,"provider":"easyocr-subprocess","detections":len(rows),"ocr_confidence":round(conf,6)}

def worker_pdf(path: Path, page: int, output: Path) -> dict[str,Any]:
    import pypdfium2 as pdfium
    doc=pdfium.PdfDocument(str(path))
    if page<0 or page>=len(doc): raise IndexError(f"PDF page index out of range: {page}")
    bitmap=doc[page].render(scale=2.0); bitmap.to_pil().save(output)
    return {"output":str(output),"provider":"pypdfium2-subprocess","page":page+1}

def _run_worker(args: list[str], timeout: int) -> dict[str,Any]:
    cmd=[sys.executable,str(Path(__file__).resolve()),*args]
    try: p=subprocess.run(cmd,capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=timeout,env=_worker_env(),check=False)
    except subprocess.TimeoutExpired as e: raise RuntimeError(f"Native worker timed out after {timeout}s; the GUI remains protected.") from e
    out=(p.stdout or "").strip(); payload=None
    if out:
        try: payload=json.loads(out.splitlines()[-1])
        except json.JSONDecodeError: payload=None
    if not payload: raise RuntimeError("Native worker exited without valid JSON" + (f": {(p.stderr or '')[-1000:]}" if p.stderr else f" (exit code {p.returncode})"))
    if not payload.get("ok"): raise RuntimeError(f"Native worker unavailable: {payload.get('error_type','Error')}: {payload.get('error','unknown error')}")
    return payload

def ocr_image(path: Path, languages=None): return _run_worker(["--worker-ocr",str(path),"--languages",",".join(languages or ["en","ar"])],max(30,int(os.environ.get("THAMUDIC_OCR_TIMEOUT","180"))))
def pdf_page(path: Path,page:int,out:Path): return _run_worker(["--worker-pdf",str(path),"--page",str(page),"--output",str(out)],max(30,int(os.environ.get("THAMUDIC_PDF_RENDER_TIMEOUT","120"))))

def extract_media(path: str|Path, languages=None):
    p=Path(path).expanduser();
    if not p.is_file(): raise FileNotFoundError(p)
    s=p.suffix.casefold()
    if s in {".txt",".md",".csv"}: return p.read_text(encoding="utf-8"),{"provider":"utf8","media_type":"text","source_file":str(p)}
    if s in {".png",".jpg",".jpeg",".webp",".bmp",".tif",".tiff"}:
        try: r=ocr_image(p,languages); return r.get("text", ""),{"provider":"easyocr-subprocess","media_type":"image","source_file":str(p),"ocr_available":True,**{k:v for k,v in r.items() if k not in {"ok","text"}}}
        except Exception as e: return "",{"provider":"easyocr-subprocess","media_type":"image","source_file":str(p),"ocr_available":False,"ocr_error":f"{type(e).__name__}: {e}"}
    if s==".pdf":
        try:
            from pypdf import PdfReader
            reader=PdfReader(str(p)); pages=[x.extract_text() or "" for x in reader.pages]; text="\n".join(pages).strip()
            if text: return text,{"provider":"pypdf","media_type":"pdf","source_file":str(p),"pages":len(pages),"scanned_pages":False}
            count=len(reader.pages)
        except Exception as e: raise RuntimeError(f"Could not read PDF '{p.name}': {type(e).__name__}: {e}") from e
        all_text=[]; errors=[]; conf=[]
        with tempfile.TemporaryDirectory(prefix="thamudic-pdf-") as tmp:
            for i in range(count):
                img=Path(tmp)/f"page-{i+1}.png"
                try:
                    pdf_page(p,i,img); r=ocr_image(img,languages); 
                    if r.get("text"): all_text.append(r["text"])
                    if "ocr_confidence" in r: conf.append(float(r["ocr_confidence"]))
                except Exception as e: errors.append(f"page {i+1}: {type(e).__name__}: {e}")
        return "\n".join(all_text).strip(),{"provider":"pypdfium2+easyocr-subprocess","media_type":"pdf","source_file":str(p),"pages":count,"scanned_pages":True,"ocr_confidence":round(sum(conf)/len(conf),6) if conf else 0.0,"ocr_errors":errors,"ocr_available":bool(all_text) or not errors}
    raise ValueError(f"unsupported media type: {s}")

def script_metadata(script="ancient-north-arabian"):
    return {"id":script,"name":"Ancient North Arabian","scripts":["Old North Arabian"],"variations":list(VARIANTS),"direction":"rtl","unicode_block":"Old North Arabian U+10A80-U+10A9F","principle":"Script recognition, transliteration, translation and historical interpretation are separate evidence layers."}

def records_path(): return Path.home()/".thamudic_scanner_history.json"
def save_record(result):
    p=records_path(); rows=[]
    try: rows=json.loads(p.read_text(encoding="utf-8"))
    except Exception: pass
    rows.append(result); p.write_text(json.dumps(rows[-500:],ensure_ascii=False,indent=2),encoding="utf-8")

def export_history_pdf(output):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    rows=json.loads(records_path().read_text(encoding="utf-8")) if records_path().exists() else []
    c=canvas.Canvas(str(output),pagesize=A4); w,h=A4; y=h-40; c.setFont("Helvetica",9); c.drawString(40,y,"Thamudic Scanner Translation History"); y-=20
    for row in rows:
        line=f"{row.get('corpus_id') or '-'} | {row.get('translation_status')} | {row.get('translation') or 'unavailable'}"
        for part in [line[i:i+105] for i in range(0,len(line),105)]:
            if y<40: c.showPage(); c.setFont("Helvetica",9); y=h-40
            c.drawString(40,y,part); y-=13
    c.save()

class Voice:
    def __init__(self):
        self.engine=None
        try:
            import pyttsx3; self.engine=pyttsx3.init()
        except Exception: pass
    def available(self): return self.engine is not None
    def speak(self,text,rate=160):
        if not self.engine: raise RuntimeError("pyttsx3 is not installed")
        self.engine.setProperty("rate",rate); self.engine.say(text); self.engine.runAndWait()
    def stop(self):
        if self.engine:
            try:self.engine.stop()
            except Exception:pass

def jdump(x): return json.dumps(x,ensure_ascii=False,indent=2,default=str)
def _set(widget,text): widget.delete("1.0","end"); widget.insert("1.0",text)

class App:
    def __init__(self):
        import tkinter as tk
        from tkinter import ttk
        self.tk=tk; self.ttk=ttk; self.root=tk.Tk(); self.root.title("Thamudic / Ancient North Arabian — All-in-One Scanner"); self.root.geometry("1180x900"); self.root.minsize(960,720); self.path=None; self.voice=Voice(); self.last=None; self.build()
    def build(self):
        tk,ttk=self.tk,self.ttk; outer=ttk.Frame(self.root,padding=16); outer.pack(fill="both",expand=True)
        ttk.Label(outer,text="Thamudic / Ancient North Arabian",font=("TkDefaultFont",18,"bold")).pack(anchor="w")
        ttk.Label(outer,text="One executable: import image/PDF/text, OCR safely, scan, transliterate, translate with evidence, export and use voice.").pack(anchor="w",pady=(2,10))
        media=ttk.LabelFrame(outer,text="Import / media"); media.pack(fill="x",pady=(0,8))
        ttk.Button(media,text="Import image / PDF",command=self.import_media).pack(side="left",padx=5,pady=6); ttk.Button(media,text="Import text",command=self.import_text).pack(side="left",padx=5,pady=6); ttk.Button(media,text="Process",command=self.process).pack(side="left",padx=5,pady=6)
        self.media_status=ttk.Label(media,text="Media: ready"); self.media_status.pack(side="left",padx=12)
        controls=ttk.Frame(outer); controls.pack(fill="x",pady=(0,8)); ttk.Label(controls,text="Variant").pack(side="left"); self.script=ttk.Combobox(controls,values=VARIANTS,state="readonly",width=18); self.script.set("Dadanitic"); self.script.pack(side="left",padx=5); ttk.Label(controls,text="Target").pack(side="left",padx=(10,4)); self.target=ttk.Combobox(controls,values=("en","ar"),state="readonly",width=8); self.target.set("en"); self.target.pack(side="left"); ttk.Button(controls,text="Scan",command=self.scan_action).pack(side="left",padx=5); ttk.Button(controls,text="Translate",command=self.translate_action).pack(side="left",padx=5); ttk.Button(controls,text="Metadata",command=self.metadata).pack(side="left",padx=5); ttk.Button(controls,text="History PDF",command=self.history_pdf).pack(side="left",padx=5)
        sf=ttk.LabelFrame(outer,text="Original / extracted OCR text"); sf.pack(fill="x",pady=(0,8)); self.source=tk.Text(sf,height=8,wrap="word"); self.source.pack(fill="x",padx=6,pady=6)
        panes=ttk.Panedwindow(outer,orient="vertical"); panes.pack(fill="both",expand=True); f1=ttk.LabelFrame(panes,text="Scan / transliteration"); f2=ttk.LabelFrame(panes,text="Translation / evidence / media diagnostics"); panes.add(f1,weight=1); panes.add(f2,weight=2); self.trans=tk.Text(f1,wrap="word"); self.trans.pack(fill="both",expand=True,padx=6,pady=6); self.out=tk.Text(f2,wrap="word"); self.out.pack(fill="both",expand=True,padx=6,pady=6)
        v=ttk.LabelFrame(outer,text="Voice"); v.pack(fill="x",pady=(8,4)); ttk.Button(v,text="▶ Read",command=self.speak).pack(side="left",padx=4,pady=7); ttk.Button(v,text="■ Stop",command=self.stop).pack(side="left",padx=4,pady=7); ttk.Label(v,text="Rate").pack(side="left",padx=(18,4)); self.rate=tk.IntVar(value=160); ttk.Scale(v,from_=60,to=300,variable=self.rate,orient="horizontal",length=190).pack(side="left"); self.status=ttk.Label(outer,text="Ready"); self.status.pack(anchor="w",pady=(5,0))
    def import_media(self):
        from tkinter import filedialog,messagebox
        p=filedialog.askopenfilename(title="Import image / PDF / text",filetypes=[("Images/PDF/text","*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.pdf *.txt *.md *.csv"),("All files","*.*")]);
        if p:self.path=Path(p); self.process()
    def import_text(self): self.import_media()
    def process(self):
        from tkinter import messagebox
        try:
            if not self.path: self.import_media(); return
            text,meta=extract_media(self.path); _set(self.source,text); self.media_status.config(text=f"Media: {self.path.name} · {meta.get('provider','unknown')}"); self.out.insert("end","\n\nMEDIA:\n"+jdump(meta)); self.translate_action()
        except Exception as e: messagebox.showerror("Media import/OCR error",f"{type(e).__name__}: {e}"); self.status.config(text="Import failed safely; application remains running")
    def scan_action(self):
        text=self.source.get("1.0","end-1c").strip()
        if not text:return
        try:r=scan(text,self.script.get()); _set(self.trans,jdump(r)); self.status.config(text=f"Matched characters: {r['matched_character_count']}")
        except Exception as e:self.status.config(text=f"Scan error: {e}")
    def translate_action(self):
        text=self.source.get("1.0","end-1c").strip()
        if not text:return
        try:r=translate(text,self.script.get(),self.target.get()); self.last=r; _set(self.trans,r["transliteration"]); _set(self.out,jdump(r)); save_record(r); self.status.config(text=f"{r['translation_status']} · confidence={r['confidence']}")
        except Exception as e:self.status.config(text=f"Translation error: {e}")
    def metadata(self): _set(self.out,jdump(script_metadata(self.script.get()))); self.status.config(text="Metadata loaded")
    def history_pdf(self):
        from tkinter import filedialog,messagebox
        p=filedialog.asksaveasfilename(defaultextension=".pdf",filetypes=[("PDF","*.pdf")],initialfile="thamudic-history.pdf")
        if p:
            try:export_history_pdf(p); self.status.config(text=f"PDF written: {p}")
            except Exception as e:messagebox.showerror("PDF export",str(e))
    def speak(self):
        text=self.trans.get("1.0","end-1c").strip()
        try:self.voice.speak(text,int(self.rate.get())); self.status.config(text="Voice playback complete")
        except Exception as e:self.status.config(text=f"Voice error: {e}")
    def stop(self):self.voice.stop(); self.status.config(text="Voice stopped")
    def run(self):self.root.mainloop()

def main(argv=None):
    p=argparse.ArgumentParser(description="Self-contained Thamudic scanner"); p.add_argument("file",nargs="?"); p.add_argument("--text",default=""); p.add_argument("--script",default="Dadanitic"); p.add_argument("--target",default="en",choices=["en","ar"]); p.add_argument("--scan",action="store_true"); p.add_argument("--translate",action="store_true"); p.add_argument("--gui",action="store_true"); p.add_argument("--worker-ocr"); p.add_argument("--worker-pdf"); p.add_argument("--page",type=int,default=0); p.add_argument("--output"); p.add_argument("--languages",default="en,ar"); a=p.parse_args(argv)
    try:
        if a.worker_ocr:
            print(json.dumps({"ok":True,**worker_ocr(Path(a.worker_ocr),[x for x in a.languages.split(",") if x]}),ensure_ascii=False)); return 0
        if a.worker_pdf:
            if not a.output: raise ValueError("--output is required")
            print(json.dumps({"ok":True,**worker_pdf(Path(a.worker_pdf),a.page,Path(a.output))},ensure_ascii=False)); return 0
        if a.file and not a.gui:
            text,meta=extract_media(a.file); r=translate(text,a.script,a.target); r["media"]=meta; print(jdump(r) if a.translate else text); return 0
        if a.text and a.scan: print(jdump(scan(a.text,a.script))); return 0
        if a.text and a.translate: print(jdump(translate(a.text,a.script,a.target))); return 0
        app=App(); app.script.set(a.script); app.target.set(a.target)
        if a.file: app.path=Path(a.file); app.process()
        elif a.text: _set(app.source,a.text); app.translate_action()
        app.run(); return 0
    except Exception as e:
        print(json.dumps({"ok":False,"error_type":type(e).__name__,"error":str(e)},ensure_ascii=False),file=sys.stderr); return 2
if __name__=="__main__": raise SystemExit(main())
