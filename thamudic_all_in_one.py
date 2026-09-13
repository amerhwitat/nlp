#!/usr/bin/env python3
"""Single-file Thamudic + NLP research workbench.

The same application provides the Thamudic scanner, NLP analysis, translator,
research/catalog pages, exports, printing, and a responsive local scan worker.
No Tesseract or camel_tools is required. Image segmentation is kept separate
from linguistic reading so the application never fabricates an inscription.
"""
from __future__ import annotations

import csv
import json
import os
import queue
import re
import sqlite3
import tempfile
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageFilter, ImageOps, ImageTk

ONA = {
    0x10A80:"h", 0x10A81:"l", 0x10A82:"ḥ", 0x10A83:"m", 0x10A84:"q", 0x10A85:"w",
    0x10A86:"s²", 0x10A87:"r", 0x10A88:"b", 0x10A89:"t", 0x10A8A:"s¹", 0x10A8B:"k",
    0x10A8C:"n", 0x10A8D:"ḫ", 0x10A8E:"ṣ", 0x10A8F:"s³", 0x10A90:"f", 0x10A91:"ʾ",
    0x10A92:"ʿ", 0x10A93:"ḍ", 0x10A94:"g", 0x10A95:"d", 0x10A96:"ġ", 0x10A97:"ṭ",
    0x10A98:"z", 0x10A99:"ḏ", 0x10A9A:"y", 0x10A9B:"ṯ", 0x10A9C:"ẓ",
}
LEXICON = {
    "mlk": ("ملك", "king"), "bn": ("بن", "son"), "bt": ("بنت", "daughter"),
    "ʾl": ("إل / الله", "El / God"), "ʿbd": ("عبد", "servant"), "byt": ("بيت", "house"),
    "ʾrs": ("أرض", "land"), "ywm": ("يوم", "day"), "šnt": ("سنة", "year"),
    "snt": ("سنة", "year"), "rb": ("رب", "lord"), "bny": ("بنى", "built"),
    "ḥrb": ("حرب", "war"), "ʿyn": ("عين", "spring/eye"),
}
SCRIPTS = ["Old North Arabian / Thamudic", "Safaitic", "Hismaic", "Dadanitic", "Early Arabic"]
PERIODS = ["Paleolithic", "Epi-Paleolithic", "Neolithic", "Chalcolithic", "Bronze Age", "Iron Age", "Hellenistic", "Roman", "Byzantine"]


def transliterate_ona(text):
    return "".join(ONA.get(ord(ch), " " if ch.isspace() else ch) for ch in text)


def _normalize_token(token):
    return token.lower().replace("š", "s").replace("ṯ", "th").replace("ḏ", "dh").replace("²", "2").replace("¹", "1").replace("³", "3")


def translate_transliteration(text):
    raw = (text or "").strip()
    if not raw:
        return "", ""
    key = _normalize_token(raw.replace(" ", ""))
    if key in LEXICON:
        return LEXICON[key]
    ar, en = [], []
    for token in re.split("[,;| ]+", raw):
        if not token:
            continue
        hit = LEXICON.get(_normalize_token(token))
        if hit:
            ar.append(hit[0]); en.append(hit[1])
        else:
            ar.append("[غير معروف]"); en.append("[unresolved: " + token + "]")
    return " ".join(ar), " ".join(en)


def build_text_outputs(source_text):
    source_text = source_text or ""
    is_ona = any(0x10A80 <= ord(c) <= 0x10A9F for c in source_text)
    tr = transliterate_ona(source_text) if is_ona else source_text.strip()
    ar, en = translate_transliteration(tr)
    return {"source_text": source_text, "transliteration": tr, "translation_ar": ar, "translation_en": en}


def normalize(image, scale=2):
    image = ImageOps.exif_transpose(image).convert("L")
    image = ImageOps.autocontrast(image).filter(ImageFilter.MedianFilter(3))
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), Image.Resampling.LANCZOS)
    return image


def segment(image, threshold=150, min_area=25):
    import numpy as np
    arr = np.asarray(image, dtype=np.uint8)
    mask = arr < threshold
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    boxes = []
    for y0 in range(h):
        for x0 in range(w):
            if not mask[y0, x0] or seen[y0, x0]:
                continue
            stack = [(y0, x0)]; seen[y0, x0] = True; xs = []; ys = []
            while stack:
                y, x = stack.pop(); xs.append(x); ys.append(y)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if not dx and not dy: continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True; stack.append((ny, nx))
            if len(xs) >= min_area:
                boxes.append((min(xs), min(ys), max(xs) + 1, max(ys) + 1, len(xs)))
    return sorted(boxes, key=lambda b: (b[1], b[0]))


def scan_image(path, threshold=150, min_area=25, scale=2, text_hint=""):
    path = Path(path)
    with Image.open(path) as opened:
        image = normalize(opened, scale)
    boxes = segment(image, threshold, min_area)
    glyphs = []
    for i, (x1, y1, x2, y2, area) in enumerate(boxes):
        width, height = x2 - x1, y2 - y1
        fill = area / max(width * height, 1)
        confidence = min(1.0, max(0.0, 0.5 * fill + 0.5 * min(width, height) / max(width, height)))
        glyphs.append({"index": i, "x": x1, "y": y1, "width": width, "height": height, "area": area,
                       "aspect_ratio": round(width / max(height, 1), 4), "confidence": round(confidence, 4)})
    out = build_text_outputs(text_hint.strip()) if text_hint.strip() else {"source_text":"", "transliteration":"", "translation_ar":"", "translation_en":""}
    out.update({"schema":"thamudic-scanner/v3", "source_image":str(path),
                "script_family":"Ancient North Arabian / Old North Arabian",
                "recognition_status":"text_hint_translated" if text_hint.strip() else "segmentation_only",
                "glyphs":glyphs,
                "notes":["Thamudic is a scholarly umbrella for multiple Ancient North Arabian varieties.",
                         "Segmentation boxes are visual candidate components, not linguistic readings.",
                         "Automatic image-to-text reading requires a validated glyph recognition model.",
                         "Candidate translations require scholarly verification."]})
    return out


def scan_image_safe(path, **kwargs):
    try:
        return {"ok": True, "result": scan_image(Path(path), **kwargs)}
    except Exception as exc:
        return {"ok": False, "error_type": type(exc).__name__, "error": str(exc),
                "traceback": traceback.format_exc(limit=8)}


def nlp_analyze(text):
    result = build_text_outputs(text)
    tokens = [t for t in re.split("[,;| ]+", result["transliteration"].strip()) if t]
    known = [t for t in tokens if _normalize_token(t) in LEXICON]
    unknown = [t for t in tokens if _normalize_token(t) not in LEXICON]
    result.update({"tokens":tokens, "known_tokens":known, "unknown_tokens":unknown,
                   "token_count":len(tokens), "confidence":"candidate / scholarly review required"})
    return result


class LocalCatalog:
    def __init__(self, path="ancient_objects.sqlite"):
        self.path = path; self.lock = threading.RLock()
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS records (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, title TEXT, script TEXT, period TEXT, object_type TEXT, source_image TEXT, transliteration TEXT, translation_ar TEXT, translation_en TEXT, provenance TEXT, evidence_json TEXT)")
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10); db.row_factory = sqlite3.Row; return db
    def add(self, r):
        with self.lock, self.connect() as db:
            cur = db.execute("INSERT INTO records (created_at,title,script,period,object_type,source_image,transliteration,translation_ar,translation_en,provenance,evidence_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (datetime.now().isoformat(timespec="seconds"), r.get("title",""), r.get("script",""), r.get("period",""), r.get("object_type",""), r.get("source_image",""), r.get("transliteration",""), r.get("translation_ar",""), r.get("translation_en",""), r.get("provenance",""), json.dumps(r.get("evidence",{}), ensure_ascii=False)))
            return cur.lastrowid
    def rows(self, query=""):
        with self.lock, self.connect() as db:
            q = "%" + query + "%"
            rows = db.execute("SELECT * FROM records WHERE ?='' OR title LIKE ? OR transliteration LIKE ? OR translation_ar LIKE ? OR translation_en LIKE ? ORDER BY id DESC", (query,q,q,q,q)).fetchall()
            return [dict(x) for x in rows]
    def stats(self):
        with self.lock, self.connect() as db: n = db.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        return {"objects":n, "annotations":n, "sources":n, "schema_version":1}


class App(tk.Tk):
    def __init__(self, mode="thamudic", db_path="ancient_objects.sqlite"):
        super().__init__(); self.mode = mode
        self.title("NLP Thamudic Scanner — Ancient Languages Research Workbench" if mode == "nlp" else "Thamudic Scanner — Ancient Languages Research Workbench")
        self.geometry("1500x920"); self.minsize(1120,720); self.configure(background="#10151c")
        self.db = LocalCatalog(db_path); self.current_image = None; self.preview = None; self.current_result = None
        self.status_var = tk.StringVar(value="Ready"); self.section_var = tk.StringVar(value="Dashboard")
        self.script_var = tk.StringVar(value=SCRIPTS[0]); self.period_var = tk.StringVar(value=PERIODS[5]); self.object_type = tk.StringVar(value="inscription"); self.search_var = tk.StringVar()
        self.worker_queue = queue.Queue(); self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="thamudic-local-worker")
        self.scan_generation = 0; self.scan_running = False
        self._style(); self._shell(); self._show_section("Dashboard"); self.protocol("WM_DELETE_WINDOW", self._close); self.after(50, self._poll_worker)

    def _style(self):
        s=ttk.Style(self); s.theme_use("clam"); s.configure("App.TFrame",background="#10151c"); s.configure("Sidebar.TFrame",background="#0c1117")
        s.configure("Card.TFrame",background="#18212c"); s.configure("Title.TLabel",background="#10151c",foreground="#edf2f7",font=("Segoe UI",22,"bold")); s.configure("Muted.TLabel",background="#10151c",foreground="#9eacba",font=("Segoe UI",10))
        s.configure("Nav.TButton",background="#0c1117",foreground="#c4cfda",anchor="w",padding=(14,9),borderwidth=0); s.map("Nav.TButton",background=[("active","#202c38")],foreground=[("active","#ffffff")]); s.configure("Primary.TButton",background="#8e6b25",foreground="#ffffff",padding=(12,8)); s.configure("CardTitle.TLabel",background="#18212c",foreground="#edf2f7",font=("Segoe UI",14,"bold")); s.configure("CardValue.TLabel",background="#18212c",foreground="#d5a84b",font=("Segoe UI",24,"bold"))

    def _shell(self):
        shell=ttk.Frame(self,style="App.TFrame"); shell.pack(fill="both",expand=True); side=ttk.Frame(shell,width=250,style="Sidebar.TFrame",padding=14); side.pack(side="left",fill="y"); side.pack_propagate(False)
        tk.Label(side,text="THAMUDIC",bg="#0c1117",fg="#edf2f7",font=("Segoe UI",17,"bold")).pack(anchor="w",padx=8); tk.Label(side,text="NLP SCANNER" if self.mode=="nlp" else "SCANNER",bg="#0c1117",fg="#d5a84b",font=("Segoe UI",17,"bold")).pack(anchor="w",padx=8,pady=(0,20))
        for name in ["Dashboard","Scanner","Translator","Historical Objects","Inscriptions","Ancient Scripts","Sources & Rights","Database","Research"]: ttk.Button(side,text=name,style="Nav.TButton",command=lambda n=name:self._show_section(n)).pack(fill="x",pady=2)
        ttk.Separator(side).pack(fill="x",pady=15)
        for text,cmd,style in [("Import Image / PDF",self.import_media,"TButton"),("Scan Current Image",self.scan,"Primary.TButton"),("NLP Analyze",self.nlp_run,"TButton"),("Translate / Transliterate",self.translate_current,"TButton"),("Export Report / JSON",self.export_report,"TButton"),("Print Report",self.print_report,"TButton"),("Clear Workspace",self.clear_current,"TButton")]: ttk.Button(side,text=text,style=style,command=cmd).pack(fill="x",pady=2)
        main=ttk.Frame(shell,style="App.TFrame",padding=(24,20,24,10)); main.pack(side="left",fill="both",expand=True); head=ttk.Frame(main,style="App.TFrame"); head.pack(fill="x",pady=(0,16)); ttk.Label(head,textvariable=self.section_var,style="Title.TLabel").pack(side="left"); ttk.Label(head,textvariable=self.status_var,style="Muted.TLabel").pack(side="right"); self.workspace=ttk.Frame(main,style="App.TFrame"); self.workspace.pack(fill="both",expand=True)

    def _clear(self):
        for c in self.workspace.winfo_children(): c.destroy()
    def _show_section(self,section):
        self.section_var.set(section); self._clear(); builders={"Dashboard":self._dashboard,"Scanner":self._scanner,"Translator":self._translator,"Historical Objects":self._catalog,"Inscriptions":self._catalog,"Ancient Scripts":self._scripts,"Sources & Rights":self._sources,"Database":self._database,"Research":self._research}; builders.get(section,self._dashboard)()
    def _card(self,p,title,value,col):
        c=ttk.Frame(p,style="Card.TFrame",padding=18); c.grid(row=0,column=col,sticky="nsew",padx=6); ttk.Label(c,text=title,style="CardTitle.TLabel").pack(anchor="w"); ttk.Label(c,text=value,style="CardValue.TLabel").pack(anchor="w",pady=(7,0))
    def _dashboard(self):
        st=self.db.stats(); cards=ttk.Frame(self.workspace,style="App.TFrame"); cards.pack(fill="x",pady=(0,20)); [cards.columnconfigure(i,weight=1) for i in range(4)]
        for i,(a,b) in enumerate([("Objects",st["objects"]),("Annotations",st["annotations"]),("Sources",st["sources"]),("Schema","v"+str(st["schema_version"]))]): self._card(cards,a,str(b),i)
        p=ttk.LabelFrame(self.workspace,text="Recent research records",padding=10); p.pack(fill="both",expand=True); t=ttk.Treeview(p,columns=("id","title","period","script","type"),show="headings"); [t.heading(c,text=c.title()) for c in ("id","title","period","script","type")]; t.pack(fill="both",expand=True)
        for r in self.db.rows()[:10]: t.insert("","end",values=(r["id"],r["title"],r["period"],r["script"],r["object_type"]))
    def _text(self,p,label,h):
        ttk.Label(p,text=label).pack(anchor="w"); w=tk.Text(p,height=h,wrap="word",undo=True); w.pack(fill="x",pady=(2,7)); return w
    def _scanner(self):
        pan=ttk.Panedwindow(self.workspace,orient="horizontal"); pan.pack(fill="both",expand=True); left=ttk.Frame(pan,padding=10); right=ttk.Frame(pan,padding=10); pan.add(left,weight=3); pan.add(right,weight=2); self.image_label=ttk.Label(left,text="Import an inscription photograph or PDF",anchor="center"); self.image_label.pack(fill="both",expand=True)
        box=ttk.LabelFrame(right,text="Classification & Evidence",padding=12); box.pack(fill="both",expand=True); ttk.Label(box,text="Script / variety").pack(anchor="w"); ttk.Combobox(box,textvariable=self.script_var,values=SCRIPTS,state="readonly").pack(fill="x",pady=(0,7)); ttk.Label(box,text="Historical period").pack(anchor="w"); ttk.Combobox(box,textvariable=self.period_var,values=PERIODS,state="readonly").pack(fill="x",pady=(0,7)); ttk.Label(box,text="Object type").pack(anchor="w"); ttk.Entry(box,textvariable=self.object_type).pack(fill="x",pady=(0,7))
        self.source_text=self._text(box,"Detected / source text (Unicode ONA or scholarly transliteration)",3); self.translit=self._text(box,"Transliteration",3); self.arabic=self._text(box,"Arabic translation / notes",3); self.english=self._text(box,"English translation / notes",3); self.notes=self._text(box,"Provenance / bibliography / NLP evidence",3)
        bar=ttk.Frame(box); bar.pack(fill="x"); ttk.Button(bar,text="Translate",style="Primary.TButton",command=self.translate_current).pack(side="left",fill="x",expand=True,padx=2); ttk.Button(bar,text="NLP Analyze",command=self.nlp_run).pack(side="left",fill="x",expand=True,padx=2); ttk.Button(bar,text="Add reviewed evidence",command=self.add_current_object).pack(side="left",fill="x",expand=True,padx=2)
    def _translator(self):
        p=ttk.LabelFrame(self.workspace,text="Evidence-aware Translation",padding=14); p.pack(fill="both",expand=True); ttk.Label(p,text="Enter Unicode Old North Arabian or scholarly transliteration. Output is advisory and requires review.").pack(anchor="w"); src=self._text(p,"Input",7); tr=self._text(p,"Transliteration",4); ar=self._text(p,"Arabic translation",4); en=self._text(p,"English translation",4)
        def run():
            r=build_text_outputs(src.get("1.0","end-1c"));
            for w,k in ((tr,"transliteration"),(ar,"translation_ar"),(en,"translation_en")): w.delete("1.0","end"); w.insert("1.0",r[k])
        ttk.Button(p,text="Analyze / Translate",style="Primary.TButton",command=run).pack(anchor="e")
    def _catalog(self):
        bar=ttk.Frame(self.workspace); bar.pack(fill="x",pady=(0,10)); ttk.Entry(bar,textvariable=self.search_var).pack(side="left",fill="x",expand=True); ttk.Button(bar,text="Search",command=self._refresh_catalog).pack(side="left",padx=6); ttk.Button(bar,text="Refresh",command=self._refresh_catalog).pack(side="left"); self.catalog_tree=ttk.Treeview(self.workspace,columns=("id","title","period","type","script","source"),show="headings"); [self.catalog_tree.heading(c,text=c.title()) for c in ("id","title","period","type","script","source")]; self.catalog_tree.pack(fill="both",expand=True); self._refresh_catalog()
    def _refresh_catalog(self):
        if not hasattr(self,"catalog_tree"): return
        for i in self.catalog_tree.get_children(): self.catalog_tree.delete(i)
        for r in self.db.rows(self.search_var.get()): self.catalog_tree.insert("","end",values=(r["id"],r["title"],r["period"],r["object_type"],r["script"],r["source_image"]))
    def _scripts(self):
        t=ttk.Treeview(self.workspace,columns=("key","name","description"),show="headings"); [t.heading(c,text=c.title()) for c in ("key","name","description")]; t.pack(fill="both",expand=True)
        for x in SCRIPTS: t.insert("","end",values=(x.lower().replace(" ","_"),x,"Ancient North Arabian research script family"))
    def _sources(self):
        t=ttk.Treeview(self.workspace,columns=("name","rights","note"),show="headings"); [t.heading(c,text=c.title()) for c in ("name","rights","note")]; t.pack(fill="both",expand=True)
        for r in [("Local image/PDF","User-controlled","Process only material you are authorized to use."),("ONA Unicode","Unicode Standard","Deterministic character mapping."),("Lexical layer","Research evidence","Unknown words remain unresolved.")]: t.insert("","end",values=r)
    def _database(self):
        p=ttk.LabelFrame(self.workspace,text="Canonical SQLite Database",padding=14); p.pack(fill="both",expand=True); t=tk.Text(p,wrap="word"); t.pack(fill="both",expand=True); t.insert("1.0",json.dumps(self.db.stats(),ensure_ascii=False,indent=2)); t.configure(state="disabled")
    def _research(self):
        p=ttk.LabelFrame(self.workspace,text="Research Workflow",padding=16); p.pack(fill="both",expand=True)
        for i,x in enumerate(["Import photograph or PDF","Normalize and segment glyphs","Review candidate components","Record transliteration and translations","Attach provenance and rights","Persist reviewed evidence in SQLite","Export JSON / CSV / PDF"],1): ttk.Label(p,text=str(i)+". "+x).pack(anchor="w",pady=5)
    def import_media(self):
        path=filedialog.askopenfilename(filetypes=[("Images/PDF","*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp *.pdf"),("All files","*.*")]);
        if not path:return
        p=Path(path)
        try:
            if p.suffix.lower()==".pdf":
                import fitz
                with fitz.open(p) as doc:
                    if not doc.page_count: raise ValueError("PDF contains no pages")
                    pix=doc[0].get_pixmap(matrix=fitz.Matrix(2,2),alpha=False); tmp=Path(tempfile.gettempdir())/(p.stem+"_page1.png"); pix.save(tmp); self.current_image=tmp
            else:self.current_image=p
            self._show_section("Scanner")
            with Image.open(self.current_image) as im: im=im.copy(); im.thumbnail((900,700)); self.preview=ImageTk.PhotoImage(im)
            self.image_label.configure(image=self.preview,text=""); self.status_var.set("Imported "+p.name)
        except Exception as exc: messagebox.showerror("Import error",type(exc).__name__+": "+str(exc))
    def scan(self):
        if self.scan_running: self.status_var.set("A scan is already running"); return
        if not self.current_image: self.import_media(); return
        if not Path(self.current_image).exists(): messagebox.showerror("Scan error","The selected image no longer exists."); return
        self._show_section("Scanner"); hint=self.source_text.get("1.0","end-1c"); self.scan_generation += 1; generation=self.scan_generation; self.scan_running=True; self.status_var.set("Scanning in local worker… GUI remains responsive")
        future=self.executor.submit(scan_image_safe,Path(self.current_image),text_hint=hint); future.add_done_callback(lambda f:self.worker_queue.put((generation,f)))
    def _poll_worker(self):
        try:
            while True:
                generation,future=self.worker_queue.get_nowait()
                if generation != self.scan_generation: continue
                self.scan_running=False
                try: payload=future.result()
                except Exception as exc: payload={"ok":False,"error_type":type(exc).__name__,"error":str(exc)}
                if not payload.get("ok"):
                    self.status_var.set("Local worker failed"); messagebox.showerror("Local worker scan error",payload.get("error_type","Error")+": "+payload.get("error","Unknown error")); continue
                self.current_result=payload["result"]; self._populate_result(); self.status_var.set("Scan complete: "+str(len(self.current_result.get("glyphs",[])))+" candidate components")
        except queue.Empty: pass
        self.after(50,self._poll_worker)
    def _populate_result(self):
        r=self.current_result or {}
        for w,k in ((self.source_text,"source_text"),(self.translit,"transliteration"),(self.arabic,"translation_ar"),(self.english,"translation_en")): w.delete("1.0","end"); w.insert("1.0",r.get(k,""))
        self.notes.delete("1.0","end"); self.notes.insert("1.0",chr(10).join(r.get("notes",[])))
    def translate_current(self):
        if not hasattr(self,"source_text"): self._show_section("Scanner")
        r=build_text_outputs(self.source_text.get("1.0","end-1c"))
        for w,k in ((self.translit,"transliteration"),(self.arabic,"translation_ar"),(self.english,"translation_en")): w.delete("1.0","end"); w.insert("1.0",r[k])
        self.status_var.set("Translation/transliteration updated")
    def nlp_run(self):
        if not hasattr(self,"source_text"): self._show_section("Scanner")
        r=nlp_analyze(self.source_text.get("1.0","end-1c"))
        for w,k in ((self.translit,"transliteration"),(self.arabic,"translation_ar"),(self.english,"translation_en")): w.delete("1.0","end"); w.insert("1.0",r[k])
        self.notes.delete("1.0","end"); self.notes.insert("1.0",json.dumps({"tokens":r["tokens"],"known_tokens":r["known_tokens"],"unknown_tokens":r["unknown_tokens"],"confidence":r["confidence"]},ensure_ascii=False,indent=2)); self.status_var.set("NLP analysis complete: "+str(r["token_count"])+" token(s)")
    def add_current_object(self):
        if not self.current_result: self.status_var.set("Run a scan first"); return
        r={"title":Path(self.current_result.get("source_image","inscription")).stem,"script":self.script_var.get(),"period":self.period_var.get(),"object_type":self.object_type.get(),"source_image":self.current_result.get("source_image",""),"transliteration":self.translit.get("1.0","end-1c"),"translation_ar":self.arabic.get("1.0","end-1c"),"translation_en":self.english.get("1.0","end-1c"),"provenance":self.notes.get("1.0","end-1c"),"evidence":self.current_result}; self.status_var.set("Catalog record created: "+str(self.db.add(r)))
    def report_data(self):
        r=self.current_result or {}; get=lambda w,k:r.get(k,"") if not hasattr(self,w) else getattr(self,w).get("1.0","end-1c")
        return {"generated_at":datetime.now().isoformat(timespec="seconds"),"mode":self.mode,"source_image":r.get("source_image",""),"script":self.script_var.get(),"period":self.period_var.get(),"source_text":get("source_text","source_text"),"transliteration":get("translit","transliteration"),"translation_ar":get("arabic","translation_ar"),"translation_en":get("english","translation_en"),"notes":get("notes",""),"glyph_count":len(r.get("glyphs",[]))}
    def export_report(self):
        p0=filedialog.asksaveasfilename(defaultextension=".json",filetypes=[("JSON","*.json"),("CSV","*.csv"),("PDF","*.pdf"),("Text","*.txt")]);
        if not p0:return
        p=Path(p0); d=self.report_data()
        try:
            if p.suffix.lower()==".json": p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding="utf-8")
            elif p.suffix.lower()==".csv":
                with p.open("w",newline="",encoding="utf-8-sig") as f:
                    w=csv.writer(f); w.writerow(["field","value"]); [w.writerow([k,v]) for k,v in d.items()]
            elif p.suffix.lower()==".pdf": self.write_pdf(p,d)
            else:p.write_text(chr(10).join(str(k)+": "+str(v) for k,v in d.items()),encoding="utf-8")
            self.status_var.set("Exported "+p.name)
        except Exception as exc: messagebox.showerror("Export error",type(exc).__name__+": "+str(exc))
    def write_pdf(self,path,d):
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        styles=getSampleStyleSheet(); story=[Paragraph("Thamudic / NLP Research Report",styles["Title"]),Spacer(1,10)]
        for k,v in d.items(): story += [Paragraph("<b>"+str(k)+"</b>: "+str(v).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace(chr(10),"<br/>"),styles["BodyText"]),Spacer(1,5)]
        SimpleDocTemplate(str(path),pagesize=A4).build(story)
    def print_report(self):
        p=Path(tempfile.gettempdir())/"thamudic_research_report.pdf"
        try:
            self.write_pdf(p,self.report_data())
            if os.name=="nt": os.startfile(str(p),"print")
            else:
                import subprocess; subprocess.Popen(["lp",str(p)])
            self.status_var.set("Print job submitted")
        except Exception as exc: messagebox.showerror("Print error",type(exc).__name__+": "+str(exc))
    def clear_current(self):
        self.scan_generation += 1; self.scan_running=False; self.current_result=None; self.current_image=None; self._show_section("Scanner"); self.status_var.set("Workspace cleared")
    def _close(self):
        self.scan_generation += 1; self.executor.shutdown(wait=False,cancel_futures=True); self.destroy()


class NLPApp(App):
    def __init__(self,db_path="ancient_objects.sqlite"): super().__init__(mode="nlp",db_path=db_path)


def main():
    import argparse
    p=argparse.ArgumentParser(description="Unified Thamudic and NLP scanner")
    p.add_argument("mode",choices=("thamudic","nlp"),default="nlp",nargs="?"); p.add_argument("--db",default="ancient_objects.sqlite"); a=p.parse_args(); (NLPApp if a.mode=="nlp" else App)(db_path=a.db).mainloop()

if __name__=="__main__": main()
