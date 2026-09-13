#!/usr/bin/env python3
"""All-in-one Thamudic / Ancient North Arabian scanner.

No Tesseract and no camel_tools. Includes image/PDF ingestion, deterministic
segmentation, Unicode Old North Arabian transliteration, a small evidence-aware
lexical translation layer, and the Tkinter desktop GUI.
"""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageFilter, ImageOps, ImageTk

ONA = {
    0x10A80:"h", 0x10A81:"l", 0x10A82:"ḥ", 0x10A83:"m", 0x10A84:"q",
    0x10A85:"w", 0x10A86:"s²", 0x10A87:"r", 0x10A88:"b", 0x10A89:"t",
    0x10A8A:"s¹", 0x10A8B:"k", 0x10A8C:"n", 0x10A8D:"ḫ", 0x10A8E:"ṣ",
    0x10A8F:"s³", 0x10A90:"f", 0x10A91:"ʾ", 0x10A92:"ʿ", 0x10A93:"ḍ",
    0x10A94:"g", 0x10A95:"d", 0x10A96:"ġ", 0x10A97:"ṭ", 0x10A98:"z",
    0x10A99:"ḏ", 0x10A9A:"y", 0x10A9B:"ṯ", 0x10A9C:"ẓ",
}
LEXICON = {
    "mlk": ("ملك", "king"), "bn": ("بن", "son"), "bt": ("بنت", "daughter"),
    "ʾl": ("إل / الله", "El / God"), "ʿbd": ("عبد", "servant"),
    "byt": ("بيت", "house"), "ʾrs": ("أرض", "land"), "ywm": ("يوم", "day"),
    "šnt": ("سنة", "year"), "snt": ("سنة", "year"), "rb": ("رب", "lord"),
    "bny": ("بنى", "built"), "ḥrb": ("حرب", "war"), "ʿyn": ("عين", "spring/eye"),
}

def transliterate_ona(text: str) -> str:
    return "".join(ONA.get(ord(ch), ch if not ch.isspace() else " ") for ch in text)

def _normalize_token(token: str) -> str:
    return token.lower().replace("š", "s").replace("ṯ", "th").replace("ḏ", "dh").replace("²", "2").replace("¹", "1").replace("³", "3")

def translate_transliteration(text: str):
    raw = text.strip()
    if not raw:
        return "", ""
    key = _normalize_token(raw.replace(" ", ""))
    if key in LEXICON:
        return LEXICON[key]
    ar, en = [], []
    for token in re.split(r"[\s,;|]+", raw):
        if not token:
            continue
        hit = LEXICON.get(_normalize_token(token))
        if hit:
            ar.append(hit[0]); en.append(hit[1])
        else:
            ar.append("[غير معروف]"); en.append(f"[unresolved: {token}]")
    return " ".join(ar), " ".join(en)

def build_text_outputs(source_text: str):
    tr = transliterate_ona(source_text) if any(0x10A80 <= ord(c) <= 0x10A9F for c in source_text) else source_text.strip()
    ar, en = translate_transliteration(tr)
    return {"source_text": source_text, "transliteration": tr, "translation_ar": ar, "translation_en": en}

def normalize(image: Image.Image, scale: int = 2):
    image = ImageOps.exif_transpose(image).convert("L")
    image = ImageOps.autocontrast(image).filter(ImageFilter.MedianFilter(3))
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), Image.Resampling.LANCZOS)
    return image

def segment(image: Image.Image, threshold: int = 150, min_area: int = 25):
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
            stack = [(y0, x0)]; seen[y0, x0] = True; xs=[]; ys=[]
            while stack:
                y, x = stack.pop(); xs.append(x); ys.append(y)
                for dy in (-1,0,1):
                    for dx in (-1,0,1):
                        if not dx and not dy: continue
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True; stack.append((ny,nx))
            if len(xs) >= min_area:
                boxes.append((min(xs), min(ys), max(xs)+1, max(ys)+1, len(xs)))
    return sorted(boxes, key=lambda b:(b[1], b[0]))

def scan_image(path: Path, threshold=150, min_area=25, scale=2, text_hint=""):
    image = normalize(Image.open(path), scale)
    boxes = segment(image, threshold, min_area)
    glyphs=[]
    for i,(x1,y1,x2,y2,area) in enumerate(boxes):
        width,height=x2-x1,y2-y1
        fill=area/max(width*height,1)
        confidence=min(1.0,max(0.0,0.5*fill+0.5*min(width,height)/max(width,height)))
        glyphs.append({"index":i,"x":x1,"y":y1,"width":width,"height":height,"area":area,"aspect_ratio":round(width/max(height,1),4),"confidence":round(confidence,4)})
    outputs=build_text_outputs(text_hint.strip()) if text_hint.strip() else {"source_text":"","transliteration":"","translation_ar":"","translation_en":""}
    outputs.update({
        "schema":"thamudic-scanner/v2", "source_image":str(path),
        "script_family":"Ancient North Arabian / Old North Arabian",
        "recognition_status":"text_hint_translated" if text_hint.strip() else "segmentation_only",
        "glyphs":glyphs,
        "notes":["Thamudic is a scholarly umbrella for multiple Ancient North Arabian varieties.","Segmentation boxes are not linguistic readings.","Candidate translations require scholarly verification."]})
    return outputs

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Thamudic Scanner — Ancient Languages Research Workbench")
        self.geometry("1500x920"); self.minsize(1120,720); self.configure(background="#10151c")
        self.current_image=None; self.preview=None; self.current_result=None
        self.status_var=tk.StringVar(value="Ready"); self.section_var=tk.StringVar(value="Scanner")
        self.script_var=tk.StringVar(value="Old North Arabian / Thamudic")
        self._style(); self._shell(); self._scanner()
    def _style(self):
        s=ttk.Style(self); s.theme_use("clam")
        s.configure("App.TFrame",background="#10151c"); s.configure("Sidebar.TFrame",background="#0c1117")
        s.configure("Title.TLabel",background="#10151c",foreground="#edf2f7",font=("Segoe UI",22,"bold"))
        s.configure("Muted.TLabel",background="#10151c",foreground="#9eacba",font=("Segoe UI",10))
        s.configure("Nav.TButton",background="#0c1117",foreground="#c4cfda",anchor="w",padding=(14,9),borderwidth=0)
        s.map("Nav.TButton",background=[("active","#202c38")],foreground=[("active","#ffffff")])
        s.configure("Primary.TButton",background="#8e6b25",foreground="#ffffff",padding=(12,8))
    def _shell(self):
        shell=ttk.Frame(self,style="App.TFrame"); shell.pack(fill="both",expand=True)
        side=ttk.Frame(shell,width=250,style="Sidebar.TFrame",padding=14); side.pack(side="left",fill="y"); side.pack_propagate(False)
        tk.Label(side,text="THAMUDIC",bg="#0c1117",fg="#edf2f7",font=("Segoe UI",17,"bold")).pack(anchor="w",padx=8)
        tk.Label(side,text="SCANNER",bg="#0c1117",fg="#d5a84b",font=("Segoe UI",17,"bold")).pack(anchor="w",padx=8,pady=(0,20))
        ttk.Button(side,text="Scanner",style="Nav.TButton",command=self._scanner).pack(fill="x",pady=2)
        ttk.Button(side,text="Translator",style="Nav.TButton",command=self._translator).pack(fill="x",pady=2)
        ttk.Separator(side).pack(fill="x",pady=15)
        ttk.Button(side,text="Import Image / PDF",command=self.import_media).pack(fill="x",pady=2)
        ttk.Button(side,text="Scan Current Image",style="Primary.TButton",command=self.scan).pack(fill="x",pady=2)
        main=ttk.Frame(shell,style="App.TFrame",padding=(24,20,24,10)); main.pack(side="left",fill="both",expand=True)
        head=ttk.Frame(main,style="App.TFrame"); head.pack(fill="x",pady=(0,16))
        ttk.Label(head,textvariable=self.section_var,style="Title.TLabel").pack(side="left")
        ttk.Label(head,textvariable=self.status_var,style="Muted.TLabel").pack(side="right")
        self.workspace=ttk.Frame(main,style="App.TFrame"); self.workspace.pack(fill="both",expand=True)
    def _clear(self):
        for c in self.workspace.winfo_children(): c.destroy()
    def _text(self,parent,label,height):
        ttk.Label(parent,text=label).pack(anchor="w")
        w=tk.Text(parent,height=height,wrap="word",undo=True); w.pack(fill="x",pady=(2,7)); return w
    def _scanner(self):
        self.section_var.set("Scanner"); self._clear()
        pan=ttk.Panedwindow(self.workspace,orient="horizontal"); pan.pack(fill="both",expand=True)
        left=ttk.Frame(pan,padding=10); right=ttk.Frame(pan,padding=10); pan.add(left,weight=3); pan.add(right,weight=2)
        self.image_label=ttk.Label(left,text="Import an inscription photograph or PDF",anchor="center"); self.image_label.pack(fill="both",expand=True)
        box=ttk.LabelFrame(right,text="Classification & Evidence",padding=12); box.pack(fill="both",expand=True)
        ttk.Label(box,text="Script / variety").pack(anchor="w"); ttk.Entry(box,textvariable=self.script_var).pack(fill="x",pady=(0,10))
        self.source_text=self._text(box,"Detected / source text (Unicode ONA or scholarly transliteration)",4)
        self.translit=self._text(box,"Transliteration",4); self.arabic=self._text(box,"Arabic translation / notes",4); self.english=self._text(box,"English translation / notes",4); self.notes=self._text(box,"Provenance / bibliography",4)
        ttk.Button(box,text="Translate / Transliterate",style="Primary.TButton",command=self.translate_current).pack(fill="x",pady=3)
    def _translator(self):
        self.section_var.set("Translator"); self._clear()
        p=ttk.LabelFrame(self.workspace,text="Evidence-aware Translation",padding=14); p.pack(fill="both",expand=True)
        src=self._text(p,"Unicode Old North Arabian or scholarly transliteration",8); tr=self._text(p,"Transliteration",5); ar=self._text(p,"Arabic translation",5); en=self._text(p,"English translation",5)
        def run():
            r=build_text_outputs(src.get("1.0","end-1c"))
            for w,key in ((tr,"transliteration"),(ar,"translation_ar"),(en,"translation_en")):
                w.delete("1.0","end"); w.insert("1.0",r[key])
        ttk.Button(p,text="Analyze",style="Primary.TButton",command=run).pack(anchor="e")
    def import_media(self):
        path=filedialog.askopenfilename(filetypes=[("Images/PDF","*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp *.pdf"),("All files","*.*")])
        if not path:return
        p=Path(path)
        try:
            if p.suffix.lower()==".pdf":
                import fitz
                doc=fitz.open(p)
                if not doc.page_count: raise ValueError("PDF contains no pages")
                pix=doc[0].get_pixmap(matrix=fitz.Matrix(2,2),alpha=False)
                tmp=Path(tempfile.gettempdir())/(p.stem+"_page1.png"); pix.save(tmp); self.current_image=tmp
            else:self.current_image=p
            image=Image.open(self.current_image); image.thumbnail((900,700)); self.preview=ImageTk.PhotoImage(image)
            self._scanner(); self.image_label.configure(image=self.preview,text=""); self.status_var.set(f"Imported {p.name}")
            self.scan()
        except Exception as e: messagebox.showerror("Import error",str(e))
    def scan(self):
        if not self.current_image:return self.import_media()
        try:
            hint=self.source_text.get("1.0","end-1c") if hasattr(self,"source_text") else ""
            self.current_result=scan_image(self.current_image,text_hint=hint)
            self._scanner(); self._populate_result()
            self.status_var.set(f"Scan complete: {len(self.current_result['glyphs'])} candidate components")
        except Exception as e: messagebox.showerror("Scan error",str(e))
    def _populate_result(self):
        r=self.current_result
        for w,key in ((self.source_text,"source_text"),(self.translit,"transliteration"),(self.arabic,"translation_ar"),(self.english,"translation_en")):
            w.delete("1.0","end"); w.insert("1.0",r.get(key,""))
        self.notes.delete("1.0","end")
        self.notes.insert("1.0", "\n".join(r.get("notes",[])))
    def translate_current(self):
        r=build_text_outputs(self.source_text.get("1.0","end-1c"))
        for w,key in ((self.translit,"transliteration"),(self.arabic,"translation_ar"),(self.english,"translation_en")):
            w.delete("1.0","end"); w.insert("1.0",r[key])
        self.status_var.set("Translation/transliteration updated")

if __name__=="__main__": App().mainloop()
