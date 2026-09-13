#!/usr/bin/env python3
"""Standalone general NLP/media scanner with GUI.

Imports TXT/MD/CSV/PDF/images, extracts text locally where possible, optionally OCRs
images/scanned PDFs with EasyOCR, detects ancient scripts, transliterates Ancient
North Arabian text, and performs evidence-backed corpus translation. It deliberately
reports unsupported translation/OCR instead of fabricating results.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from thamudic.media_pipeline import extract_media_text, scan_translate_media
from thamudic.source_language_scanner import scan_source_language
from thamudic.ancient_translation import translate


class NLPScannerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NLP Scanner — Media → OCR → Scan → Transliterate → Translate")
        self.geometry("1200x850")
        self._build()

    def _build(self):
        bar=ttk.Frame(self); bar.pack(fill="x", padx=10, pady=8)
        ttk.Button(bar,text="Import image/PDF/text",command=self.import_file).pack(side="left")
        ttk.Label(bar,text="Script").pack(side="left",padx=(16,4))
        self.script=ttk.Combobox(bar,values=["Dadanitic","Safaitic","Hismaic","Taymanitic","Thamudic B"],state="readonly",width=16); self.script.set("Dadanitic"); self.script.pack(side="left")
        ttk.Label(bar,text="Target").pack(side="left",padx=(12,4))
        self.target=ttk.Combobox(bar,values=["en","ar"],state="readonly",width=8); self.target.set("en"); self.target.pack(side="left")
        ttk.Button(bar,text="Scan + translate",command=self.process).pack(side="left",padx=12)
        self.status=ttk.Label(bar,text="Ready"); self.status.pack(side="left",padx=10)
        ttk.Label(self,text="Extracted source / OCR text").pack(anchor="w",padx=10)
        self.source=tk.Text(self,height=12,wrap="word"); self.source.pack(fill="x",padx=10)
        panes=ttk.Panedwindow(self,orient="vertical"); panes.pack(fill="both",expand=True,padx=10,pady=8)
        f1=ttk.LabelFrame(panes,text="Transliteration / script scan"); f2=ttk.LabelFrame(panes,text="Translation / evidence")
        panes.add(f1,weight=1); panes.add(f2,weight=1)
        self.trans=tk.Text(f1,wrap="word"); self.trans.pack(fill="both",expand=True)
        self.out=tk.Text(f2,wrap="word"); self.out.pack(fill="both",expand=True)
        self.path=None

    def import_file(self):
        p=filedialog.askopenfilename(filetypes=[("Supported media","*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.pdf *.txt *.md *.csv"),("All files","*.*")])
        if not p:return
        self.path=Path(p)
        try:
            text,meta=extract_media_text(self.path)
            self.source.delete("1.0","end"); self.source.insert("1.0",text)
            self.status.config(text=f"Imported: {self.path.name} · {meta.get('provider','unknown')}" )
            self.process()
        except Exception as exc:
            messagebox.showerror("Import/OCR error",str(exc)); self.status.config(text="Import failed")

    def process(self):
        text=self.source.get("1.0","end-1c").strip()
        if not text:
            self.status.config(text="Import media or enter text first"); return
        try:
            scan=scan_source_language(text,language="ancient-north-arabian")
            result=translate(text,self.script.get(),self.target.get())
            self.trans.delete("1.0","end"); self.trans.insert("1.0",result["transliteration"])
            self.out.delete("1.0","end"); self.out.insert("1.0",result.get("translation") or "No corpus-backed translation is available for this reading.\n\n"+str({"status":result["translation_status"],"confidence":result["confidence"],"provenance":result["provenance"]}))
            self.status.config(text=f"{result['translation_status']} · {scan['matched_character_count']} script characters")
        except Exception as exc:
            messagebox.showerror("Processing error",str(exc))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("file",nargs="?",help="image/PDF/text file to process")
    parser.add_argument("--script",default="Dadanitic"); parser.add_argument("--target",default="en")
    parser.add_argument("--gui",action="store_true")
    args=parser.parse_args()
    if args.file and not args.gui:
        result=scan_translate_media(args.file,args.script,args.target)
        print(result["transliteration"]); print(result.get("translation") or "TRANSLATION_UNAVAILABLE")
        return
    app=NLPScannerApp()
    if args.file:
        app.path=Path(args.file)
        try:
            text,_=extract_media_text(args.file); app.source.insert("1.0",text); app.process()
        except Exception as exc: messagebox.showerror("Import/OCR error",str(exc))
    app.mainloop()
if __name__=="__main__": main()
