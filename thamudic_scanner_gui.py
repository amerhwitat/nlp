#!/usr/bin/env python3
"""Unified Tkinter desktop application for Thamudic/ANA research.

Tabs: Scanner, Object Catalog, Sources. Keeps recognition evidence,
translations, provenance, rights and object metadata separate and exportable.
"""
from __future__ import annotations
import json, tempfile
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from ancient_objects_db import ObjectDatabase
from ancient_script_registry import SCRIPTS
from historical_periods import PERIODS
from object_sources import source_catalog
from softr_export import export_softr_csv, export_softr_json
from thamudic_scanner import scan_image

class AncientResearchApp(tk.Tk):
    def __init__(self,db_path='ancient_objects.sqlite'):
        super().__init__(); self.title('Ancient Script Scanner & Historical Object Catalog'); self.geometry('1400x900'); self.minsize(1050,700)
        self.db=ObjectDatabase(db_path); self.current_image=None; self.current_result=None; self.preview=None; self._build(); self.refresh_catalog(); self.protocol('WM_DELETE_WINDOW',self._close)
    def _close(self): self.db.close(); self.destroy()
    def _build(self):
        top=ttk.Frame(self,padding=8); top.pack(fill='x')
        for text,cmd in [('Import Image/PDF',self.import_media),('Scan',self.scan),('Add to Catalog',self.add_current_object),('Export Softr CSV',self.export_softr),('Export JSON',self.export_json)]: ttk.Button(top,text=text,command=cmd).pack(side='left',padx=2)
        self.status=tk.StringVar(value='Ready'); ttk.Label(top,textvariable=self.status).pack(side='right')
        nb=ttk.Notebook(self); nb.pack(fill='both',expand=True,padx=8,pady=(0,8)); self.scan_tab=ttk.Frame(nb); self.catalog_tab=ttk.Frame(nb); self.sources_tab=ttk.Frame(nb)
        nb.add(self.scan_tab,text='Scanner / Evidence'); nb.add(self.catalog_tab,text='Object Catalog'); nb.add(self.sources_tab,text='Sources & Rights'); self._build_scan_tab(); self._build_catalog_tab(); self._build_sources_tab()
    def _build_scan_tab(self):
        pan=ttk.Panedwindow(self.scan_tab,orient='horizontal'); pan.pack(fill='both',expand=True); left=ttk.Frame(pan,padding=8); right=ttk.Frame(pan,padding=8); pan.add(left,weight=3); pan.add(right,weight=2)
        self.image_label=ttk.Label(left,text='Import an inscription photograph or PDF',anchor='center'); self.image_label.pack(fill='both',expand=True)
        meta=ttk.LabelFrame(right,text='Classification',padding=8); meta.pack(fill='x'); ttk.Label(meta,text='Script / variety').pack(anchor='w')
        self.script_var=tk.StringVar(value='old_north_arabian'); ttk.Combobox(meta,textvariable=self.script_var,state='readonly',values=sorted(SCRIPTS)).pack(fill='x')
        ttk.Label(meta,text='Historical period').pack(anchor='w',pady=(8,0)); self.period_var=tk.StringVar(value='iron_age'); ttk.Combobox(meta,textvariable=self.period_var,state='readonly',values=[p['key'] for p in PERIODS]).pack(fill='x')
        ttk.Label(meta,text='Object type').pack(anchor='w',pady=(8,0)); self.object_type=tk.StringVar(value='inscription'); ttk.Entry(meta,textvariable=self.object_type).pack(fill='x')
        fields=ttk.LabelFrame(right,text='Human-reviewed evidence',padding=8); fields.pack(fill='both',expand=True,pady=8)
        self.translit=self._textfield(fields,'Transliteration',5); self.arabic=self._textfield(fields,'Arabic translation / notes',5,rtl=True); self.english=self._textfield(fields,'English translation / notes',5); self.notes=self._textfield(fields,'Research / provenance / bibliography',7)
        self.confidence=tk.DoubleVar(value=0.0); ttk.Label(fields,text='Reviewer confidence (0–1)').pack(anchor='w'); ttk.Scale(fields,from_=0,to=1,variable=self.confidence,orient='horizontal').pack(fill='x')
    def _textfield(self,parent,label,height,rtl=False):
        ttk.Label(parent,text=label).pack(anchor='w'); w=tk.Text(parent,height=height,wrap='word',undo=True); w.pack(fill='x',pady=(1,7));
        if rtl: w.tag_configure('rtl',justify='right'); w.tag_add('rtl','1.0','end')
        return w
    def _build_catalog_tab(self):
        bar=ttk.Frame(self.catalog_tab,padding=8); bar.pack(fill='x'); self.search_var=tk.StringVar(); ttk.Entry(bar,textvariable=self.search_var,width=30).pack(side='left'); ttk.Button(bar,text='Search',command=self.refresh_catalog).pack(side='left',padx=4)
        self.filter_period=tk.StringVar(value=''); ttk.Combobox(bar,textvariable=self.filter_period,state='readonly',values=['']+[p['key'] for p in PERIODS],width=18).pack(side='left',padx=4)
        self.tree=ttk.Treeview(self.catalog_tab,columns=('id','title','period','type','script','source','rights'),show='headings'); self.tree.pack(fill='both',expand=True,padx=8,pady=8)
        for c,w in [('id',180),('title',240),('period',130),('type',130),('script',140),('source',150),('rights',150)]: self.tree.heading(c,text=c.title()); self.tree.column(c,width=w)
        self.tree.bind('<Double-1>',self.show_selected)
    def _build_sources_tab(self):
        cols=('name','homepage','rights_policy','image_policy'); self.sources=ttk.Treeview(self.sources_tab,columns=cols,show='headings'); self.sources.pack(fill='both',expand=True,padx=8,pady=8)
        for c in cols: self.sources.heading(c,text=c.replace('_',' ').title()); self.sources.column(c,width=280 if c!='name' else 180)
        for s in source_catalog(): self.sources.insert('', 'end', values=tuple(s.get(c,'') for c in cols))
        ttk.Label(self.sources_tab,text='Remote media is linked by URL/IIIF by default. Image reuse must follow each source record rights statement.',wraplength=1000).pack(anchor='w',padx=10,pady=8)
    def import_media(self):
        path=filedialog.askopenfilename(filetypes=[('Images/PDF','*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp *.pdf'),('All files','*.*')]);
        if not path:return
        p=Path(path)
        if p.suffix.lower()=='.pdf':
            try:
                import fitz; doc=fitz.open(p); pix=doc[0].get_pixmap(matrix=fitz.Matrix(2,2),alpha=False); tmp=Path(tempfile.gettempdir())/(p.stem+'_page1.png'); pix.save(tmp); self.current_image=tmp
            except Exception as exc: messagebox.showerror('PDF import',f'PyMuPDF is required for PDF pages: {exc}'); return
        else: self.current_image=p
        image=Image.open(self.current_image); image.thumbnail((850,700)); self.preview=ImageTk.PhotoImage(image); self.image_label.configure(image=self.preview,text=''); self.status.set(f'Imported {p.name}')
    def scan(self):
        if not self.current_image: messagebox.showinfo('Import','Import an image or PDF first.'); return
        try: self.current_result=scan_image(self.current_image)
        except Exception as exc: messagebox.showerror('Scan error',str(exc)); return
        profile=SCRIPTS[self.script_var.get()]; self.current_result['script_profile']=profile.__dict__; self.current_result['period_key']=self.period_var.get(); self.current_result['object_type']=self.object_type.get(); self.current_result['human_review']=self._review(); self.status.set(f"Scan complete: {len(self.current_result.get('glyphs',[]))} candidate components")
    def _review(self): return {'transliteration':self.translit.get('1.0','end-1c'),'translation_ar':self.arabic.get('1.0','end-1c'),'translation_en':self.english.get('1.0','end-1c'),'notes':self.notes.get('1.0','end-1c'),'confidence':round(self.confidence.get(),3)}
    def add_current_object(self):
        if not self.current_result: self.scan()
        if not self.current_result:return
        r=self._review(); profile=SCRIPTS[self.script_var.get()]; rid=self.db.add_object({'title':Path(self.current_result.get('source_image','inscription')).stem,'period_key':self.period_var.get(),'object_type':self.object_type.get(),'script_key':profile.key,'description':'Scanner evidence record','transliteration':r['transliteration'],'translation_ar':r['translation_ar'],'translation_en':r['translation_en'],'source_name':'Local scanner','image_local_path':self.current_result.get('source_image',''),'confidence':r['confidence'],'provenance':r['notes'],'tags':['thamudic','ancient-script']}); self.status.set(f'Catalog record created: {rid}'); self.refresh_catalog()
    def refresh_catalog(self):
        if not hasattr(self,'tree'): return
        for item in self.tree.get_children(): self.tree.delete(item)
        for r in self.db.list_objects(query=self.search_var.get() if hasattr(self,'search_var') else '',period_key=self.filter_period.get() if hasattr(self,'filter_period') and self.filter_period.get() else None): self.tree.insert('', 'end', values=(r.get('id',''),r.get('title',''),r.get('period_name',''),r.get('object_type',''),r.get('script_key',''),r.get('source_name',''),r.get('license','')))
    def show_selected(self,_=None):
        sel=self.tree.selection();
        if not sel:return
        rid=self.tree.item(sel[0])['values'][0]; messagebox.showinfo('Object record',json.dumps(self.db.get_object(rid),ensure_ascii=False,indent=2))
    def export_softr(self):
        path=filedialog.asksaveasfilename(defaultextension='.csv',filetypes=[('CSV','*.csv')]);
        if path: export_softr_csv(self.db.list_objects(),path); self.status.set(f'Exported Softr CSV: {path}')
    def export_json(self):
        path=filedialog.asksaveasfilename(defaultextension='.json',filetypes=[('JSON','*.json')]);
        if path: export_softr_json(self.db.list_objects(),path); self.status.set(f'Exported JSON: {path}')

if __name__=='__main__': AncientResearchApp().mainloop()
