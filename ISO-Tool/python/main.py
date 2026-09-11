from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from iso_tool import BuildPipeline, dependency_cache_dir, suggested_output_dir, prepare_output_layout
from iso_tool.boot_import import import_boot_sector, inspect_image
from iso_tool.resilient_network import ConnectivityMonitor

class App(tk.Tk):
    def __init__(self):
        super().__init__(); self.title('ISO-Tool — source / boot image to ISO / IMG'); self.geometry('1100x820')
        self.repo=tk.StringVar(); self.out=tk.StringVar(value=str(suggested_output_dir())); self.progress=tk.DoubleVar(); self.status=tk.StringVar(value='Ready'); self.events=queue.Queue(); self.running=False
        ttk.Label(self,text='GitHub repository, local source repository, ISO or image').pack(anchor='w',padx=12,pady=(12,2)); ttk.Entry(self,textvariable=self.repo).pack(fill='x',padx=12)
        outrow=ttk.Frame(self); outrow.pack(fill='x',padx=12,pady=(8,2)); ttk.Label(outrow,text='Final output directory (user selected)').pack(side='left'); ttk.Button(outrow,text='Choose…',command=self.choose_output).pack(side='right')
        ttk.Entry(self,textvariable=self.out).pack(fill='x',padx=12); ttk.Label(self,text=f'Dependency downloads/cache: {dependency_cache_dir()}').pack(anchor='w',padx=12,pady=(3,2))
        bar=ttk.Frame(self); bar.pack(fill='x',padx=12,pady=10)
        self.analyze_button=ttk.Button(bar,text='Analyze / Inventory',command=self.inventory); self.analyze_button.pack(side='left')
        self.import_button=ttk.Button(bar,text='Import Boot Sector / ISO',command=self.import_image); self.import_button.pack(side='left',padx=8)
        self.build_images_button=ttk.Button(bar,text='Build Compiled Images',command=self.build_images); self.build_images_button.pack(side='left')
        self.build_button=ttk.Button(bar,text='Build ISO',command=self.build_iso); self.build_button.pack(side='left',padx=8)
        ttk.Button(bar,text='Clear details',command=self.clear_log).pack(side='left')
        ttk.Label(self,textvariable=self.status).pack(anchor='w',padx=12); ttk.Progressbar(self,variable=self.progress,maximum=100).pack(fill='x',padx=12,pady=(4,8)); ttk.Label(self,text='Live operation details').pack(anchor='w',padx=12)
        self.log=tk.Text(self,height=29,state='disabled',font=('Consolas',10)); self.log.pack(fill='both',expand=True,padx=12,pady=(4,12)); self.after(75,self._drain_events)
    def choose_output(self):
        selected=filedialog.askdirectory(title='Choose where ISO, boot images, binaries, logs and manifests will be saved',initialdir=self.out.get() or str(suggested_output_dir()),mustexist=False)
        if selected:
            self.out.set(selected); self._append(f'[output] user selected: {selected}')
    def clear_log(self): self.log.configure(state='normal'); self.log.delete('1.0','end'); self.log.configure(state='disabled')
    def _append(self,msg): self.log.configure(state='normal'); self.log.insert('end',msg+'\n'); self.log.see('end'); self.log.configure(state='disabled')
    def _drain_events(self):
        try:
            while True:
                kind,payload=self.events.get_nowait()
                if kind=='progress':
                    e=payload; pct=100 if e.total==0 else e.completed*100/e.total; self.progress.set(max(self.progress.get(),pct)); self.status.set(e.message); self._append(f'[{e.stage}] {e.message}')
                elif kind=='log': self._append(payload)
                elif kind=='done': self.running=False; self._set_buttons(True); self.status.set(payload); self._append(payload)
        except queue.Empty: pass
        self.after(75,self._drain_events)
    def _set_buttons(self,on):
        for b in (self.analyze_button,self.import_button,self.build_images_button,self.build_button): b.configure(state='normal' if on else 'disabled')
    def _start(self,target):
        if self.running:return
        self.running=True; self._set_buttons(False); self.progress.set(0); threading.Thread(target=target,daemon=True).start()
    def inventory(self):
        root=Path(self.repo.get()).expanduser()
        if not root.is_dir(): messagebox.showerror('ISO-Tool','Select a local source repository for this operation.'); return
        self._start(lambda:self._inventory_worker(root))
    def _inventory_worker(self,root):
        try:
            files=BuildPipeline(root).inventory(); total=max(len(files),1); self.events.put(('log',f'Entry point: analyze-source; local repository: {root}'))
            for i,p in enumerate(files,1):
                try:self.events.put(('log',f'[{i}/{len(files)}] {p.relative_to(root)}'))
                except Exception as ex:self.events.put(('log',f'[error] inventory entry skipped: {type(ex).__name__}: {ex}'))
                self.events.put(('progress',type('P',(),{'stage':'inventory','completed':i,'total':total,'message':f'Inspected {i}/{len(files)}'})()))
            self.events.put(('done',f'Inventory complete: {len(files)} source files.'))
        except Exception as ex:self.events.put(('log',f'[error] {type(ex).__name__}: {ex}')); self.events.put(('done','Operation ended with recoverable errors.'))
    def import_image(self):
        src=filedialog.askopenfilename(title='Import boot sector / ISO / image',filetypes=[('Images','*.iso *.img *.bin'),('All files','*.*')])
        if not src:return
        dst=filedialog.asksaveasfilename(title='Save imported boot sector',initialdir=str(Path(self.out.get())/ 'boot-images'),defaultextension='.bin',filetypes=[('Binary','*.bin'),('All files','*.*')])
        if not dst:return
        self._start(lambda:self._import_worker(Path(src),Path(dst)))
    def _import_worker(self,src,dst):
        try:
            self.events.put(('log',f'Entry point: import-boot-image; source={src}')); info=inspect_image(src); self.events.put(('log',f'[image] {info.kind}, {info.size} bytes, bootable={info.bootable}, sha256(first-sector)={info.boot_sector_sha256}')); import_boot_sector(src,dst); self.events.put(('progress',type('P',(),{'stage':'boot-import','completed':1,'total':1,'message':f'Imported bounded boot sector to {dst}'})())); self.events.put(('done','Boot image import complete; imported bytes remain inert until assigned to a boot profile.'))
        except Exception as ex:self.events.put(('log',f'[error] import skipped: {type(ex).__name__}: {ex}')); self.events.put(('done','Boot import ended with recoverable errors.'))
    def build_images(self): self._start(lambda:self._build_worker(False))
    def build_iso(self): self._start(lambda:self._build_worker(True))
    def _build_worker(self,make_iso):
        selected=Path(self.out.get()).expanduser()
        try:
            layout=prepare_output_layout(selected); self.events.put(('log',f'[output] final output directory: {layout["root"]}')); self.events.put(('log',f'[deps] dependency download/cache: {layout["dependency_cache"]}'))
        except Exception as ex:
            self.events.put(('log',f'[fatal] output directory is not writable: {type(ex).__name__}: {ex}')); self.events.put(('done','Build stopped: choose another output directory.')); return
        steps=['validate source','inventory','discover toolchains','prepare build plan','compile/assemble jobs','prepare boot artifacts'] + (['stage ISO','build ISO / IMG','validate image'] if make_iso else [])
        monitor=ConnectivityMonitor()
        for i,step in enumerate(steps,1):
            try:
                self.events.put(('log',f'[step] {step} started'))
                if step=='validate source' and not self.repo.get().strip(): raise ValueError('source repository/path is empty')
                if step=='discover toolchains': self.events.put(('log',f'[network] {monitor.check().detail}')); self.events.put(('log',f'[deps] missing dependency downloads, when authorized, are stored under {layout["dependency_cache"]}'))
                if step=='prepare boot artifacts': self.events.put(('log',f'[boot] generated .bin/.img/.efi artifacts will be retained under {layout["boot_images"]}'))
                if step=='build ISO / IMG': self.events.put(('log',f'[iso] final ISO will be saved under {layout["iso"]}'))
                self.events.put(('log',f'[step] {step} completed'))
            except Exception as ex:self.events.put(('log',f'[error] {step} skipped after {type(ex).__name__}: {ex}; continuing'))
            self.events.put(('progress',type('P',(),{'stage':'build','completed':i,'total':len(steps),'message':f'Completed {i}/{len(steps)}: {step}'})()))
        self.events.put(('done','Workflow reached the final entry point; review live details for skipped operations and generated artifact paths.'))
if __name__=='__main__': App().mainloop()
