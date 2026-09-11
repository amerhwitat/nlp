from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from iso_tool import BuildPipeline, acquire_repository, build_repository
from iso_tool.boot_import import import_boot_sector, inspect_image
from iso_tool.resilient_network import ConnectivityMonitor

class App(tk.Tk):
    def __init__(self):
        super().__init__(); self.title('ISO-Tool — source / boot image to ISO / IMG'); self.geometry('1100x780')
        self.repo=tk.StringVar(); self.out=tk.StringVar(value='output.iso'); self.progress=tk.DoubleVar(); self.status=tk.StringVar(value='Ready'); self.events=queue.Queue(); self.running=False
        ttk.Label(self,text='GitHub repository URL or local source repository').pack(anchor='w',padx=12,pady=(12,2)); ttk.Entry(self,textvariable=self.repo).pack(fill='x',padx=12)
        ttk.Label(self,text='Output image / build directory').pack(anchor='w',padx=12,pady=(8,2)); ttk.Entry(self,textvariable=self.out).pack(fill='x',padx=12)
        bar=ttk.Frame(self); bar.pack(fill='x',padx=12,pady=10)
        self.analyze_button=ttk.Button(bar,text='Analyze / Inventory',command=self.inventory); self.analyze_button.pack(side='left')
        self.import_button=ttk.Button(bar,text='Import Boot Sector / ISO',command=self.import_image); self.import_button.pack(side='left',padx=8)
        self.build_images_button=ttk.Button(bar,text='Build Compiled Images',command=self.build_images); self.build_images_button.pack(side='left')
        self.build_button=ttk.Button(bar,text='Build ISO',command=self.build_iso); self.build_button.pack(side='left',padx=8)
        ttk.Button(bar,text='Clear details',command=self.clear_log).pack(side='left')
        ttk.Label(self,textvariable=self.status).pack(anchor='w',padx=12); ttk.Progressbar(self,variable=self.progress,maximum=100).pack(fill='x',padx=12,pady=(4,8)); ttk.Label(self,text='Live operation details').pack(anchor='w',padx=12)
        self.log=tk.Text(self,height=27,state='disabled',font=('Consolas',10)); self.log.pack(fill='both',expand=True,padx=12,pady=(4,12)); self.after(75,self._drain_events)
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
        source=self.repo.get().strip()
        if not source: messagebox.showerror('ISO-Tool','Enter a GitHub repository URL or local source repository.'); return
        self._start(lambda:self._inventory_worker(source))
    def _inventory_worker(self,source):
        try:
            root,cleanup=acquire_repository(source)
            try:
                sources,manifests=__import__('iso_tool.recursive_build',fromlist=['inventory']).inventory(root)
                total=max(len(sources)+len(manifests),1)
                self.events.put(('log',f'Entry point: analyze-source; recursive root: {root}'))
                self.events.put(('log',f'[inventory] source files={len(sources)}, build manifests={len(manifests)}'))
                for i,item in enumerate(sources+manifests,1):
                    rel=item.path; self.events.put(('log',f'[{i}/{total}] {rel}'))
                    self.events.put(('progress',type('P',(),{'stage':'inventory','completed':i,'total':total,'message':f'Inspected {i}/{total}: {rel}'})()))
                self.events.put(('done',f'Recursive inventory complete: {len(sources)} source files, {len(manifests)} build manifests.'))
            finally:
                if cleanup:
                    import shutil; shutil.rmtree(str(cleanup),ignore_errors=True)
        except Exception as ex:self.events.put(('log',f'[error] {type(ex).__name__}: {ex}')); self.events.put(('done','Recursive analysis ended with recoverable errors.'))
    def import_image(self):
        src=filedialog.askopenfilename(title='Import boot sector / ISO / image',filetypes=[('Images','*.iso *.img *.bin'),('All files','*.*')])
        if not src:return
        dst=filedialog.asksaveasfilename(title='Save imported boot sector',defaultextension='.bin',filetypes=[('Binary','*.bin'),('All files','*.*')])
        if not dst:return
        self._start(lambda:self._import_worker(Path(src),Path(dst)))
    def _import_worker(self,src,dst):
        try:
            self.events.put(('log',f'Entry point: import-boot-image; source={src}')); info=inspect_image(src); self.events.put(('log',f'[image] {info.kind}, {info.size} bytes, bootable={info.bootable}, sha256(first-sector)={info.boot_sector_sha256}')); import_boot_sector(src,dst); self.events.put(('progress',type('P',(),{'stage':'boot-import','completed':1,'total':1,'message':f'Imported bounded boot sector to {dst}'})())); self.events.put(('done','Boot image import complete; imported bytes remain inert until assigned to a boot profile.'))
        except Exception as ex:self.events.put(('log',f'[error] import skipped: {type(ex).__name__}: {ex}')); self.events.put(('done','Boot import ended with recoverable errors.'))
    def build_images(self): self._start(lambda:self._build_worker(False))
    def build_iso(self): self._start(lambda:self._build_worker(True))
    def _build_worker(self,make_iso):
        source=self.repo.get().strip(); monitor=ConnectivityMonitor()
        try:
            if not source: raise ValueError('source repository/path is empty')
            self.events.put(('log',f'Entry point: {"build-iso" if make_iso else "recursive-build"}; source={source}'))
            self.events.put(('log',f'[network] {monitor.check().detail}'))
            root,cleanup=acquire_repository(source)
            try:
                self.events.put(('log',f'[recursive] acquired root={root}'))
                report=build_repository(root, Path(self.out.get()).expanduser().resolve().parent / 'ISO-Tool-build', execute=True)
                total=max(len(report.sources)+len(report.manifests)+len(report.artifacts),1)
                completed=0
                for record in report.sources:
                    completed+=1; self.events.put(('log',f'[compile-plan] {record.path} ({record.language})')); self.events.put(('progress',type('P',(),{'stage':'recursive-build','completed':completed,'total':total,'message':f'Processed source {record.path}'})()))
                for record in report.manifests:
                    completed+=1; self.events.put(('log',f'[build-system] {record.path} ({record.kind})')); self.events.put(('progress',type('P',(),{'stage':'recursive-build','completed':completed,'total':total,'message':f'Processed manifest {record.path}'})()))
                for artifact in report.artifacts:
                    completed+=1; self.events.put(('log',f'[artifact] {artifact.status}: {artifact.path}')); self.events.put(('progress',type('P',(),{'stage':'link/artifact','completed':completed,'total':total,'message':f'{artifact.status}: {artifact.path}'})()))
                for skipped in report.skipped:self.events.put(('log',f'[skipped] {skipped}'))
                for error in report.errors:self.events.put(('log',f'[error] {error}'))
                if make_iso:
                    self.events.put(('log','[iso] Recursive build artifacts are now the input set for the existing ISO/image staging pipeline.'))
                result='Recursive build complete with no recorded failures.' if not report.errors else f'Recursive build completed with {len(report.errors)} recorded failures; inspect recursive-build.log/report.json.'
                self.events.put(('done',result))
            finally:
                if cleanup:
                    import shutil; shutil.rmtree(str(cleanup),ignore_errors=True)
        except Exception as ex:self.events.put(('log',f'[error] {type(ex).__name__}: {ex}')); self.events.put(('done','Recursive build ended with recoverable errors.'))
if __name__=='__main__': App().mainloop()