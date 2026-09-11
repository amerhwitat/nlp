from pathlib import Path
import queue, threading, tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from iso_tool import BuildPipeline, dependency_cache_dir, suggested_output_dir, prepare_output_layout
from iso_tool.boot_import import import_boot_sector, inspect_image
from iso_tool.github_source import is_github_reference, normalize_github_repository, prepare_source
from iso_tool.build_entrypoint import build

class App(tk.Tk):
    def __init__(self):
        super().__init__(); self.title('ISO-Tool — GitHub repository → compile/link → ISO/IMG'); self.geometry('1120x840')
        self.repo=tk.StringVar(value='https://github.com/amerhwitat/ChimeraIIOS'); self.out=tk.StringVar(value=str(suggested_output_dir())); self.progress=tk.DoubleVar(); self.status=tk.StringVar(value='Ready'); self.events=queue.Queue(); self.running=False
        ttk.Label(self,text='GitHub repository or local source repository').pack(anchor='w',padx=12,pady=(12,2)); row=ttk.Frame(self); row.pack(fill='x',padx=12); ttk.Entry(row,textvariable=self.repo).pack(side='left',fill='x',expand=True); ttk.Button(row,text='Select GitHub repo…',command=self.select_repo).pack(side='left',padx=(8,0)); ttk.Button(row,text='Browse local…',command=self.select_local).pack(side='left',padx=(8,0))
        outrow=ttk.Frame(self); outrow.pack(fill='x',padx=12,pady=(8,2)); ttk.Label(outrow,text='Final output directory').pack(side='left'); ttk.Button(outrow,text='Choose…',command=self.choose_output).pack(side='right'); ttk.Entry(self,textvariable=self.out).pack(fill='x',padx=12); ttk.Label(self,text=f'Dependency cache: {dependency_cache_dir()}').pack(anchor='w',padx=12,pady=(3,2))
        bar=ttk.Frame(self); bar.pack(fill='x',padx=12,pady=10); self.analyze_button=ttk.Button(bar,text='Analyze / Inventory',command=self.inventory); self.analyze_button.pack(side='left'); self.import_button=ttk.Button(bar,text='Import Boot Sector / ISO',command=self.import_image); self.import_button.pack(side='left',padx=8); self.build_images_button=ttk.Button(bar,text='Compile + Link (GNU + MSVC)',command=self.build_images); self.build_images_button.pack(side='left'); self.build_button=ttk.Button(bar,text='Compile + Link + Build ISO',command=self.build_iso); self.build_button.pack(side='left',padx=8); ttk.Button(bar,text='Clear details',command=self.clear_log).pack(side='left')
        ttk.Label(self,textvariable=self.status).pack(anchor='w',padx=12); ttk.Progressbar(self,variable=self.progress,maximum=100).pack(fill='x',padx=12,pady=(4,8)); ttk.Label(self,text='Live operation details').pack(anchor='w',padx=12); self.log=tk.Text(self,height=30,state='disabled',font=('Consolas',10)); self.log.pack(fill='both',expand=True,padx=12,pady=(4,12)); self.after(75,self._drain_events)
    def select_repo(self):
        value=simpledialog.askstring('Select GitHub repository','Enter GitHub URL or owner/repository:',initialvalue=self.repo.get())
        if value:
            try:self.repo.set('https://github.com/'+normalize_github_repository(value)); self._append(f'[source] selected GitHub repository: {self.repo.get()}')
            except ValueError as ex: messagebox.showerror('ISO-Tool',str(ex))
    def select_local(self):
        p=filedialog.askdirectory(title='Select local Git repository/source tree')
        if p:self.repo.set(p); self._append(f'[source] selected local repository: {p}')
    def choose_output(self):
        p=filedialog.askdirectory(title='Choose final ISO/build output directory',initialdir=self.out.get() or str(suggested_output_dir()),mustexist=False)
        if p:self.out.set(p); self._append(f'[output] user selected: {p}')
    def clear_log(self): self.log.configure(state='normal'); self.log.delete('1.0','end'); self.log.configure(state='disabled')
    def _append(self,msg): self.log.configure(state='normal'); self.log.insert('end',msg+'\n'); self.log.see('end'); self.log.configure(state='disabled')
    def _drain_events(self):
        try:
            while True:
                kind,payload=self.events.get_nowait()
                if kind=='progress': self.progress.set(payload[0]); self.status.set(payload[1]); self._append(payload[1])
                elif kind=='log': self._append(payload)
                elif kind=='done': self.running=False; self._set_buttons(True); self.status.set(payload); self._append(payload)
        except queue.Empty: pass
        self.after(75,self._drain_events)
    def _set_buttons(self,on):
        for b in (self.analyze_button,self.import_button,self.build_images_button,self.build_button): b.configure(state='normal' if on else 'disabled')
    def _start(self,target):
        if self.running:return
        self.running=True; self._set_buttons(False); self.progress.set(0); threading.Thread(target=target,daemon=True).start()
    def _source(self):
        value=self.repo.get().strip()
        if not value: raise ValueError('Select a GitHub repository or local source repository first.')
        if is_github_reference(value):
            checkout=Path(self.out.get()).expanduser()/'sources'/normalize_github_repository(value).replace('/','__')
            self.events.put(('log',f'[github] cloning selected source to {checkout}'))
            return prepare_source(value,checkout)
        p=Path(value).expanduser()
        if not p.is_dir(): raise FileNotFoundError(f'Source repository does not exist: {p}')
        return p.resolve()
    def inventory(self): self._start(lambda:self._inventory_worker())
    def _inventory_worker(self):
        try:
            root=self._source(); files=BuildPipeline(root).inventory(); self.events.put(('log',f'Entry point: analyze-source; {root}'))
            for i,p in enumerate(files,1): self.events.put(('log',f'[{i}/{len(files)}] {p.relative_to(root)}'))
            self.events.put(('done',f'Inventory complete: {len(files)} source files.'))
        except Exception as ex:self.events.put(('log',f'[error] {type(ex).__name__}: {ex}')); self.events.put(('done','Inventory ended with recoverable errors.'))
    def import_image(self):
        src=filedialog.askopenfilename(title='Import boot sector / ISO / image',filetypes=[('Images','*.iso *.img *.bin'),('All files','*.*')])
        if not src:return
        dst=filedialog.asksaveasfilename(title='Save imported boot sector',initialdir=str(Path(self.out.get())/'boot-images'),defaultextension='.bin',filetypes=[('Binary','*.bin'),('All files','*.*')])
        if dst:self._start(lambda:self._import_worker(Path(src),Path(dst)))
    def _import_worker(self,src,dst):
        try: info=inspect_image(src); self.events.put(('log',f'[image] {info.kind}, {info.size} bytes, bootable={info.bootable}')); import_boot_sector(src,dst); self.events.put(('done',f'Boot image imported: {dst}'))
        except Exception as ex:self.events.put(('log',f'[error] {type(ex).__name__}: {ex}')); self.events.put(('done','Boot import ended with recoverable errors.'))
    def build_images(self): self._start(lambda:self._build_worker(False))
    def build_iso(self): self._start(lambda:self._build_worker(True))
    def _build_worker(self,make_iso):
        try:
            layout=prepare_output_layout(Path(self.out.get()).expanduser()); self.events.put(('log',f'[output] {layout["root"]}')); source=self._source(); self.events.put(('log',f'[source] {source}'))
            compilers=[('gnu','GNU C++'),('msvc','Microsoft Visual C++')]; results=[]
            for idx,(key,label) in enumerate(compilers,1):
                try:
                    self.events.put(('log',f'[{label}] configure / compile / link started'))
                    result=build(source,layout['root'],key,log=lambda m:self.events.put(('log',f'[{label}] {m}'))); results.append(result); self.events.put(('progress',(idx*35,f'{label} build completed')))
                except Exception as ex: self.events.put(('log',f'[{label}] unavailable or failed: {type(ex).__name__}: {ex}'))
            if not results: raise RuntimeError('Neither GNU C++ nor MSVC produced a build.')
            self.events.put(('log',f'[artifacts] compiled and linked files staged under {layout["executables"]} and {layout["libraries"]}'))
            if make_iso: self.events.put(('log',f'[iso] ISO mastering stage requested; compiled artifacts are ready at {layout["root"]}.'))
            self.events.put(('done','Compile/link entry point completed; review artifact paths above.'))
        except Exception as ex:self.events.put(('log',f'[fatal] {type(ex).__name__}: {ex}')); self.events.put(('done','Build stopped with a recoverable error.'))
if __name__=='__main__': App().mainloop()
