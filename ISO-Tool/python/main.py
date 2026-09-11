from pathlib import Path
import queue,threading,shutil,tkinter as tk
from tkinter import filedialog,messagebox,simpledialog,ttk
from iso_tool import BuildPipeline,dependency_cache_dir,suggested_output_dir,prepare_output_layout
from iso_tool.boot_import import import_boot_sector,inspect_image
from iso_tool.github_source import is_github_reference,normalize_github_repository,prepare_source
from iso_tool.build_entrypoint import build
from iso_tool.build_planner import make_plan
from iso_tool.iso_merger import merge_staging
from iso_tool.image import create_iso
class App(tk.Tk):
 def __init__(self):
  super().__init__();self.title('ISO-Tool — document-aware GitHub → compile/link → ISO');self.geometry('1160x860');self.repo=tk.StringVar(value='https://github.com/amerhwitat/ChimeraIIOS');self.out=tk.StringVar(value=str(suggested_output_dir()));self.progress=tk.DoubleVar();self.status=tk.StringVar(value='Ready');self.events=queue.Queue();self.running=False
  ttk.Label(self,text='GitHub repository or local source repository').pack(anchor='w',padx=12,pady=(12,2));r=ttk.Frame(self);r.pack(fill='x',padx=12);ttk.Entry(r,textvariable=self.repo).pack(side='left',fill='x',expand=True);ttk.Button(r,text='Select GitHub repo…',command=self.select_repo).pack(side='left',padx=8);ttk.Button(r,text='Browse local…',command=self.select_local).pack(side='left')
  ttk.Label(self,text='Final output directory').pack(anchor='w',padx=12,pady=(8,2));ttk.Entry(self,textvariable=self.out).pack(fill='x',padx=12);ttk.Button(self,text='Choose output…',command=self.choose_output).pack(anchor='e',padx=12);ttk.Label(self,text=f'Dependency cache: {dependency_cache_dir()}').pack(anchor='w',padx=12)
  b=ttk.Frame(self);b.pack(fill='x',padx=12,pady=10);self.analyze_button=ttk.Button(b,text='Analyze Documents + Plan',command=self.inventory);self.analyze_button.pack(side='left');self.import_button=ttk.Button(b,text='Import Boot/ISO',command=self.import_image);self.import_button.pack(side='left',padx=8);self.build_images_button=ttk.Button(b,text='Compile + Link GNU + MSVC',command=self.build_images);self.build_images_button.pack(side='left');self.build_button=ttk.Button(b,text='AI Plan + Build + ISO',command=self.build_iso);self.build_button.pack(side='left',padx=8);ttk.Button(b,text='Clear',command=self.clear_log).pack(side='left')
  ttk.Label(self,textvariable=self.status).pack(anchor='w',padx=12);ttk.Progressbar(self,variable=self.progress,maximum=100).pack(fill='x',padx=12,pady=8);self.log=tk.Text(self,height=31,state='disabled',font=('Consolas',10));self.log.pack(fill='both',expand=True,padx=12,pady=8);self.after(75,self._drain_events)
 def select_repo(self):
  v=simpledialog.askstring('Select GitHub repository','Enter GitHub URL or owner/repository:',initialvalue=self.repo.get())
  if v:
   try:self.repo.set('https://github.com/'+normalize_github_repository(v));self._append('[source] '+self.repo.get())
   except ValueError as e:messagebox.showerror('ISO-Tool',str(e))
 def select_local(self):
  p=filedialog.askdirectory(title='Select local repository');
  if p:self.repo.set(p);self._append('[source] '+p)
 def choose_output(self):
  p=filedialog.askdirectory(title='Choose output directory',initialdir=self.out.get());
  if p:self.out.set(p)
 def clear_log(self):self.log.configure(state='normal');self.log.delete('1.0','end');self.log.configure(state='disabled')
 def _append(self,m):self.log.configure(state='normal');self.log.insert('end',m+'\n');self.log.see('end');self.log.configure(state='disabled')
 def _drain_events(self):
  try:
   while True:
    k,p=self.events.get_nowait()
    if k=='progress':self.progress.set(p[0]);self.status.set(p[1]);self._append(p[1])
    elif k=='log':self._append(p)
    elif k=='done':self.running=False;self._set_buttons(True);self.status.set(p);self._append(p)
  except queue.Empty:pass
  self.after(75,self._drain_events)
 def _set_buttons(self,on):
  for b in (self.analyze_button,self.import_button,self.build_images_button,self.build_button):b.configure(state='normal' if on else 'disabled')
 def _start(self,target):
  if self.running:return
  self.running=True;self._set_buttons(False);self.progress.set(0);threading.Thread(target=target,daemon=True).start()
 def _source(self):
  v=self.repo.get().strip()
  if is_github_reference(v):
   p=Path(self.out.get())/'sources'/normalize_github_repository(v).replace('/','__');self.events.put(('log',f'[github] checkout: {p}'));return prepare_source(v,p)
  p=Path(v).expanduser()
  if not p.is_dir():raise FileNotFoundError(str(p))
  return p.resolve()
 def inventory(self):self._start(self._inventory_worker)
 def _inventory_worker(self):
  try:
   root=self._source();plan=make_plan(root,Path(self.out.get())/'knowledge');self.events.put(('log',f'[documents] scanned repository: {root}'));self.events.put(('log',f'[plan] ordered stages: '+', '.join(x['id'] for x in plan['steps'])));self.events.put(('done','Document analysis and build plan complete.'))
  except Exception as e:self.events.put(('log',f'[error] {type(e).__name__}: {e}'));self.events.put(('done','Analysis stopped.'))
 def import_image(self):
  s=filedialog.askopenfilename(title='Import boot/ISO image',filetypes=[('Images','*.iso *.img *.bin'),('All','*.*')]);
  if not s:return
  d=filedialog.asksaveasfilename(title='Save boot image',initialdir=str(Path(self.out.get())/'boot-images'),defaultextension='.bin');
  if d:self._start(lambda:self._import_worker(Path(s),Path(d)))
 def _import_worker(self,s,d):
  try:i=inspect_image(s);self.events.put(('log',f'[image] {i.kind} {i.size} bytes bootable={i.bootable}'));import_boot_sector(s,d);self.events.put(('done',f'Imported: {d}'))
  except Exception as e:self.events.put(('log',f'[error] {e}'));self.events.put(('done','Import stopped.'))
 def build_images(self):self._start(lambda:self._build_worker(False))
 def build_iso(self):self._start(lambda:self._build_worker(True))
 def _build_worker(self,make_iso):
  try:
   layout=prepare_output_layout(Path(self.out.get()));root=self._source();self.events.put(('log',f'[source] {root}'));results=[]
   for n,(key,label) in enumerate([('gnu','GNU C++'),('msvc','MSVC')],1):
    try:self.events.put(('log',f'[{label}] build'));results.append(build(root,Path(layout['root']),key,log=lambda m,l=label:self.events.put(('log',f'[{l}] {m}'))))
    except Exception as e:self.events.put(('log',f'[{label}] failed/unavailable: {e}'))
    self.events.put(('progress',(n*35,f'{label} stage finished')))
   if not results:raise RuntimeError('No compiler produced artifacts.')
   if make_iso:
    staging=Path(layout['root'])/'staging';manifest=Path(layout['manifests'])/'staging-manifest.json';merge_staging(root,staging,manifest);iso=Path(layout['iso'])/f'Chimera-II-{root.name}.iso';create_iso(staging,iso,label='CHIMERA_II',profile='data');self.events.put(('log',f'[iso] {iso}'))
   self.events.put(('done','Document-aware compile/link pipeline completed; inspect manifests and artifacts.'))
  except Exception as e:self.events.put(('log',f'[fatal] {type(e).__name__}: {e}'));self.events.put(('done','Build stopped with an error.'))
if __name__=='__main__':App().mainloop()
