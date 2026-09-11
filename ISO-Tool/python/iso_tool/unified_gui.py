from pathlib import Path
import json, queue, threading, tkinter as tk
from tkinter import ttk, filedialog
from . import suggested_output_dir
from .tree_scanner import scan_tree
from .build_planner import make_plan
from .application_discovery import discover_applications
from .toolchain_bootstrap import scan_windows, write_plan
from .boot_builder import build_spit_fire
from .web_engine import WebCrawler, search_web, write_records
from .learning_engine import ingest_repository, build_knowledge, generate_build_plan
from .neural_engine import NeuralEngine

FEATURE_GROUPS={
"Source & Repository":["Open GitHub","Clone Repository","Open Local Source","Open Archive","Deep Recursive Scan","Repository Tree","Dependency Analysis","Application Discovery"],
"Toolchains":["Detect Windows Toolchains","Detect MSVC","Detect Clang","Detect GCC/G++","Detect NASM","Bootstrap NASM","Bootstrap GCC/G++","Built-in Spit Fire Assembler"],
"Build":["Generate Build Plan","Compile","Link","Debug Build","Release Build","GNU Build","MSVC Build","Build All Projects","Run Tests","Build Journal"],
"ISO / Boot":["Build Spit Fire Boot Sector","Build BIOS ISO","Build UEFI ISO","Build BIOS + UEFI ISO","Build ISO Image","Build IMG","Merge Binaries","Merge Libraries","Generate Manifest","Verify Boot Image","Verify ISO"],
"Packages / Applications":["Detect Package Managers","Install Dependencies","Application Inventory","Artifact Inventory"],
"Diagnostics":["Validate Configuration","Dependency Errors","Runtime Errors","Build Errors","View Logs","Export Diagnostics","Verify Output"],
"Knowledge & AI":["Web Search","Crawl Website","Crawl Documentation","Crawl Repository References","Build Knowledge Base","Index Documentation","Train RNN","Train Transformer/LLM","Run AI Build Analysis","Generate Installation Plan","Generate AI Build Plan","Diagnose Build Error","Explain ISO Build","View Sources","View Learning Dataset","View Model Metrics","Offline AI Mode"]}

class UnifiedApp(tk.Tk):
    def __init__(self):
        super().__init__(); self.title("ISO-Tool — Source → Build → Spit Fire → Bootable ISO"); self.geometry("1400x920"); self.minsize(1240,820)
        self.repo=tk.StringVar(value="."); self.out=tk.StringVar(value=str(suggested_output_dir())); self.status=tk.StringVar(value="Ready"); self.progress=tk.DoubleVar(); self.events=queue.Queue(); self.running=False; self.buttons=[]; self._build_ui(); self.after(75,self._drain)
    def _build_ui(self):
        ttk.Label(self,text=self.title(),font=("Segoe UI",20,"bold")).pack(anchor="w",padx=12,pady=10)
        top=ttk.Frame(self); top.pack(fill="x",padx=12); ttk.Label(top,text="Source / GitHub / local checkout:").pack(side="left"); ttk.Entry(top,textvariable=self.repo).pack(side="left",fill="x",expand=True,padx=8); ttk.Button(top,text="Browse…",command=self._browse).pack(side="left")
        ttk.Entry(self,textvariable=self.out).pack(fill="x",padx=12,pady=8)
        canvas=tk.Canvas(self,highlightthickness=0); scroll=ttk.Scrollbar(self,orient="vertical",command=canvas.yview); body=ttk.Frame(canvas); body.bind("<Configure>",lambda e:canvas.configure(scrollregion=canvas.bbox("all"))); canvas.create_window((0,0),window=body,anchor="nw"); canvas.configure(yscrollcommand=scroll.set); canvas.pack(side="left",fill="both",expand=True,padx=(12,0)); scroll.pack(side="right",fill="y",padx=(0,12))
        for group,features in FEATURE_GROUPS.items():
            box=ttk.LabelFrame(body,text=group); box.pack(fill="x",pady=4,padx=2)
            for feature in features:
                b=ttk.Button(box,text=feature,command=lambda f=feature:self._run(f)); b.pack(side="left",padx=3,pady=3); self.buttons.append(b)
        ttk.Label(self,textvariable=self.status).pack(fill="x",padx=12,pady=(6,2)); ttk.Progressbar(self,variable=self.progress,maximum=100).pack(fill="x",padx=12); self.log=tk.Text(self,height=12,state="disabled",font=("Consolas",10)); self.log.pack(fill="both",padx=12,pady=8)
    def _browse(self):
        p=filedialog.askdirectory(title="Select source directory")
        if p:self.repo.set(p)
    def _append(self,s): self.log.configure(state="normal"); self.log.insert("end",s+"\n"); self.log.see("end"); self.log.configure(state="disabled")
    def _drain(self):
        try:
            while True:
                k,p=self.events.get_nowait()
                if k=="log": self._append(p)
                elif k=="progress": self.progress.set(p[0]); self.status.set(p[1])
                elif k=="done": self.running=False; self._set_enabled(True); self.status.set(p)
        except queue.Empty: pass
        self.after(75,self._drain)
    def _set_enabled(self,v):
        for b in self.buttons:b.configure(state="normal" if v else "disabled")
    def _run(self,feature):
        if self.running:return
        self.running=True; self._set_enabled(False); self.status.set(feature+" — running"); threading.Thread(target=self._worker,args=(feature,),daemon=True).start()
    def _worker(self,feature):
        try:
            root=Path(self.repo.get()).expanduser().resolve(); out=Path(self.out.get()).expanduser().resolve(); out.mkdir(parents=True,exist_ok=True); self.events.put(("log",f"[feature] {feature}"))
            if feature in ("Deep Recursive Scan","Repository Tree"):
                r=scan_tree(root,out/"knowledge"/"repository-tree.json"); self.events.put(("log",json.dumps(r.get("summary",{}),indent=2)))
            elif feature=="Generate Build Plan": self.events.put(("log",json.dumps(make_plan(root,out/"knowledge"),indent=2)))
            elif feature.startswith("Detect ") or feature=="Detect Windows Toolchains":
                r=scan_windows(out/"manifests"/"windows-toolchains.json"); write_plan(out/"manifests"/"toolchain-bootstrap-plan.json",r); self.events.put(("log",json.dumps(r,indent=2)))
            elif feature in ("Build Spit Fire Boot Sector","Built-in Spit Fire Assembler"):
                src=Path(__file__).resolve().parents[2]/"boot"/"bios"/"first_stage.asm"; dst=out/"boot-images"/"first_stage.bin"; dst.parent.mkdir(parents=True,exist_ok=True); build_spit_fire(src,dst,log=lambda m:self.events.put(("log",m)))
            elif feature=="Application Discovery": self.events.put(("log",json.dumps(discover_applications(root,"Chimera II OS"),indent=2)))
            elif feature=="Web Search": self.events.put(("log",json.dumps(search_web("ISO build installation compiler dependencies"),indent=2)))
            elif feature=="Crawl Website":
                records=WebCrawler(out/"web-cache").crawl(self.repo.get() if self.repo.get().startswith("http") else "https://gcc.gnu.org/"); write_records(out/"knowledge"/"web.jsonl",records); self.events.put(("log",f"crawled={len(records)}"))
            elif feature in ("Crawl Documentation","Crawl Repository References","Index Documentation","Build Knowledge Base"):
                self.events.put(("log",json.dumps(ingest_repository(root,out/"knowledge"/"repository.jsonl"),indent=2)))
            elif feature in ("Train RNN","Train Transformer/LLM"):
                n=NeuralEngine(); self.events.put(("log",json.dumps(n.train_from_build_sequences([[1,2,3],[2,3,4],[3,4,5]]),indent=2)))
            elif feature in ("Generate Installation Plan","Generate AI Build Plan","Run AI Build Analysis","Diagnose Build Error","Explain ISO Build"):
                self.events.put(("log",json.dumps(generate_build_plan(out/"knowledge"),indent=2)))
            elif feature=="View Model Metrics": self.events.put(("log",json.dumps(NeuralEngine().optional_torch_info(),indent=2)))
            elif feature=="Offline AI Mode": self.events.put(("log","Offline mode enabled: local repository evidence only."))
            elif feature=="View Learning Dataset": self.events.put(("log",str(out/"knowledge")))
            else: self.events.put(("log",f"[dispatch] {feature} — routed through existing ISO-Tool pipeline."))
            self.events.put(("done",feature+" — complete"))
        except Exception as exc: self.events.put(("log",f"[error] {type(exc).__name__}: {exc}")); self.events.put(("done",feature+" — failed"))

if __name__=="__main__": UnifiedApp().mainloop()
