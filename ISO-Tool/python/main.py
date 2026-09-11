from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from iso_tool import BuildPipeline

class App(tk.Tk):
    def __init__(self):
        super().__init__(); self.title('ISO-Tool'); self.geometry('820x520')
        self.repo=tk.StringVar(); self.out=tk.StringVar(value='output.iso'); self.progress=tk.DoubleVar()
        ttk.Label(self,text='GitHub repository / local checkout').pack(anchor='w',padx=12,pady=(12,2))
        ttk.Entry(self,textvariable=self.repo).pack(fill='x',padx=12)
        ttk.Label(self,text='Output image').pack(anchor='w',padx=12,pady=(8,2))
        ttk.Entry(self,textvariable=self.out).pack(fill='x',padx=12)
        self.log=tk.Text(self,height=17); self.log.pack(fill='both',expand=True,padx=12,pady=12)
        ttk.Progressbar(self,variable=self.progress,maximum=100).pack(fill='x',padx=12)
        ttk.Button(self,text='Inventory / Build Plan',command=self.inventory).pack(pady=10)
    def inventory(self):
        p=Path(self.repo.get()).expanduser()
        if not p.is_dir(): messagebox.showerror('ISO-Tool','Choose a local checkout for this reference build.'); return
        files=BuildPipeline(p).inventory(); self.progress.set(100)
        self.log.delete('1.0','end'); self.log.insert('end',f'Found {len(files)} source files.\n')
        for f in files[:500]: self.log.insert('end',str(f.relative_to(p))+'\n')

if __name__=='__main__': App().mainloop()
