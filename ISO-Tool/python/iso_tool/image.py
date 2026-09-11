from __future__ import annotations
from pathlib import Path
import shutil, subprocess

def find_backend():
    for n in ('xorriso','xorrisofs','oscdimg'):
        p=shutil.which(n)
        if p: return p
    return None

def create_iso(staging: Path, output: Path, label='ISO_TOOL'):
    backend=find_backend()
    if not backend: raise RuntimeError('No supported ISO backend found (xorriso/xorrisofs/oscdimg).')
    if Path(backend).name.lower() in ('xorriso','xorrisofs'):
        cmd=[backend,'-as','mkisofs','-iso-level','3','-V',label,'-o',str(output),str(staging)]
    else:
        cmd=[backend,'-l','-m','*','-o',str(output),str(staging)]
    return subprocess.run(cmd,check=True,capture_output=True,text=True,timeout=3600)
