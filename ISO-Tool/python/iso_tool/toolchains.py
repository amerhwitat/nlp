from __future__ import annotations
from dataclasses import dataclass
import os, shutil, subprocess

@dataclass(frozen=True)
class Toolchain:
    name: str
    path: str
    version: str = ''

def _version(exe: str, args=('--version',)) -> str:
    try:
        p=subprocess.run([exe,*args],capture_output=True,text=True,timeout=5)
        return (p.stdout or p.stderr).splitlines()[0][:240] if p.returncode==0 else ''
    except (OSError, subprocess.SubprocessError): return ''

def discover() -> list[Toolchain]:
    names=['gcc','g++','clang','clang++','nasm','ml','ml64','cl','msbuild','cmake','make','dotnet','xorriso','xorrisofs','oscdimg']
    out=[]
    for name in names:
        p=shutil.which(name)
        if p: out.append(Toolchain(name,p,_version(p)))
    return out
