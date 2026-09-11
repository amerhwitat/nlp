"""Windows toolchain discovery using PATH, environment variables and Visual Studio registry keys."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json, os, shutil, subprocess
from pathlib import Path
from typing import Dict, List

@dataclass(frozen=True)
class Toolchain:
    family: str
    role: str
    name: str
    path: str
    version: str
    source: str

def _version(exe: str) -> str:
    for args in (('--version',),('/?',),('-version',)):
        try:
            p=subprocess.run([exe,*args],capture_output=True,text=True,timeout=5,errors='replace')
            text=(p.stdout or '')+'\n'+(p.stderr or '')
            line=next((x.strip() for x in text.splitlines() if x.strip()),'')
            if line:return line[:240]
        except (OSError,subprocess.SubprocessError): pass
    return 'version unavailable'

def discover() -> List[Toolchain]:
    found: Dict[str,Toolchain]={}
    specs=[('MSVC','c++','cl.exe'),('MSVC','linker','link.exe'),('MASM','assembler','ml.exe'),('MASM','assembler','ml64.exe'),('LLVM','c++','clang++.exe'),('LLVM','c++','clang-cl.exe'),('LLVM','assembler','clang.exe'),('LLVM','linker','lld-link.exe'),('GNU/MinGW','c++','g++.exe'),('GNU/MinGW','c','gcc.exe'),('GNU/MinGW','linker','ld.exe'),('GNU/MinGW','assembler','as.exe'),('NASM','assembler','nasm.exe'),('YASM','assembler','yasm.exe'),('Build','builder','cmake.exe'),('Build','builder','msbuild.exe'),('Build','builder','make.exe'),('ISO','master','xorriso.exe'),('ISO','master','xorrisofs.exe'),('ISO','master','oscdimg.exe')]
    def add(family,role,name,path,source):
        p=Path(path).expanduser()
        if p.is_file():
            key=os.path.normcase(os.path.normpath(str(p)))
            if key not in found:found[key]=Toolchain(family,role,name,str(p),_version(str(p)),source)
    for family,role,name in specs:
        p=shutil.which(name) or shutil.which(name[:-4])
        if p:add(family,role,name,p,'PATH')
    for var in ('VCINSTALLDIR','VCToolsInstallDir','VSINSTALLDIR','LLVMInstallDir','NASM_PREFIX','MINGW_HOME','MINGW64_HOME','CODEBLOCKS'):
        value=os.environ.get(var,'')
        if not value:continue
        root=Path(value)
        for base in (root,root/'bin',root/'bin'/'Hostx64'/'x64'):
            for family,role,name in specs:add(family,role,name,str(base/name),'environment:'+var)
    if os.name=='nt':
        try:
            import winreg
            keys=[(winreg.HKEY_LOCAL_MACHINE,r'SOFTWARE\Microsoft\VisualStudio\SxS\VC7'),(winreg.HKEY_LOCAL_MACHINE,r'SOFTWARE\WOW6432Node\Microsoft\VisualStudio\SxS\VC7'),(winreg.HKEY_CURRENT_USER,r'SOFTWARE\Microsoft\VisualStudio\SxS\VC7')]
            for hive,keypath in keys:
                try:
                    with winreg.OpenKey(hive,keypath) as key:
                        for i in range(winreg.QueryInfoKey(key)[1]):
                            ver,value,_=winreg.EnumValue(key,i);base=Path(str(value))/'bin'/'Hostx64'/'x64'
                            for family,role,name in specs:
                                if family in ('MSVC','MASM'):add(family,role,name,str(base/name),'registry:'+keypath+':'+ver)
                except OSError:pass
        except ImportError:pass
    return sorted(found.values(),key=lambda x:(x.role,x.family,x.name,x.path.lower()))

def write_report(path:Path)->List[Toolchain]:
    tools=discover();path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps({'toolchains':[asdict(x) for x in tools]},indent=2),encoding='utf-8');return tools
