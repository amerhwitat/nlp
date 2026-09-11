"""Portable orchestration primitives with fail-forward job isolation."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
import hashlib, os, subprocess
from .boot_validator import load_menu, select_with_fallback, emulator_available, qemu_bios_command

@dataclass
class BuildProgress: stage:str; completed:int; total:int; message:str
@dataclass
class JobResult: ok:bool; value:Any=None; error:str=""; index:int=0

class BuildPipeline:
    def __init__(self,workspace:Path,workers:int|None=None): self.workspace=Path(workspace); self.workers=workers or max(1,(os.cpu_count() or 2)-1)
    def inventory(self):
        suffixes={'.c','.cc','.cpp','.cxx','.h','.hpp','.asm','.s','.S','.cs'}
        return [p for p in self.workspace.rglob('*') if p.is_file() and p.suffix in suffixes]
    def sha256(self,path:Path)->str:
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
        return h.hexdigest()
    def run(self,argv,cwd=None,timeout=3600): return subprocess.run(list(argv),cwd=cwd or self.workspace,check=True,capture_output=True,text=True,timeout=timeout)
    def run_safe(self,argv,cwd=None,timeout=3600)->JobResult:
        try:return JobResult(True,self.run(argv,cwd,timeout))
        except Exception as exc:return JobResult(False,error=f'{type(exc).__name__}: {exc}')
    def validate_boot(self,menu_path:Path,staging:Path,preferred:str|None=None,firmware:str|None=None)->dict:
        menu=load_menu(menu_path); selected,attempts=select_with_fallback(menu,staging,preferred)
        result={'selected':selected,'attempts':[a.__dict__ for a in attempts],'emulator':'available' if emulator_available() else 'unavailable','evidence':'static'}
        if selected and emulator_available() and firmware=='bios': result['emulator_command']=qemu_bios_command(staging)
        elif selected and firmware=='uefi': result['emulator_command']='QEMU+OVMF required; configure OVMF firmware path'
        return result
    def parallel(self,jobs:Iterable[Callable[[],Any]],progress:Callable[[BuildProgress],None]|None=None,on_error:Callable[[Exception,int],None]|None=None)->list[JobResult]:
        job_list=list(jobs); total=len(job_list); results=[]
        if not total:
            if progress:progress(BuildProgress('build',0,0,'No jobs to execute'))
            return results
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            mapping={pool.submit(job):i for i,job in enumerate(job_list)}
            for completed,future in enumerate(as_completed(mapping),1):
                index=mapping[future]
                try: result=JobResult(True,future.result(),index=index)
                except Exception as exc:
                    result=JobResult(False,error=f'{type(exc).__name__}: {exc}',index=index)
                    if on_error:
                        try:on_error(exc,index)
                        except Exception:pass
                results.append(result)
                if progress:progress(BuildProgress('error' if not result.ok else 'build',completed,total,f'Job {index+1} '+('failed; continuing: '+result.error if not result.ok else 'completed')))
        return results
