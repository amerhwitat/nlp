from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import shutil, subprocess

@dataclass(frozen=True)
class SourceSpec:
    url: str
    ref: str = ''
    submodules: bool = True

def clone(spec: SourceSpec, destination: Path) -> Path:
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    args=['git','clone']
    if spec.ref: args += ['--branch',spec.ref]
    if spec.submodules: args += ['--recurse-submodules']
    args += [spec.url,str(destination)]
    subprocess.run(args,check=True,timeout=3600)
    return destination

def archive_url(repository: str, ref: str='main') -> str:
    base=repository.rstrip('/')
    return f'{base}/archive/refs/heads/{ref}.zip' if ref else f'{base}/archive/refs/heads/main.zip'
