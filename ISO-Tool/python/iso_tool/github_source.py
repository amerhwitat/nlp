from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import subprocess
from .resilient_network import ConnectivityMonitor, retry_when_online

@dataclass(frozen=True)
class SourceSpec:
    url: str
    ref: str = ''
    submodules: bool = True
    retry_interval: float = 15.0
    retry_forever: bool = True

def clone(spec: SourceSpec, destination: Path, log=None, stop=None) -> Path:
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    args=['git','clone']
    if spec.ref: args += ['--branch',spec.ref]
    if spec.submodules: args += ['--recurse-submodules']
    args += [spec.url,str(destination)]
    monitor=ConnectivityMonitor(interval=spec.retry_interval)
    def operation():
        if destination.exists() and any(destination.iterdir()):
            raise RuntimeError(f'destination is not empty: {destination}')
        subprocess.run(args,check=True,timeout=3600)
        return destination
    return retry_when_online(operation,monitor,log=log,stop=stop,max_attempts=None if spec.retry_forever else 3)

def archive_url(repository: str, ref: str='main') -> str:
    base=repository.rstrip('/')
    return f'{base}/archive/refs/heads/{ref}.zip' if ref else f'{base}/archive/refs/heads/main.zip'
