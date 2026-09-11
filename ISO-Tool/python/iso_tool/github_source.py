from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
from .resilient_network import ConnectivityMonitor, retry_when_online

_GITHUB_RE = re.compile(r"^(?:https?://github\.com/|git@github\.com:)([^/ :]+/[^/]+?)(?:\.git)?/?$")

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

def normalize_github_repository(value: str) -> str:
    value=value.strip()
    match=_GITHUB_RE.match(value)
    if not match:
        raise ValueError('Enter a GitHub URL, owner/repository, or a local source path.')
    return match.group(1).removesuffix('.git')

def is_github_reference(value: str) -> bool:
    try: normalize_github_repository(value); return True
    except ValueError: return False

def repository_url(value: str) -> str:
    return 'https://github.com/'+normalize_github_repository(value)+'.git'

def prepare_source(value: str, destination: Path, branch: str|None=None) -> Path:
    raw=value.strip(); local=Path(raw).expanduser()
    if local.is_dir(): return local.resolve()
    if not is_github_reference(raw): raise FileNotFoundError(f'Local repository does not exist: {local}')
    if destination.exists() and any(destination.iterdir()): raise FileExistsError(f'GitHub checkout destination is not empty: {destination}')
    destination.parent.mkdir(parents=True,exist_ok=True)
    git=shutil.which('git')
    if not git: raise RuntimeError('Git was not found. Install Git before cloning a GitHub source.')
    spec=SourceSpec(repository_url(raw),branch or '',True,retry_forever=False)
    return clone(spec,destination)

def repository_display_name(value: str) -> str:
    return normalize_github_repository(value) if is_github_reference(value) else Path(value).expanduser().name or value
