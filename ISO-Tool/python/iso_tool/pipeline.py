"""Portable orchestration primitives. Repository commands require explicit trust."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib, os, shutil, subprocess

@dataclass
class BuildProgress:
    stage: str
    completed: int
    total: int
    message: str

class BuildPipeline:
    def __init__(self, workspace: Path, workers: int | None = None):
        self.workspace = Path(workspace)
        self.workers = workers or max(1, (os.cpu_count() or 2) - 1)

    def inventory(self):
        suffixes = {'.c', '.cc', '.cpp', '.cxx', '.h', '.hpp', '.asm', '.s', '.S', '.cs'}
        return [p for p in self.workspace.rglob('*') if p.is_file() and p.suffix in suffixes]

    def sha256(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
        return h.hexdigest()

    def run(self, argv, cwd=None, timeout=3600):
        # Structured argv; never pass untrusted repository text through a shell.
        return subprocess.run(list(argv), cwd=cwd or self.workspace, check=True,
                              capture_output=True, text=True, timeout=timeout)

    def parallel(self, jobs, progress=None):
        results = []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(job) for job in jobs]
            for i, future in enumerate(as_completed(futures), 1):
                results.append(future.result())
                if progress:
                    progress(BuildProgress('build', i, len(futures), f'Completed {i}/{len(futures)}'))
        return results
