"""Portable orchestration primitives with fail-forward job isolation.

A failed source/build job is recorded and does not abort unrelated jobs. The
caller receives structured results and progress events so GUI front ends can
show what is happening in real time.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
import hashlib
import os
import subprocess


@dataclass
class BuildProgress:
    stage: str
    completed: int
    total: int
    message: str


@dataclass
class JobResult:
    ok: bool
    value: Any = None
    error: str = ""
    index: int = 0


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
        return subprocess.run(
            list(argv), cwd=cwd or self.workspace, check=True,
            capture_output=True, text=True, timeout=timeout
        )

    def run_safe(self, argv, cwd=None, timeout=3600) -> JobResult:
        """Run one external job without allowing a runtime failure to abort the pipeline."""
        try:
            completed = self.run(argv, cwd=cwd, timeout=timeout)
            return JobResult(True, completed)
        except (OSError, subprocess.SubprocessError, TimeoutError, RuntimeError) as exc:
            return JobResult(False, error=f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # defensive boundary for third-party/toolchain failures
            return JobResult(False, error=f"{type(exc).__name__}: {exc}")

    def parallel(
        self,
        jobs: Iterable[Callable[[], Any]],
        progress: Callable[[BuildProgress], None] | None = None,
        on_error: Callable[[Exception, int], None] | None = None,
    ) -> list[JobResult]:
        """Execute independent jobs concurrently and fail forward.

        Every job produces a JobResult. Exceptions are logged/reported and the
        remaining jobs continue. Completion is reported once per job and never
        moves backwards, making it suitable for a single GUI progress bar.
        """
        job_list = list(jobs)
        total = len(job_list)
        results: list[JobResult] = []
        if total == 0:
            if progress:
                progress(BuildProgress('build', 0, 0, 'No jobs to execute'))
            return results

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            future_to_index = {
                pool.submit(job): index for index, job in enumerate(job_list)
            }
            for completed, future in enumerate(as_completed(future_to_index), 1):
                index = future_to_index[future]
                try:
                    results.append(JobResult(True, future.result(), index=index))
                except Exception as exc:
                    result = JobResult(False, error=f"{type(exc).__name__}: {exc}", index=index)
                    results.append(result)
                    if on_error:
                        try:
                            on_error(exc, index)
                        except Exception:
                            # Error reporting must never become a new pipeline failure.
                            pass
                    if progress:
                        progress(BuildProgress('error', completed, total,
                                                f'Job {index + 1} failed; continuing: {exc}'))
                        continue
                if progress:
                    message = f'Completed {completed}/{total}'
                    if results[-1].ok:
                        message += f' (job {index + 1})'
                    progress(BuildProgress('build', completed, total, message))
        return results
