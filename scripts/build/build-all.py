#!/usr/bin/env python3
"""Portable build orchestrator; delegates platform-specific compilation to stable scripts."""
from __future__ import annotations
import argparse, os, platform, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

def run(label: str, command: list[str]) -> None:
    print(f"== {label} ==")
    subprocess.run(command, cwd=ROOT, check=True)

def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument('--configuration', default='Release', choices=['Debug','Release'])
    p.add_argument('--skip-database', action='store_true')
    args=p.parse_args()
    if (ROOT/'web/package.json').exists():
        npm = 'npm.cmd' if platform.system() == 'Windows' else 'npm'
        run('Web install/build', [npm, 'ci' if (ROOT/'web/package-lock.json').exists() else 'install'])
        run('Web compile', [npm, 'run', 'build'])
    if (ROOT/'python').exists(): run('Python validation', [sys.executable, '-m', 'compileall', '-q', 'python'])
    if (ROOT/'dotnet').exists(): run('.NET build', ['dotnet','build','dotnet','-c',args.configuration,'--nologo'])
    if not args.skip_database:
        db = ROOT/'scripts/database/init-database.py'
        if db.exists(): run('Database validation', [sys.executable, str(db), 'check'])
    print(f'Build-all completed: {args.configuration}')
    return 0
if __name__ == '__main__': raise SystemExit(main())
