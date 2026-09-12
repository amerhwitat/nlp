#!/usr/bin/env python3
"""Idempotent dependency bootstrap helper for CI and local development."""
import argparse, os, shutil, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def need(name):
    if not shutil.which(name): raise SystemExit(f"Missing required command: {name}")
def run(cmd, cwd=ROOT): print('+',' '.join(map(str,cmd))); subprocess.run(cmd,cwd=cwd,check=True)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('profile',nargs='?',default='developer'); args=ap.parse_args(); os.chdir(ROOT)
    for tool in ('git','node','npm', 'python' if os.name=='nt' else 'python3'): need(tool)
    venv=ROOT/'.venv'; py=venv/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not venv.exists(): run([sys.executable,'-m','venv',str(venv)])
    req=ROOT/'python/requirements.txt'
    if req.exists(): run([str(py),'-m','pip','install','--upgrade','pip']); run([str(py),'-m','pip','install','-r',str(req)])
    web=ROOT/'web';
    if (web/'package.json').exists(): run(['npm.cmd' if os.name=='nt' else 'npm','ci' if (web/'package-lock.json').exists() else 'install'],web)
    print('Bootstrap completed:',args.profile)
if __name__=='__main__': main()
