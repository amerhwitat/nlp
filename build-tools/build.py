#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, platform, shlex, shutil, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'build'/'artifacts'
def log(stage,msg): print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] [{stage.upper():10}] {msg}',flush=True)
def which(name): return shutil.which(name)
def run(cmd,dry=False,cwd=ROOT):
 log('command',' '.join(shlex.quote(str(x)) for x in cmd))
 if dry:return 0
 t=time.monotonic(); p=subprocess.Popen([str(x) for x in cmd],cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 assert p.stdout
 for line in p.stdout: print(line.rstrip(),flush=True)
 rc=p.wait();log('result',f'exit={rc} elapsed={time.monotonic()-t:.2f}s');return rc
def detect():
 return {'python':bool(list(ROOT.rglob('*.py'))),'java':bool(list(ROOT.rglob('*.java')) or (ROOT/'pom.xml').exists() or (ROOT/'build.gradle').exists()),'node':(ROOT/'package.json').exists(),'native':bool((ROOT/'CMakeLists.txt').exists() or (ROOT/'Makefile').exists() or list(ROOT.glob('*.sln'))),'sql':bool(list(ROOT.rglob('*.sql'))),'web':(ROOT/'package.json').exists()}
def python_stage(a):
 if not which('python'):log('skip','Python 3 missing');return 0
 req=next((p for p in [ROOT/'requirements.txt',ROOT/'requirements-dev.txt'] if p.exists()),None)
 if req and run([sys.executable,'-m','pip','install','-r',str(req)],a.dry_run):return 1
 src=[p for p in ROOT.rglob('*.py') if p.name not in {'__init__.py','setup.py'} and not any(x in p.parts for x in {'.git','build','dist','.venv','venv','__pycache__'})]
 if a.python:src=[ROOT/a.python]
 if not src:log('skip','No Python entry point');return 0
 p=src[0]; cmd=[sys.executable,'-m','PyInstaller','--noconfirm','--clean','--distpath',str(ART/'python'),'--workpath',str(ROOT/'build'/'pyinstaller')]
 if a.onefile:cmd+=['--onefile']
 return run(cmd+[str(p)],a.dry_run)
def java_stage(a):
 if (ROOT/'pom.xml').exists() and which('mvn'):return run(['mvn','-B','test','package'],a.dry_run)
 if (ROOT/'gradlew').exists():return run([str(ROOT/'gradlew'),'build'],a.dry_run)
 if (ROOT/'build.gradle').exists() and which('gradle'):return run(['gradle','build'],a.dry_run)
 src=[p for p in ROOT.rglob('*.java') if '.git' not in p.parts]
 if src and which('javac'):
  out=ROOT/'build'/'java-classes';out.mkdir(parents=True,exist_ok=True);return run(['javac','-d',str(out),*map(str,src)],a.dry_run)
 log('skip','No Java build target');return 0
def node_stage(a):
 if not (ROOT/'package.json').exists():return 0
 pm='pnpm' if (ROOT/'pnpm-lock.yaml').exists() and which('pnpm') else 'yarn' if (ROOT/'yarn.lock').exists() and which('yarn') else 'npm'
 if not which(pm):log('skip',f'{pm} missing');return 0
 install=['npm','ci'] if pm=='npm' and (ROOT/'package-lock.json').exists() else [pm,'install']
 if run(install,a.dry_run):return 1
 return run([pm,'run','build'],a.dry_run)
def native_stage(a):
 if (ROOT/'CMakeLists.txt').exists() and which('cmake'):
  b=ROOT/'build'/'cmake';b.mkdir(parents=True,exist_ok=True)
  if run(['cmake','-S',ROOT,'-B',b,'-DCMAKE_BUILD_TYPE=Release'],a.dry_run):return 1
  if run(['cmake','--build',b,'--config','Release','--parallel'],a.dry_run):return 1
  if which('ctest'):return run(['ctest','--test-dir',b,'--output-on-failure'],a.dry_run)
  return 0
 if (ROOT/'Makefile').exists() and which('make'):return run(['make','-j'],a.dry_run)
 if list(ROOT.glob('*.sln')) and which('dotnet'):return run(['dotnet','build','--configuration','Release'],a.dry_run)
 return 0
def sql_stage(a):
 ss=sorted(p for p in ROOT.rglob('*.sql') if '.git' not in p.parts and 'build' not in p.parts);log('database',f'{len(ss)} SQL scripts discovered')
 for p in ss:log('database',str(p.relative_to(ROOT)))
 log('database','Use NLP_DB_* environment variables and native clients for execution; no credentials are stored in source control');return 0
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dry-run',action='store_true');ap.add_argument('--only',choices=['all','python','java','node','native','sql'],default='all');ap.add_argument('--python');ap.add_argument('--onefile',action='store_true');a=ap.parse_args();ART.mkdir(parents=True,exist_ok=True);log('build',f'repo={ROOT.name} os={platform.system()} arch={platform.machine()}');c=detect();log('detect',json.dumps(c,sort_keys=True));t=time.monotonic();f={'python':python_stage,'java':java_stage,'node':node_stage,'native':native_stage,'sql':sql_stage}
 for n in ([a.only] if a.only!='all' else ['python','java','node','native','sql']):
  if not c.get(n,True):log('skip',f'{n}: not detected');continue
  log('stage',n)
  if f[n](a):log('build',f'FAILED stage={n}');return 1
 log('build',f'DONE elapsed={time.monotonic()-t:.2f}s artifacts={ART}');return 0
if __name__=='__main__':raise SystemExit(main())
