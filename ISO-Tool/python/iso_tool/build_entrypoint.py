"""Single build entry point: inspect documents, plan precedence, compile/link, stage artifacts."""
from __future__ import annotations
import argparse,json,shutil,subprocess
from pathlib import Path
from .output_paths import prepare_output_layout
from .build_planner import make_plan
from .ai_engine import propose_refinement

def _run(cmd:list[str],cwd:Path,log):
    log('[exec] '+' '.join(str(x) for x in cmd)); p=subprocess.run(cmd,cwd=cwd,text=True,capture_output=True,timeout=7200)
    if p.stdout: log(p.stdout.rstrip())
    if p.stderr: log(p.stderr.rstrip())
    if p.returncode: raise RuntimeError(f'command failed ({p.returncode}): {cmd[0]}')
    return p

def _stage_tree(build:Path,layout:dict,log):
    staged=[]
    for p in build.rglob('*'):
        if not p.is_file(): continue
        s=p.suffix.lower()
        if s in {'.exe','.com'}: target=layout['executables']/p.name
        elif s in {'.dll','.so','.dylib','.lib','.a'}: target=layout['libraries']/p.name
        elif s in {'.bin','.img','.efi'}: target=layout['boot_images']/p.name
        else: continue
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,target); staged.append(str(target)); log(f'[artifact] {target}')
    return staged

def build(source:Path,output:Path,compiler:str='auto',log=print)->dict:
    source=source.resolve(); output=output.resolve(); layout=prepare_output_layout(output)
    knowledge=output/'knowledge'; plan=make_plan(source,knowledge); log(f'[intelligence] scanned {len(plan["steps"])} ordered build stages')
    ai=propose_refinement(plan, prompt_context='Keep the deterministic precedence unchanged unless a registered dependency proves otherwise.')
    (knowledge/'ai-plan.json').write_text(json.dumps(ai,indent=2),encoding='utf-8')
    build_root=output/'build'/compiler.lower().replace('+','p').replace(' ','-'); build_root.mkdir(parents=True,exist_ok=True)
    cmake=shutil.which('cmake')
    if not cmake: raise RuntimeError('CMake was not found.')
    if not (source/'CMakeLists.txt').exists(): raise RuntimeError(f'No CMakeLists.txt found at repository root: {source}')
    if compiler in ('gnu','gcc'):
        generator='Ninja' if shutil.which('ninja') else 'MinGW Makefiles'; _run([cmake,'-S',str(source),'-B',str(build_root),'-G',generator,'-DCMAKE_BUILD_TYPE=Release'],source,log)
    elif compiler in ('msvc','cl'):
        _run([cmake,'-S',str(source),'-B',str(build_root),'-G','Visual Studio 17 2022','-A','x64'],source,log)
    else: _run([cmake,'-S',str(source),'-B',str(build_root)],source,log)
    _run([cmake,'--build',str(build_root),'--config','Release','--parallel'],source,log)
    artifacts=_stage_tree(build_root,layout,log)
    manifest=layout['manifests']/f'build-result-{compiler}.json'; manifest.write_text(json.dumps({'source':str(source),'output':str(output),'compiler':compiler,'build_tree':str(build_root),'artifacts':artifacts,'plan':str(knowledge/'build-plan.json'),'ai_plan':str(knowledge/'ai-plan.json')},indent=2),encoding='utf-8'); log(f'[manifest] {manifest}')
    return {'layout':layout,'build_tree':build_root,'artifacts':artifacts,'manifest':manifest}

def main(argv=None):
    ap=argparse.ArgumentParser(description='ISO-Tool document-aware compile/link entry point'); ap.add_argument('source'); ap.add_argument('--output',required=True); ap.add_argument('--compiler',choices=['auto','gnu','msvc'],default='auto'); a=ap.parse_args(argv); r=build(Path(a.source),Path(a.output),a.compiler); print(json.dumps({k:str(v) for k,v in r.items() if k!='layout'},indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
