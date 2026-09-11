from __future__ import annotations
from pathlib import Path
import hashlib, json, re, time

DOC_EXT={'.md','.markdown','.txt','.rst','.adoc','.json','.yaml','.yml','.toml','.xml','.cmake','.py','.c','.cc','.cpp','.cxx','.h','.hpp','.cs','.java','.js','.ts','.sh','.bat','.ps1','.s','.asm'}
BUILD_NAMES={'CMakeLists.txt','Makefile','pom.xml','package.json','Cargo.toml','pyproject.toml','requirements.txt','setup.py','Directory.Build.props','*.sln'}

def ingest_repository(root, output):
    root=Path(root).resolve(); output=Path(output); output.parent.mkdir(parents=True,exist_ok=True); rows=[]
    for p in root.rglob('*'):
        if not p.is_file() or any(x in {'.git','node_modules','bin','obj','__pycache__'} for x in p.parts): continue
        if p.suffix.lower() not in DOC_EXT and p.name not in BUILD_NAMES: continue
        try: raw=p.read_bytes(); text=raw.decode('utf-8','replace')
        except Exception: continue
        rows.append({'source':str(p.relative_to(root)),'source_type':'repository','retrieved_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'sha256':hashlib.sha256(raw).hexdigest(),'text':text[:500000]})
    with output.open('w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    return {'records':len(rows),'output':str(output)}

def build_knowledge(repo_root, web_records, output_dir):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    local=ingest_repository(repo_root,output_dir/'repository.jsonl')
    web_path=output_dir/'web.jsonl'; web_path.write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in web_records),encoding='utf-8')
    return {'repository':local,'web_records':len(web_records),'knowledge_dir':str(output_dir)}

def generate_build_plan(knowledge_dir):
    p=Path(knowledge_dir); text='\n'.join(x.read_text(encoding='utf-8',errors='ignore') for x in p.glob('*.jsonl'))
    commands=[]
    for cmd in ('cmake','make','ninja','mvn','gradle','dotnet build','msbuild','npm run build','cargo build','python -m build'):
        if re.search(r'\b'+re.escape(cmd.split()[0])+r'\b',text,re.I): commands.append(cmd)
    if not commands: commands=['detect build system','resolve dependencies','compile','link','stage binaries','build BIOS/UEFI ISO','verify ISO']
    return {'steps':commands,'evidence':str(p),'generated_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'authorization_required':True}
