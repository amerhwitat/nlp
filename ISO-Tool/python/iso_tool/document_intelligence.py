"""Repository document intelligence and deterministic build-plan extraction."""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

TEXT_EXTENSIONS={'.md','.markdown','.txt','.rst','.adoc','.json','.yaml','.yml','.toml','.ini','.cfg','.cmake','.mk','.make','.c','.cc','.cpp','.cxx','.h','.hh','.hpp','.hxx','.s','.asm','.inc','.rc','.def','.ld','.lds','.bat','.cmd','.ps1','.sh','.py','.java','.rs','.go','.js','.ts','.cs','.sln','.vcxproj','.props','.targets','.xml'}
KEYWORDS=("cmake","make","compile","compiler","link","linker","boot","bootloader","kernel","driver","efi","uefi","iso","img","filesystem","library","executable","dependency","install","build")

def scan_repository(root:Path)->dict:
    root=root.resolve(); documents=[]; files=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file() or '.git' in p.parts: continue
        files.append(str(p.relative_to(root)))
        if p.suffix.lower() not in TEXT_EXTENSIONS and p.name not in {'Makefile','CMakeLists.txt','Dockerfile'}: continue
        try: text=p.read_text(encoding='utf-8',errors='replace')
        except OSError: continue
        hits=[k for k in KEYWORDS if re.search(r'\b'+re.escape(k)+r'\b',text,re.I)]
        documents.append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(text.encode('utf-8')).hexdigest(),'keywords':hits,'lines':text.count('\n')+1})
    return {'root':str(root),'file_count':len(files),'document_count':len(documents),'files':files,'documents':documents}

def infer_components(index:dict)->dict:
    names=' '.join(d['path'].lower() for d in index['documents'])
    def has(*terms): return any(t in names for t in terms)
    return {'boot':has('boot','uefi','grub','spit'), 'kernel':has('kernel','koronos'), 'drivers':has('driver'), 'libraries':has('lib','library'), 'applications':has('app','application','bin'), 'filesystem':has('fs','filesystem','xfs','zfs','ntfs','fat'), 'build_system':has('cmakelists.txt','makefile','vcxproj','sln')}

def write_index(root:Path,out:Path)->Path:
    idx=scan_repository(root); idx['components']=infer_components(idx); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(idx,indent=2),encoding='utf-8'); return out
