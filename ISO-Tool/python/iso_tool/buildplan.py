from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class BuildJob:
    kind: str
    source: Path
    command: tuple[str,...]
    depends_on: tuple[str,...]=()

def make_inventory(root: Path):
    groups={'c':[], 'cpp':[], 'asm':[], 'csharp':[], 'projects':[], 'boot':[]}
    for p in root.rglob('*'):
        if not p.is_file(): continue
        s=p.suffix.lower()
        if s=='.c': groups['c'].append(p)
        elif s in ('.cc','.cpp','.cxx'): groups['cpp'].append(p)
        elif s in ('.asm','.s','.spp','.inc'): groups['asm'].append(p)
        elif s=='.cs': groups['csharp'].append(p)
        elif s in ('.sln','.vcxproj','.csproj','.cmake','.mk'): groups['projects'].append(p)
        if p.name.lower() in ('boot.asm','boot.s','bootsector.asm','bootsector.s'): groups['boot'].append(p)
    return groups
