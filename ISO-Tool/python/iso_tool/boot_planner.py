from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json

@dataclass
class BootCandidate:
    id: str
    label: str
    loader: str
    mode: str
    artifact: str|None
    score: float

def load_catalog(path: Path): return json.loads(path.read_text(encoding='utf-8'))
def load_menu(path: Path): return json.loads(path.read_text(encoding='utf-8'))

def plan(root: Path, catalog_path: Path, menu_path: Path):
    catalog=load_catalog(catalog_path); menu=load_menu(menu_path)
    names={p.name.lower() for p in root.rglob('*') if p.is_file()}
    result=[]
    for e in menu['entries']:
        loader=next((x for x in catalog['loaders'] if x['id']==e['loader']),None)
        if not loader: continue
        artifact=e.get('efi') or e.get('image') or e.get('kernel')
        present=artifact is None or Path(artifact.lstrip('/')).name.lower() in names
        result.append(BootCandidate(e['id'],e['label'],e['loader'],e['mode'],artifact,0.9 if present else 0.2))
    return [asdict(x) for x in result]
