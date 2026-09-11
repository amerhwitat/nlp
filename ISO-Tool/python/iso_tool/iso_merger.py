"""Canonical ISO staging/merge utilities."""
from __future__ import annotations
import hashlib, json, shutil
from pathlib import Path
IMAGE_SUFFIXES={'.iso','.img','.bin','.efi'}

def collect_images(root:Path): return sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)

def merge_staging(source:Path, staging:Path, manifest_path:Path)->dict:
    source=source.resolve(); staging=staging.resolve(); shutil.rmtree(staging,ignore_errors=True); staging.mkdir(parents=True)
    copied=[]
    for p in source.rglob('*'):
        if not p.is_file() or '.git' in p.parts: continue
        rel=p.relative_to(source); dst=staging/rel; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,dst)
        copied.append({'path':str(rel),'sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'bytes':dst.stat().st_size})
    manifest={'schema':1,'source':str(source),'staging':str(staging),'files':copied,'embedded_images':[str(p.relative_to(source)) for p in collect_images(source)]}
    manifest_path.parent.mkdir(parents=True,exist_ok=True); manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8'); return manifest
