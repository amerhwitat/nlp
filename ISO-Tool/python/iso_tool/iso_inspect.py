"""Offline ISO structure inspection and boot-artifact discovery."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib

SECTOR = 2048
@dataclass(frozen=True)
class IsoInspection:
    path: str
    size: int
    sha256: str
    iso9660: bool
    joliet: bool
    udf: bool
    el_torito: bool
    boot_catalog_sector: int | None
    notes: tuple[str, ...]

def _hash(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def inspect_iso(path: Path) -> IsoInspection:
    path=Path(path)
    notes=[]; iso=False; joliet=False; udf=False; et=False; catalog=None
    with path.open('rb') as f:
        for sector in range(16, 64):
            f.seek(sector*SECTOR); d=f.read(SECTOR)
            if len(d)<7: break
            ident=d[1:6]
            if ident==b'CD001':
                iso=True
                if d[0]==0: notes.append('Primary Volume Descriptor found')
                if d[0]==255: break
                if d[0]==0 and b'EL TORITO SPECIFICATION' in d:
                    et=True; catalog=int.from_bytes(d[71:75],'little'); notes.append(f'El Torito boot catalog sector {catalog}')
                if d[0]==2 and d[88:120].decode('latin1','ignore').startswith('%/'): joliet=True
            if d[1:6] in (b'BEA01',b'NSR02',b'NSR03'): udf=True
    if et: notes.append('Boot catalog detected; use backend tooling to extract individual boot images')
    return IsoInspection(str(path),path.stat().st_size,_hash(path),iso,joliet,udf,et,catalog,tuple(notes))

def inspection_dict(path: Path) -> dict:
    return asdict(inspect_iso(path))
