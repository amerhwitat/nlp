from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import re, json

@dataclass
class DocumentInsight:
    path: str
    role: str
    confidence: float
    keywords: list[str]
    boot_candidates: list[str]
    install_candidates: list[str]

class DocumentationBrain:
    """Deterministic documentation intelligence layer.

    This is deliberately separate from an optional neural/LLM provider. It provides
    useful behavior offline and gives an LLM/RNN provider normalized evidence rather
    than allowing a model to execute arbitrary repository instructions.
    """
    DOC_EXT={'.md','.markdown','.txt','.rst','.adoc','.html','.htm','.pdf'}
    BOOT_WORDS=('boot','loader','grub','lilo','uefi','efi','mbr','gpt','kernel','initrd','bootsector')
    INSTALL_WORDS=('install','setup','installer','live cd','live dvd','filesystem','partition','disk','nvme','ssd')

    def read_text(self, path: Path) -> str:
        if path.suffix.lower()=='.pdf': return ''  # parsed by optional PDF provider
        try: return path.read_text(encoding='utf-8', errors='replace')
        except OSError: return ''

    def analyze(self, root: Path) -> list[DocumentInsight]:
        out=[]
        for p in root.rglob('*'):
            if not p.is_file() or p.suffix.lower() not in self.DOC_EXT: continue
            text=self.read_text(p); low=text.lower()
            boot=[w for w in self.BOOT_WORDS if w in low]
            install=[w for w in self.INSTALL_WORDS if w in low]
            role='boot' if boot else ('installer' if install else 'documentation')
            hits=len(set(boot+install)); confidence=min(0.99,0.35+hits*0.08)
            words=re.findall(r'[A-Za-z][A-Za-z0-9_-]{2,}',low)
            freq={w:words.count(w) for w in set(words)}
            keys=[w for w,_ in sorted(freq.items(),key=lambda x:x[1],reverse=True)[:12]]
            out.append(DocumentInsight(str(p.relative_to(root)),role,confidence,keys,boot,install))
        return out

    def write_report(self, root: Path, output: Path):
        data=[asdict(x) for x in self.analyze(root)]
        output.write_text(json.dumps({'documents':data},indent=2),encoding='utf-8')
        return output
