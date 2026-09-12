from __future__ import annotations
from dataclasses import dataclass
import hashlib

@dataclass(frozen=True)
class CharacterSpec:
    id: str
    role: str
    height_m: float
    clothing_profile: str
    activity: str
    evidence_ids: tuple[str, ...]

class CharacterGenerator:
    """Creates reproducible placeholder agents from evidence, not invented identities."""
    def generate(self, role: str, evidence_ids: list[str], activity: str = "idle") -> CharacterSpec:
        seed = '|'.join(sorted(evidence_ids)) + '|' + role + '|' + activity
        digest = hashlib.sha256(seed.encode()).hexdigest()
        height = 1.55 + (int(digest[:4],16) / 65535.0) * 0.35
        profile = ['simple_robe','tunic','work_clothing','military_profile'][int(digest[4:6],16)%4]
        return CharacterSpec('char-' + digest[:12], role, round(height,3), profile, activity, tuple(evidence_ids))
