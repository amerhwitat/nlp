"""Language-neutral ancient-script analysis pipeline.

The implementation deliberately separates recognition from interpretation so
uncertain or damaged readings can survive into translation and review.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class Reading:
    script: str
    text: str
    transliteration: Optional[str] = None
    unicode_text: Optional[str] = None
    confidence: float = 0.0
    alternates: List[str] = field(default_factory=list)
    damage: List[str] = field(default_factory=list)
    provenance: List[str] = field(default_factory=list)

@dataclass
class TranslationCandidate:
    text: str
    confidence: float
    evidence: List[str] = field(default_factory=list)

class AncientScriptEngine:
    def normalize(self, reading: Reading) -> Reading:
        reading.text = " ".join(reading.text.split())
        if reading.transliteration:
            reading.transliteration = " ".join(reading.transliteration.split())
        return reading

    def rank(self, candidates: List[TranslationCandidate]) -> List[TranslationCandidate]:
        return sorted(candidates, key=lambda c: c.confidence, reverse=True)

    def analyze(self, reading: Reading, candidates: List[TranslationCandidate]) -> Dict:
        reading = self.normalize(reading)
        ranked = self.rank(candidates)
        return {"reading": reading, "translations": ranked,
                "requires_review": reading.confidence < 0.85 or bool(reading.damage)}
