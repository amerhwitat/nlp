from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal
import unicodedata

TranslationMode = Literal["literal", "meaning", "interlinear", "scholarly"]

@dataclass
class TranslationResult:
    source_text: str
    source_language: str
    target_language: str
    mode: TranslationMode
    output: str
    confidence: float
    proof_notes: list[str]
    alternatives: list[str]
    provenance: dict


def normalize_source(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def prepare_translation(source_text: str, source_language: str, target_language: str,
                        mode: TranslationMode = "meaning", *, output: str = "",
                        confidence: float = 0.0, proof_notes: list[str] | None = None,
                        alternatives: list[str] | None = None, provenance: dict | None = None) -> dict:
    """Package an engine-produced translation without pretending a missing model exists."""
    result = TranslationResult(
        normalize_source(source_text), source_language, target_language, mode,
        output, max(0.0, min(1.0, confidence)), proof_notes or [],
        alternatives or [], provenance or {}
    )
    return asdict(result)


def proof_translation(result: dict) -> dict:
    """Conservative proofing gate for neural/LLM output."""
    text = result.get("output", "")
    notes = list(result.get("proof_notes", []))
    if not text:
        notes.append("No translation output was supplied by the selected engine.")
    if result.get("confidence", 0) < 0.5:
        notes.append("Low confidence: human review required.")
    if result.get("mode") in {"literal", "meaning"} and not result.get("alternatives"):
        notes.append("No alternative reading supplied; do not treat absence as certainty.")
    result["proof_notes"] = notes
    result["review_required"] = any("required" in n.lower() for n in notes)
    return result
