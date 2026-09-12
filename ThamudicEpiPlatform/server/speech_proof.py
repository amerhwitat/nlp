from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib

@dataclass
class SpeechProof:
    text: str
    language: str
    voice_profile: str
    phonemes: list[str]
    model_type: str
    reconstructed: bool
    confidence: float
    review_required: bool
    text_sha256: str


def prove_for_speech(text: str, language: str, *, voice_profile: str = 'default',
                     phonemes: list[str] | None = None, model_type: str = 'unconfigured',
                     reconstructed: bool = False, confidence: float = 0.0) -> dict:
    """Create a provenance-safe speech proof record.

    RNN/Transformer/LLM phonology adapters can populate phonemes. This function
    does not invent a pronunciation when no model supplies one.
    """
    return asdict(SpeechProof(text=text, language=language, voice_profile=voice_profile,
        phonemes=phonemes or [], model_type=model_type, reconstructed=reconstructed,
        confidence=max(0.0,min(1.0,confidence)), review_required=confidence < 0.5,
        text_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest()))
