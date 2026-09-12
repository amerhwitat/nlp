"""Low-resource RNN/LLM orchestration interface.

This is an inference/training contract, not a claim of pretrained weights.
Adapters can plug in PyTorch, TensorFlow, ONNX or external local models.
"""
from dataclasses import dataclass
from typing import Protocol, Sequence

@dataclass
class ModelResult:
    text: str
    confidence: float
    evidence: Sequence[str]

class AncientLanguageModel(Protocol):
    def translate(self, source: str, context: Sequence[str] = ()) -> ModelResult: ...

class RNNLLMEngine:
    def __init__(self, models=()):
        self.models = list(models)

    def translate(self, source: str, context=()):
        results = [m.translate(source, context) for m in self.models]
        return sorted(results, key=lambda r: r.confidence, reverse=True)

    def ensemble(self, source: str, context=()):
        return self.translate(source, context)
