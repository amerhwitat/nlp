from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import math

@dataclass
class ComprehensionResult:
    labels: list[str]
    confidence: float
    evidence_ids: list[str]
    interpretation_class: str

class SequenceComprehender:
    """Small dependency-free sequence feature engine.

    This is a deterministic reference layer, not a claim of historical truth and not
    a substitute for a trained language model. Production deployments may attach an
    RNN/Transformer/ONNX model while preserving the same evidence/provenance contract.
    """
    def __init__(self, feature_size: int = 128):
        self.feature_size = feature_size

    def encode(self, scene) -> list[float]:
        x = [0.0] * self.feature_size
        counts = [len(scene.events), len(scene.characters), len(scene.evidence)]
        x[:3] = [float(v) for v in counts]
        if scene.environment:
            x[3] = float(scene.environment.get("temperature_c", 0.0))
            x[4] = float(scene.environment.get("rain_mm", 0.0))
            x[5] = float(scene.environment.get("visibility_km", 0.0))
        for i, value in enumerate(scene.tensor128[:122]):
            x[6 + i] += float(value)
        return [v / (1.0 + abs(v)) for v in x]

    def classify(self, scene) -> ComprehensionResult:
        score = 0.5 + min(0.45, len(scene.evidence) * 0.03)
        labels = ["historical-scene", "event-sequence"]
        if scene.characters:
            labels.append("agent-interaction")
        if scene.sky:
            labels.append("astronomical-context")
        kinds = {e.kind for e in scene.evidence}
        if "observed" in kinds and "supported" in kinds:
            cls = "mixed-evidence-reconstruction"
        elif "observed" in kinds:
            cls = "evidence-led"
        else:
            cls = "inferred-visualization"
        return ComprehensionResult(labels, score, [e.id for e in scene.evidence], cls)
