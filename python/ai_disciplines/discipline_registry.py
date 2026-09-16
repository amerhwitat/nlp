"""AI discipline registry for the NLP/Thamudic research stack."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class Discipline:
    name: str
    core_approach: str
    data_dependency: str
    primary_use_cases: Tuple[str, ...]
    reference_tools: Tuple[str, ...]

DISCIPLINES = (
    Discipline("ML", "statistical pattern learning", "structured/tabular and mixed data", ("prediction", "risk", "recommendation"), ("scikit-learn",)),
    Discipline("DL", "hierarchical neural representation learning", "text/audio/images/tensors", ("vision", "speech", "LLMs"), ("PyTorch", "TensorFlow", "JAX")),
    Discipline("RL", "policy optimization from environment interaction and rewards", "simulation/interaction trajectories", ("games", "robotics", "routing"), ("Gymnasium", "Stable-Baselines3")),
    Discipline("Symbolic AI", "explicit rules, logic and knowledge representation", "facts/rules/ontologies", ("expert systems", "verification", "reasoning"), ("SymPy", "RDF/OWL")),
    Discipline("Computer Vision", "spatial feature and geometry extraction", "image/video/3D data", ("OCR", "detection", "segmentation", "tracking"), ("OpenCV", "scikit-image")),
    Discipline("NLP", "computational linguistics plus statistical/neural language models", "text/speech/corpora", ("tokenization", "translation", "NER", "generation"), ("spaCy", "Transformers", "PyTorch")),
)

def get_registry():
    return [d.__dict__ for d in DISCIPLINES]
