"""Unified ONNX Runtime provider selection for AVRS.

Provider names are discovered at runtime; unsupported accelerators fall back to CPU.
"""
from __future__ import annotations
from typing import Iterable

PREFERRED = [
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "OpenVINOExecutionProvider",
    "CoreMLExecutionProvider",
    "NNAPIExecutionProvider",
    "WebGPUExecutionProvider",
    "XNNPACKExecutionProvider",
    "CPUExecutionProvider",
]

def select_providers(available: Iterable[str], preferred: Iterable[str] = PREFERRED) -> list[str]:
    available_set = set(available)
    selected = [p for p in preferred if p in available_set]
    return selected or ["CPUExecutionProvider"]

def create_session(model_path: str, preferred: Iterable[str] = PREFERRED):
    import onnxruntime as ort
    providers = select_providers(ort.get_available_providers(), preferred)
    return ort.InferenceSession(model_path, providers=providers)
