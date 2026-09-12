from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any


class ReadingDirection(str, Enum):
    RTL = "rtl"
    LTR = "ltr"
    TTB = "ttb"
    BTT = "btt"
    SPIRAL = "spiral"
    REVERSE = "reverse"
    UNKNOWN = "unknown"


@dataclass
class GeometryHypothesis:
    direction: ReadingDirection
    rotation_degrees: float
    skew_degrees: float
    perspective_score: float
    weathering_score: float
    confidence: float
    operations: list[str]


def analyze_geometry(width: int, height: int, *, edge_density: float = 0.0,
                     contrast: float = 0.0, blur: float = 0.0,
                     orientation_hint: str | None = None) -> GeometryHypothesis:
    """Produce a conservative geometry hypothesis for an OCR pipeline.

    This is intentionally a routing layer, not a historical-script classifier.
    Engine-specific document orientation/unwarping models can replace the
    heuristic values while preserving the same contract.
    """
    hint = (orientation_hint or "").lower()
    mapping = {
        "rtl": ReadingDirection.RTL, "ltr": ReadingDirection.LTR,
        "ttb": ReadingDirection.TTB, "btt": ReadingDirection.BTT,
        "spiral": ReadingDirection.SPIRAL, "reverse": ReadingDirection.REVERSE,
    }
    direction = mapping.get(hint, ReadingDirection.UNKNOWN)
    weathering = max(0.0, min(1.0, (0.35 - contrast) + (blur * 0.002)))
    perspective = 0.0 if width == 0 or height == 0 else min(1.0, abs(math.log(max(width, 1) / max(height, 1))) / 3.0)
    confidence = 0.65 if direction != ReadingDirection.UNKNOWN else 0.25
    if weathering > 0.5:
        confidence *= 0.85
    operations = ["normalize", "quality-check"]
    if perspective > 0.35:
        operations.append("perspective-unwarp")
    if weathering > 0.35:
        operations.extend(["denoise", "contrast-enhance", "local-threshold"])
    if direction in {ReadingDirection.TTB, ReadingDirection.BTT}:
        operations.append("vertical-line-segmentation")
    elif direction == ReadingDirection.SPIRAL:
        operations.append("spiral-path-segmentation")
    elif direction == ReadingDirection.REVERSE:
        operations.append("reverse-reading-hypothesis")
    else:
        operations.append("line-segmentation")
    return GeometryHypothesis(direction, 0.0, 0.0, perspective, weathering,
                              max(0.0, min(1.0, confidence)), operations)


def build_engine_options(h: GeometryHypothesis) -> dict[str, Any]:
    """Map geometry hypotheses into adapter-neutral OCR options."""
    return {
        "reading_direction": h.direction.value,
        "deskew": True,
        "unwarp": "perspective-unwarp" in h.operations,
        "vertical_text": h.direction in {ReadingDirection.TTB, ReadingDirection.BTT},
        "reverse_hypothesis": h.direction == ReadingDirection.REVERSE,
        "spiral_hypothesis": h.direction == ReadingDirection.SPIRAL,
        "weathered_preprocess": h.weathering_score >= 0.35,
        "operations": h.operations,
    }
