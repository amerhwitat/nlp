"""Cross-discipline dispatcher used by OCR/NLP research applications.

Heavy frameworks are optional. The dispatcher records which discipline owns a
processing stage and keeps provenance explicit instead of inventing results.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable

@dataclass
class Stage:
    discipline: str
    name: str
    processor: Callable[[Any], Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def run(self, value: Any) -> Any:
        if self.processor is None:
            return value
        return self.processor(value)

class AIPipeline:
    def __init__(self, stages: list[Stage] | None = None):
        self.stages = stages or []
        self.audit: list[dict[str, Any]] = []

    def add(self, stage: Stage) -> "AIPipeline":
        self.stages.append(stage)
        return self

    def run(self, value: Any) -> Any:
        current = value
        for stage in self.stages:
            current = stage.run(current)
            self.audit.append({"discipline": stage.discipline, "stage": stage.name, "metadata": stage.metadata})
        return current
