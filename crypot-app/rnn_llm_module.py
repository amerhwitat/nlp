"""RNN/LLM integration facade for the crypto research application.

No remote model is contacted automatically. A model adapter can be supplied
by the host application, while the included recurrent state implementation is
safe for demonstrations and feature extraction.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Callable


@dataclass
class RNNState:
    size: int = 16
    values: list[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.values:
            self.values = [0.0] * self.size

    def step(self, token: str) -> list[float]:
        digest = sha256(token.encode("utf-8")).digest()
        x = [(b / 255.0) * 2.0 - 1.0 for b in digest[: self.size]]
        self.values = [0.65 * old + 0.35 * new for old, new in zip(self.values, x)]
        return list(self.values)


class LLMAdapter:
    """Adapter interface; pass a callable supplied by the application."""
    def __init__(self, infer: Callable[[str], str] | None = None):
        self.infer = infer

    def generate(self, prompt: str) -> str:
        if self.infer is None:
            return "No external LLM configured; local analysis only."
        return self.infer(prompt)


class CryptoResearchAssistant:
    def __init__(self, llm: LLMAdapter | None = None):
        self.rnn = RNNState()
        self.llm = llm or LLMAdapter()

    def analyze(self, text: str) -> dict:
        state = self.rnn.step(text)
        return {
            "sha256": sha256(text.encode("utf-8")).hexdigest(),
            "state_dimension": len(state),
            "state_norm": sum(v * v for v in state) ** 0.5,
            "llm": self.llm.generate(text),
        }
