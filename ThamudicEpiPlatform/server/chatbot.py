from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

@dataclass
class ChatMessage:
    role: str
    content: str

@dataclass
class ChatResponse:
    answer: str
    confidence: float
    citations: list[str]
    tool_calls: list[dict[str, Any]]
    review_required: bool = False

class ResearchChatbot:
    """Provider-neutral chatbot orchestration boundary.

    The platform can connect an approved local/self-hosted LLM, RAG system,
    or hosted model through `generate`. No model or web crawler is silently
    invoked here; evidence must be supplied as context with citations.
    """
    def __init__(self, generator=None):
        self.generator = generator

    def generate(self, messages: list[ChatMessage], context: list[dict[str, Any]] | None = None) -> ChatResponse:
        if self.generator is None:
            return ChatResponse(
                answer="No chatbot model is configured. Supply a local or approved model adapter.",
                confidence=0.0,
                citations=[],
                tool_calls=[],
                review_required=True,
            )
        result = self.generator(messages, context or [])
        return ChatResponse(
            answer=str(result.get("answer", "")),
            confidence=float(result.get("confidence", 0.0)),
            citations=list(result.get("citations", [])),
            tool_calls=list(result.get("tool_calls", [])),
            review_required=float(result.get("confidence", 0.0)) < 0.5,
        )


def health() -> dict[str, Any]:
    return {"service": "research-chatbot", "provider": "pluggable", "safe_default": True}
