from __future__ import annotations

from pydantic import BaseModel, Field


class ScanRequest(BaseModel):
    text: str = Field(default="", max_length=2_000_000)
    keywords: list[str] = Field(default_factory=list, max_length=100)
    source: str = Field(default="text-input", max_length=512)


class ValidationRequest(BaseModel):
    text: str = Field(default="", max_length=2_000_000)


class ScanResult(BaseModel):
    id: str
    session_id: str
    source: str
    text: str
    transliteration: str
    confidence: float = Field(ge=0, le=1)
    language: str = "Old North Arabian"
    script_variant: str = "Dadanitic"
    codepoints: list[int] = Field(default_factory=list)


class ScanSummary(BaseModel):
    processed_count: int
    match_count: int
    status: str


class ScanResponse(BaseModel):
    session_id: str
    results: list[ScanResult]
    summary: ScanSummary


class ProgressEvent(BaseModel):
    sequence: int
    type: str
    status: str
    progress: int = Field(ge=0, le=100)
    processed_count: int = 0
    match_count: int = 0
    source: str | None = None
    message: str | None = None
