"""Deterministic ISO build metadata and provenance helpers."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class BuildProvenance:
    source: str
    source_sha256: str
    output: str
    output_sha256: str
    profile: str
    source_date_epoch: Optional[int]
    tool_backend: str
    tool_version: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_date_epoch() -> Optional[int]:
    value = os.environ.get("SOURCE_DATE_EPOCH")
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        raise ValueError("SOURCE_DATE_EPOCH must be an integer Unix timestamp")


def make_provenance(source: Path, output: Path, profile: str,
                    backend: str, tool_version: str) -> BuildProvenance:
    return BuildProvenance(
        source=str(source), source_sha256=sha256_file(source),
        output=str(output), output_sha256=sha256_file(output),
        profile=profile, source_date_epoch=source_date_epoch(),
        tool_backend=backend, tool_version=tool_version,
    )


def write_manifest(path: Path, provenance: BuildProvenance) -> None:
    payload: Dict[str, object] = asdict(provenance)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
