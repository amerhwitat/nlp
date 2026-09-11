"""Recursive external-reference discovery and build dependency planning.

This module does not guess that unrelated programs should be linked together.
It discovers source-level references, project manifests and package-manager
metadata, then exposes deterministic build/link inputs for the orchestrator.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

INCLUDE_RE = re.compile(r"^\s*#\s*include\s*[<\"]([^>\"]+)[>\"]", re.MULTILINE)
PRAGMA_LIB_RE = re.compile(r"#\s*pragma\s+comment\s*\(\s*lib\s*,\s*[\"']([^\"']+)[\"']\s*\)", re.IGNORECASE)

@dataclass
class ExternalReference:
    source: str
    reference: str
    kind: str
    resolved: bool = False
    target: str = ""

@dataclass
class DependencyGraph:
    references: List[ExternalReference] = field(default_factory=list)
    project_dependencies: List[str] = field(default_factory=list)
    unresolved: List[str] = field(default_factory=list)


def _iter_sources(root: Path) -> Iterable[Path]:
    skip = {".git", "node_modules", "build", "dist", "out", "target", "obj", "bin", ".venv", "venv", "__pycache__"}
    for base in root.rglob("*"):
        if not base.is_file() or any(part in skip for part in base.parts):
            continue
        if base.suffix.lower() in {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}:
            yield base


def _resolve_include(source: Path, root: Path, ref: str) -> Optional[Path]:
    candidates = [source.parent / ref, root / ref]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    matches = list(root.rglob(Path(ref).name))
    return matches[0].resolve() if len(matches) == 1 else None


def discover_external_references(root: Path) -> DependencyGraph:
    graph = DependencyGraph()
    manifests = {
        "package.json": "npm", "package-lock.json": "npm-lock", "yarn.lock": "yarn",
        "Cargo.toml": "cargo", "Cargo.lock": "cargo-lock", "go.mod": "go",
        "pom.xml": "maven", "build.gradle": "gradle", "build.gradle.kts": "gradle",
        "requirements.txt": "pip", "pyproject.toml": "python", "*.sln": "msbuild",
        "*.vcxproj": "msbuild", "*.csproj": "dotnet", "CMakeLists.txt": "cmake",
    }
    for p in root.rglob("*"):
        if not p.is_file() or any(part in {".git", "node_modules", "build", "dist", "target", "obj", "bin"} for part in p.parts):
            continue
        kind = manifests.get(p.name)
        if kind:
            graph.project_dependencies.append(f"{p.relative_to(root).as_posix()}:{kind}")
        elif p.name.endswith(".sln"):
            graph.project_dependencies.append(f"{p.relative_to(root).as_posix()}:msbuild")
        elif p.name.endswith(".vcxproj"):
            graph.project_dependencies.append(f"{p.relative_to(root).as_posix()}:msbuild")
        elif p.name.endswith(".csproj"):
            graph.project_dependencies.append(f"{p.relative_to(root).as_posix()}:dotnet")

    for source in _iter_sources(root):
        try:
            text = source.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = source.relative_to(root).as_posix()
        for inc in INCLUDE_RE.findall(text):
            target = _resolve_include(source, root, inc)
            graph.references.append(ExternalReference(rel, inc, "include", target is not None, target.relative_to(root).as_posix() if target else ""))
            if target is None and not inc.startswith(("windows.h", "commctrl.h", "string", "vector", "thread", "exception", "stdexcept")):
                graph.unresolved.append(f"{rel}: include <{inc}>")
        for lib in PRAGMA_LIB_RE.findall(text):
            graph.references.append(ExternalReference(rel, lib, "link-library", False, ""))

    graph.references.sort(key=lambda x: (x.source, x.kind, x.reference))
    graph.project_dependencies.sort()
    graph.unresolved = sorted(set(graph.unresolved))
    return graph


def write_dependency_report(root: Path, output: Path) -> DependencyGraph:
    graph = discover_external_references(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "references": [asdict(x) for x in graph.references],
        "project_dependencies": graph.project_dependencies,
        "unresolved": graph.unresolved,
    }, indent=2), encoding="utf-8")
    return graph
