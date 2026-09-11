"""Recursive repository inventory, build orchestration and native-link planning.

The builder walks a complete checkout, detects language/build manifests, compiles
supported projects through their native build systems, and records artifacts.
Unrelated languages are never incorrectly forced into one native executable:
native targets are linked per compatible target, while managed/interpreted code
is built or byte-compiled separately.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    "build", "dist", "out", "target", "bin", "obj", ".vs", ".idea",
}
EXTENSIONS = {
    ".c": "c", ".h": "c-header", ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp",
    ".hpp": "cpp-header", ".hh": "cpp-header", ".hxx": "cpp-header",
    ".s": "asm", ".asm": "asm", ".S": "asm", ".rs": "rust", ".go": "go",
    ".py": "python", ".js": "javascript", ".mjs": "javascript", ".ts": "typescript",
    ".java": "java", ".cs": "csharp", ".fs": "fsharp", ".fsx": "fsharp",
}
MANIFESTS = {
    "CMakeLists.txt": "cmake", "Makefile": "make", "makefile": "make",
    "configure.ac": "autotools", "meson.build": "meson", "Cargo.toml": "cargo",
    "go.mod": "go", "package.json": "node", "pyproject.toml": "python",
    "setup.py": "python", "pom.xml": "maven", "build.gradle": "gradle",
    "build.gradle.kts": "gradle", "*.sln": "msbuild", "*.vcxproj": "msbuild",
    "*.csproj": "dotnet",
}

@dataclass
class SourceRecord:
    path: str
    language: str
    size: int
    entry_point: bool = False

@dataclass
class ManifestRecord:
    path: str
    kind: str

@dataclass
class BuildArtifact:
    path: str
    kind: str
    command: List[str] = field(default_factory=list)
    returncode: int = 0
    status: str = "planned"

@dataclass
class RecursiveReport:
    root: str
    sources: List[SourceRecord]
    manifests: List[ManifestRecord]
    artifacts: List[BuildArtifact]
    skipped: List[str]
    errors: List[str]


def _manifest_kind(name: str) -> Optional[str]:
    if name in MANIFESTS:
        return MANIFESTS[name]
    for pattern, kind in MANIFESTS.items():
        if pattern.startswith("*") and name.endswith(pattern[1:]):
            return kind
    return None


def iter_files(root: Path) -> Iterable[Path]:
    for base, dirs, files in os.walk(str(root)):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            yield Path(base) / name


def _looks_like_entry(path: Path) -> bool:
    if path.suffix.lower() not in {".c", ".cc", ".cpp", ".cxx"}:
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return bool(re.search(r"\b(?:int|auto)\s+main\s*\(", text)) or "wWinMain(" in text


def inventory(root: Path) -> Tuple[List[SourceRecord], List[ManifestRecord]]:
    sources: List[SourceRecord] = []
    manifests: List[ManifestRecord] = []
    for path in iter_files(root):
        rel = path.relative_to(root).as_posix()
        kind = _manifest_kind(path.name)
        if kind:
            manifests.append(ManifestRecord(rel, kind))
        language = EXTENSIONS.get(path.suffix)
        if language:
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            sources.append(SourceRecord(rel, language, size, _looks_like_entry(path)))
    sources.sort(key=lambda x: x.path)
    manifests.sort(key=lambda x: x.path)
    return sources, manifests


def _tool(name: str) -> Optional[str]:
    return shutil.which(name)


def _run(cmd: Sequence[str], cwd: Path, log: Optional[Path] = None) -> int:
    proc = subprocess.run(list(cmd), cwd=str(cwd), stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, errors="replace")
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as fh:
            fh.write("$ " + " ".join(cmd) + "\n")
            fh.write(proc.stdout)
            fh.write("\n")
    return proc.returncode


def acquire_repository(source: str, destination: Optional[Path] = None) -> Tuple[Path, Optional[Path]]:
    """Accept a local checkout or clone a GitHub/HTTPS repository into a temporary workspace."""
    candidate = Path(source).expanduser()
    if candidate.exists():
        return candidate.resolve(), None
    parsed = urlparse(source)
    if parsed.scheme not in {"https", "http", "git"} or not parsed.netloc:
        raise ValueError("repository must be an existing path or a Git URL")
    git = _tool("git")
    if not git:
        raise RuntimeError("git is required to acquire a remote repository")
    if destination:
        destination.mkdir(parents=True, exist_ok=True)
        checkout = destination / "repository"
        cleanup = None
    else:
        temp = Path(tempfile.mkdtemp(prefix="iso-tool-repo-"))
        checkout = temp / "repository"
        cleanup = temp
    rc = _run([git, "clone", "--recursive", "--depth", "1", source, str(checkout)], checkout.parent)
    if rc != 0:
        if cleanup:
            shutil.rmtree(str(cleanup), ignore_errors=True)
        raise RuntimeError("git clone failed")
    return checkout.resolve(), cleanup


def _native_direct_build(root: Path, sources: List[SourceRecord], out: Path,
                         log: Path) -> Tuple[List[BuildArtifact], List[str]]:
    """Compile standalone C/C++ sources recursively; link only a single-entry target."""
    native = [s for s in sources if s.language in {"c", "cpp"}]
    if not native:
        return [], []
    compiler = _tool("c++") or _tool("g++") or _tool("clang++")
    ccompiler = _tool("cc") or _tool("gcc") or _tool("clang")
    if not compiler and not ccompiler:
        return [], ["No C/C++ compiler found; native direct-build jobs remain planned."]

    objdir = out / "objects"
    objdir.mkdir(parents=True, exist_ok=True)
    artifacts: List[BuildArtifact] = []
    errors: List[str] = []
    objects: List[Path] = []
    for record in native:
        src = root / record.path
        obj = objdir / (record.path.replace("/", "__") + ".o")
        obj.parent.mkdir(parents=True, exist_ok=True)
        cc = ccompiler if record.language == "c" else compiler
        if not cc:
            errors.append("No compiler for " + record.path)
            continue
        standard = "-std=c++17" if record.language == "cpp" else "-std=c11"
        cmd = [cc, "-c", "-O2", standard, str(src), "-o", str(obj)]
        rc = _run(cmd, root, log)
        artifacts.append(BuildArtifact(str(obj.relative_to(out)), "object", cmd, rc,
                                       "built" if rc == 0 else "failed"))
        if rc == 0:
            objects.append(obj)
        else:
            errors.append("Compile failed: " + record.path)

    entries = [s for s in native if s.entry_point]
    if len(entries) == 1 and objects:
        exe = out / ("recursive-native.exe" if os.name == "nt" else "recursive-native")
        cmd = [compiler or ccompiler] + [str(p) for p in objects] + ["-o", str(exe)]
        rc = _run(cmd, root, log)
        artifacts.append(BuildArtifact(str(exe.relative_to(out)), "executable", cmd, rc,
                                       "built" if rc == 0 else "failed"))
        if rc != 0:
            errors.append("Native link failed; inspect the recursive build log.")
    elif len(entries) > 1:
        errors.append("Multiple native entry points detected; direct sources are compiled but not force-linked into one executable.")
    return artifacts, errors


def build_repository(root: Path, output: Optional[Path] = None, execute: bool = False) -> RecursiveReport:
    root = root.resolve()
    out = (output or root / "ISO-Tool-build").resolve()
    out.mkdir(parents=True, exist_ok=True)
    sources, manifests = inventory(root)
    artifacts: List[BuildArtifact] = []
    errors: List[str] = []
    skipped: List[str] = []

    if not execute:
        for m in manifests:
            artifacts.append(BuildArtifact(m.path, "build-manifest", status="planned"))
        return RecursiveReport(str(root), sources, manifests, artifacts, skipped, errors)

    log = out / "recursive-build.log"
    handlers = {
        "cmake": ["cmake", "--build", "build", "--config", "Release"],
        "make": ["make"], "meson": ["meson", "compile", "-C", "build"],
        "cargo": ["cargo", "build", "--release"], "go": ["go", "build", "./..."],
        "maven": ["mvn", "-B", "package"], "gradle": ["gradle", "build"],
        "msbuild": ["msbuild", "/m", "/p:Configuration=Release"],
        "dotnet": ["dotnet", "build", "-c", "Release"],
        "node": ["npm", "run", "build"],
    }
    # Build every discovered manifest, not only the first one. Each project is built in its own directory.
    for manifest in manifests:
        kind = manifest.kind
        command = handlers.get(kind)
        if not command:
            if kind in {"python", "autotools"}:
                skipped.append(manifest.path + ": no universal native build command")
            continue
        if not _tool(command[0]):
            skipped.append(manifest.path + ": " + command[0] + " not installed")
            continue
        work = root / Path(manifest.path).parent
        if kind == "cmake" and not (work / "build").exists():
            rc = _run(["cmake", "-S", str(work), "-B", str(work / "build"), "-DCMAKE_BUILD_TYPE=Release"], root, log)
            if rc != 0:
                errors.append("CMake configure failed: " + manifest.path)
                continue
        rc = _run(command, work, log)
        artifacts.append(BuildArtifact(manifest.path, kind, command, rc,
                                       "built" if rc == 0 else "failed"))
        if rc != 0:
            errors.append("Build failed: " + manifest.path)

    direct, direct_errors = _native_direct_build(root, sources, out, log)
    artifacts.extend(direct)
    errors.extend(direct_errors)

    py = [s for s in sources if s.language == "python"]
    python = _tool("python") or _tool("python3")
    if py and python:
        rc = _run([python, "-m", "compileall", "-q", str(root)], root, log)
        artifacts.append(BuildArtifact("__pycache__", "python-bytecode", [python, "-m", "compileall", "-q", str(root)], rc,
                                       "built" if rc == 0 else "failed"))

    report = RecursiveReport(str(root), sources, manifests, artifacts, skipped, errors)
    (out / "recursive-build-report.json").write_text(
        json.dumps({"root": report.root, "sources": [asdict(x) for x in report.sources],
                    "manifests": [asdict(x) for x in report.manifests],
                    "artifacts": [asdict(x) for x in report.artifacts],
                    "skipped": report.skipped, "errors": report.errors}, indent=2),
        encoding="utf-8")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Recursively inventory/build a GitHub or local repository")
    parser.add_argument("source", help="local checkout or Git/HTTPS repository URL")
    parser.add_argument("--output", default=None, help="artifact/report directory or clone parent")
    parser.add_argument("--execute", action="store_true", help="actually invoke discovered build tools")
    args = parser.parse_args(argv)
    root, cleanup = acquire_repository(args.source, Path(args.output) if args.output else None)
    try:
        report = build_repository(root, Path(args.output) / "artifacts" if args.output and cleanup is None else None, args.execute)
        print(json.dumps({"root": report.root, "source_count": len(report.sources),
                          "manifest_count": len(report.manifests), "artifact_count": len(report.artifacts),
                          "errors": report.errors, "skipped": report.skipped}, indent=2))
        return 1 if report.errors else 0
    finally:
        if cleanup:
            shutil.rmtree(str(cleanup), ignore_errors=True)

if __name__ == "__main__":
    raise SystemExit(main())
