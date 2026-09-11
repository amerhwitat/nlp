"""Recursive repository inventory, build orchestration and native-link planning.

The builder walks a complete checkout, detects language/build manifests, compiles
supported projects through their native build systems, and records artifacts.
It deliberately does not pretend that unrelated languages can be linked into one
native executable: native C/C++/ASM targets are linkable when a target has one
entry point; managed/interpreted projects are built or byte-compiled separately.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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


def _first_manifest(manifests: List[ManifestRecord], kinds: Sequence[str]) -> Optional[ManifestRecord]:
    return next((m for m in manifests if m.kind in kinds), None)


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
        cmd = [cc, "-c", "-O2", "-std=c++17" if record.language == "cpp" else "-std=c11",
               str(src), "-o", str(obj)]
        rc = _run(cmd, root, log)
        artifacts.append(BuildArtifact(str(obj.relative_to(out)), "object", cmd, rc,
                                       "built" if rc == 0 else "failed"))
        if rc == 0:
            objects.append(obj)
        else:
            errors.append("Compile failed: " + record.path)

    entries = [s for s in native if s.entry_point]
    if len(entries) == 1 and objects:
        exe = out / "recursive-native" + (".exe" if os.name == "nt" else "")
        # The executable name is created from Path pieces to avoid shell interpolation.
        exe = Path(str(exe))
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
    # Prefer project-native build systems. This avoids incorrectly linking unrelated projects.
    handlers = [
        ("cmake", ["cmake", "--build", "build", "--config", "Release"]),
        ("make", ["make"]),
        ("meson", ["meson", "compile", "-C", "build"]),
        ("cargo", ["cargo", "build", "--release"]),
        ("go", ["go", "build", "./..."]),
        ("maven", ["mvn", "-B", "package"]),
        ("gradle", ["gradle", "build"]),
        ("msbuild", ["msbuild", "/m", "/p:Configuration=Release"]),
        ("dotnet", ["dotnet", "build", "-c", "Release"]),
        ("node", ["npm", "run", "build"]),
    ]
    for kind, command in handlers:
        manifest = _first_manifest(manifests, [kind])
        if not manifest:
            continue
        if not _tool(command[0]):
            skipped.append(kind + ": tool not installed")
            continue
        work = root / Path(manifest.path).parent
        # CMake/meson commonly use a generated build directory; create it only for explicit execution.
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

    # Compile direct C/C++ sources not owned by an obvious project manifest.
    direct, direct_errors = _native_direct_build(root, sources, out, log)
    artifacts.extend(direct)
    errors.extend(direct_errors)

    # Python bytecode is useful for recursive packaging, but is not mislabeled as native code.
    py = [s for s in sources if s.language == "python"]
    if py and _tool("python"):
        rc = _run([_tool("python") or "python", "-m", "compileall", "-q", str(root)], root, log)
        artifacts.append(BuildArtifact("__pycache__", "python-bytecode", ["python", "-m", "compileall", "-q", str(root)], rc,
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
    parser.add_argument("root", help="local checkout root")
    parser.add_argument("--output", default=None, help="artifact/report directory")
    parser.add_argument("--execute", action="store_true", help="actually invoke discovered build tools")
    args = parser.parse_args(argv)
    report = build_repository(Path(args.root), Path(args.output) if args.output else None, args.execute)
    print(json.dumps({"root": report.root, "source_count": len(report.sources),
                      "manifest_count": len(report.manifests), "artifact_count": len(report.artifacts),
                      "errors": report.errors, "skipped": report.skipped}, indent=2))
    return 1 if report.errors else 0

if __name__ == "__main__":
    raise SystemExit(main())
