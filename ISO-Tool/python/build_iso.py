"""End-to-end repository build followed by ISO mastering.

The caller chooses the final ISO path/name. Repository source is recursively
analyzed, external project dependencies are resolved through native build
systems, build artifacts are collected, and the requested ISO backend receives
an explicit output path.
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
from .recursive_build import acquire_repository, build_repository
from .image import create_iso


def stage_repository(root: Path, build_output: Path) -> Path:
    staging = build_output / "iso-staging"
    if staging.exists(): shutil.rmtree(str(staging))
    staging.mkdir(parents=True)
    excluded = {".git", ".github", "ISO-Tool-build", "node_modules", "__pycache__", ".venv", "venv", "build", "dist", "target", "obj", "bin"}
    for source in root.rglob("*"):
        rel = source.relative_to(root)
        if any(part in excluded for part in rel.parts): continue
        target = staging / rel
        if source.is_dir(): target.mkdir(parents=True, exist_ok=True)
        elif source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(source), str(target))
    artifact_root = build_output / "artifacts"
    if artifact_root.exists():
        target = staging / "compiled-artifacts"
        shutil.copytree(str(artifact_root), str(target), dirs_exist_ok=True)
    return staging


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Recursively build a repository and create an ISO at the user-selected path")
    parser.add_argument("source", help="local repository or Git URL")
    parser.add_argument("--output", required=True, help="final ISO path, including the desired filename")
    parser.add_argument("--label", default="ISO_TOOL")
    parser.add_argument("--profile", default="data")
    parser.add_argument("--build-output", default=None)
    parser.add_argument("--no-resolve-dependencies", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    build_output = Path(args.build_output).expanduser().resolve() if args.build_output else output.parent / (output.stem + ".iso-tool-build")
    root, cleanup = acquire_repository(args.source, build_output / "source" if Path(args.source).as_posix() != str(Path(args.source).resolve()) else None)
    try:
        report = build_repository(root, build_output / "artifacts", True, not args.no_resolve_dependencies)
        if report.errors:
            raise RuntimeError("recursive build reported errors; see recursive-build-report.json")
        staging = stage_repository(root, build_output)
        result = create_iso(staging, output, args.label, args.profile)
        summary = {"output": str(output), "backend_returncode": result.returncode, "source_count": len(report.sources), "artifact_count": len(report.artifacts), "external_reference_count": report.external_references, "staging": str(staging)}
        print(json.dumps(summary, indent=2))
        return 0
    finally:
        if cleanup: shutil.rmtree(str(cleanup), ignore_errors=True)

if __name__ == "__main__": raise SystemExit(main())
