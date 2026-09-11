"""Deterministic discovery for ISO-Tool assembler/disassembler backends."""
from __future__ import annotations
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REGISTRY = ROOT / "registry.json"


def load_registry() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def _windows_vs_paths() -> list[Path]:
    roots = [
        Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Microsoft Visual Studio",
        Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio",
    ]
    return [p for r in roots if r.exists() for p in r.rglob("ml64.exe")]


def find_command(name: str) -> str | None:
    hit = shutil.which(name)
    if hit:
        return hit
    if name in {"ml.exe", "ml64.exe", "ml", "ml64"}:
        hits = _windows_vs_paths()
        return str(hits[0]) if hits else None
    return None


def discover() -> list[dict]:
    result = []
    for backend in load_registry()["backends"]:
        commands = []
        for command in backend["commands"]:
            found = find_command(command)
            if found:
                commands.append(found)
        result.append({**backend, "found": commands, "available": bool(commands)})
    return result


def select(kind: str, target: str | None = None) -> dict | None:
    candidates = [x for x in discover() if x["available"] and (kind in x["kind"].split("-") or x["kind"] == kind)]
    if target:
        targeted = [x for x in candidates if target in x["targets"] or "multi-target" in x["targets"] or "llvm-targets" in x["targets"]]
        candidates = targeted or candidates
    return candidates[0] if candidates else None


def version(executable: str) -> str:
    try:
        p = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=5)
        return (p.stdout or p.stderr).splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


if __name__ == "__main__":
    for item in discover():
        exe = item["found"][0] if item["found"] else ""
        print(f'{item["id"]}: {"available" if item["available"] else "missing"} {exe} {version(exe) if exe else ""}'.rstrip())
