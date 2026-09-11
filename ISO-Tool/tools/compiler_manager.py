"""Dual C++ compiler discovery and optional MSYS2/UCRT64 GCC provisioning.

ISO-Tool builds with both GNU C++ and MSVC when available. GCC is provisioned
through an official MSYS2 installation/package flow; arbitrary scripts are not
executed and downloads are kept in the user's profile Downloads cache.
"""
from __future__ import annotations
import os, shutil, subprocess
from pathlib import Path


def find_gxx() -> str | None:
    for name in ("g++.exe", "g++", "x86_64-w64-mingw32-g++.exe"):
        hit = shutil.which(name)
        if hit:
            return hit
    roots = [Path(os.environ.get("LOCALAPPDATA", "")), Path(os.environ.get("USERPROFILE", "")) / "Downloads"]
    candidates = []
    for root in roots:
        if root.exists():
            candidates.extend(root.glob("**/ucrt64/bin/g++.exe"))
            candidates.extend(root.glob("**/mingw64/bin/g++.exe"))
    return str(candidates[0]) if candidates else None


def find_msvc() -> str | None:
    for name in ("cl.exe", "cl"):
        hit = shutil.which(name)
        if hit:
            return hit
    roots = [Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Microsoft Visual Studio",
             Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio"]
    for root in roots:
        if root.exists():
            hits = list(root.rglob("Hostx64/x64/cl.exe"))
            if hits:
                return str(hits[0])
    return None


def compiler_report() -> dict:
    return {"gnu_cxx": find_gxx(), "msvc": find_msvc()}


def gcc_install_plan() -> dict:
    """Return the approved official MSYS2 UCRT64 acquisition plan.

    The caller must explicitly authorize execution. The installer/package is
    not downloaded or executed merely by importing this module.
    """
    return {
        "provider": "MSYS2",
        "environment": "UCRT64",
        "package": "mingw-w64-ucrt-x86_64-gcc",
        "command": "pacman -S --needed mingw-w64-ucrt-x86_64-gcc",
        "cache": str(Path(os.environ.get("USERPROFILE", "~")) / "Downloads" / "Chimera-II-ISO-Tool" / "dependencies" / "msys2"),
        "official": "https://www.msys2.org/",
    }


def run_gcc_version(executable: str) -> str:
    try:
        p = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=10)
        return (p.stdout or p.stderr).splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
