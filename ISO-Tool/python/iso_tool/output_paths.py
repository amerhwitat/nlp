"""User-controlled output and per-user dependency-cache paths for ISO-Tool.

The final ISO location is always selected by the user.  The Downloads directory
is used only as the default suggestion and as the dependency download/cache root.
"""
from __future__ import annotations

import os
from pathlib import Path


def profile_downloads() -> Path:
    """Return the current user's profile Downloads directory when available."""
    home = Path.home()
    candidates = [home / "Downloads", home / "downloads"]
    if os.name == "nt":
        # USERPROFILE is preferable to a translated working directory.
        profile = os.environ.get("USERPROFILE")
        if profile:
            candidates.insert(0, Path(profile) / "Downloads")
    for path in candidates:
        if path.exists() or path.parent.exists():
            path.mkdir(parents=True, exist_ok=True)
            return path
    path = home / "Downloads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def suggested_output_dir() -> Path:
    """Default GUI suggestion; the user may change this before building."""
    return profile_downloads() / "Chimera-II-ISO-Tool"


def dependency_cache_dir() -> Path:
    """Persistent download/cache location for discovered dependencies."""
    path = profile_downloads() / "Chimera-II-ISO-Tool" / "dependencies"
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_output_layout(selected_dir: str | Path) -> dict[str, Path]:
    """Create a deterministic artifact layout beneath the user-selected folder."""
    root = Path(selected_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "root": root,
        "iso": root / "iso",
        "boot_images": root / "boot-images",
        "binaries": root / "binaries",
        "executables": root / "binaries" / "executables",
        "libraries": root / "binaries" / "libraries",
        "dependency_cache": dependency_cache_dir(),
        "logs": root / "logs",
        "manifests": root / "manifests",
    }
    for path in paths.values():
        if path != paths["dependency_cache"]:
            path.mkdir(parents=True, exist_ok=True)
    return paths
