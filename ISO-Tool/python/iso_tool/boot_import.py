"""Safe boot-sector/image inspection and import staging.

Imports boot artifacts as opaque files; it does not execute imported boot code.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import struct


@dataclass(frozen=True)
class BootImageInfo:
    source: str
    kind: str
    size: int
    bootable: bool
    boot_sector_sha256: str
    notes: str


def inspect_image(path: Path) -> BootImageInfo:
    path = Path(path)
    data = path.read_bytes()[:512]
    sha = hashlib.sha256(data).hexdigest()
    size = path.stat().st_size
    suffix = path.suffix.lower()
    kind = "iso" if suffix == ".iso" else "image"
    bootable = len(data) >= 512 and data[510:512] == b"\x55\xaa"
    notes = "512-byte boot sector signature detected" if bootable else "no 0x55AA signature in first 512 bytes"
    return BootImageInfo(str(path), kind, size, bootable, sha, notes)


def import_boot_sector(source: Path, destination: Path, offset: int = 0, length: int = 512) -> BootImageInfo:
    """Copy a bounded boot-sector region into the ISO-Tool staging area.

    The default imports only the first 512 bytes. The caller must explicitly
    choose any other offset/length; no imported bytes are executed automatically.
    """
    source, destination = Path(source), Path(destination)
    if offset < 0 or length <= 0 or length > 1024 * 1024:
        raise ValueError("invalid bounded boot-sector range")
    with source.open("rb") as src:
        src.seek(offset)
        data = src.read(length)
    if not data:
        raise ValueError("boot-sector range is empty")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return inspect_image(source)
