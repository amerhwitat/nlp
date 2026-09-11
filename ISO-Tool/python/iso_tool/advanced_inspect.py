"""Bounded, read-only ISO image analyzer.

Parses ISO 9660 descriptors, Joliet/Rock Ridge hints, El Torito entries and
common system-area MBR/GPT markers without mounting or executing the image.
Python 3.8 compatible.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import struct
from pathlib import Path
from typing import List, Optional, Tuple

SECTOR = 2048
SYSTEM_AREA = 16 * SECTOR


@dataclass(frozen=True)
class VolumeDescriptor:
    sector: int
    type_code: int
    identifier: str
    version: int
    volume_id: str
    path_table_size: int
    root_extent: int
    root_size: int
    boot_system_id: str


@dataclass(frozen=True)
class BootEntry:
    catalog_sector: int
    entry_offset: int
    platform_id: int
    platform: str
    bootable: bool
    emulation: int
    load_segment: int
    system_type: int
    sector_count: int
    image_sector: int


@dataclass(frozen=True)
class AdvancedInspection:
    path: str
    size: int
    sha256: str
    sector_size: int
    iso9660: bool
    joliet: bool
    rock_ridge_hint: bool
    udf: bool
    mbr: bool
    gpt: bool
    descriptors: Tuple[VolumeDescriptor, ...]
    boot_entries: Tuple[BootEntry, ...]
    warnings: Tuple[str, ...]


def _clean(value: bytes) -> str:
    return value.decode("ascii", "replace").rstrip(" \x00")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _platform(value: int) -> str:
    return {0: "x86 BIOS", 0xEF: "EFI/UEFI"}.get(value, "platform-%02X" % value)


def _parse_descriptor(sector: int, data: bytes) -> Optional[VolumeDescriptor]:
    if len(data) != SECTOR or data[1:6] != b"CD001":
        return None
    return VolumeDescriptor(
        sector=sector,
        type_code=data[0],
        identifier=data[1:6].decode("ascii", "replace"),
        version=data[6],
        volume_id=_clean(data[40:72]),
        path_table_size=struct.unpack_from("<I", data, 132)[0],
        root_extent=struct.unpack_from("<I", data, 158)[0],
        root_size=struct.unpack_from("<I", data, 166)[0],
        boot_system_id=_clean(data[7:39]),
    )


def _parse_catalog(handle, sector: int, warnings: List[str]) -> List[BootEntry]:
    if sector < 16:
        warnings.append("El Torito catalog points into the system area")
        return []
    handle.seek(sector * SECTOR)
    catalog = handle.read(SECTOR)
    if len(catalog) < 64:
        warnings.append("El Torito catalog is truncated")
        return []
    entries: List[BootEntry] = []
    if catalog[0] != 1 or catalog[30] != 0x55 or catalog[31] != 0xAA:
        warnings.append("El Torito validation entry has an invalid signature")
        return []
    for offset in range(32, min(len(catalog), 32 + 32 * 16), 32):
        entry = catalog[offset:offset + 32]
        if len(entry) < 32 or entry[0] == 0:
            break
        if entry[0] not in (0x88, 0x00, 0x90, 0x91):
            continue
        platform_id = entry[1]
        bootable = entry[0] == 0x88
        emulation = entry[2]
        load_segment = struct.unpack_from("<H", entry, 2)[0] if emulation else 0
        system_type = entry[4]
        sector_count = struct.unpack_from("<H", entry, 6)[0]
        image_sector = struct.unpack_from("<I", entry, 8)[0]
        entries.append(BootEntry(sector, offset, platform_id, _platform(platform_id),
                                 bootable, emulation, load_segment, system_type,
                                 sector_count, image_sector))
    return entries


def inspect_advanced(path: Path) -> AdvancedInspection:
    path = Path(path)
    warnings: List[str] = []
    descriptors: List[VolumeDescriptor] = []
    boot_entries: List[BootEntry] = []
    iso = joliet = rock_ridge = udf = mbr = gpt = False
    with path.open("rb") as handle:
        system = handle.read(SYSTEM_AREA)
        if len(system) >= 512 and system[510:512] == b"\x55\xAA":
            mbr = True
        if len(system) >= 520 and system[512:520] == b"EFI PART":
            gpt = True
        for sector in range(16, 256):
            handle.seek(sector * SECTOR)
            data = handle.read(SECTOR)
            if len(data) < SECTOR:
                break
            if data[1:6] in (b"BEA01", b"NSR02", b"NSR03"):
                udf = True
            descriptor = _parse_descriptor(sector, data)
            if descriptor is None:
                continue
            iso = True
            descriptors.append(descriptor)
            if descriptor.type_code == 0:
                catalog_sector = struct.unpack_from("<I", data, 71)[0]
                if data[7:39].startswith(b"EL TORITO"):
                    boot_entries.extend(_parse_catalog(handle, catalog_sector, warnings))
            if descriptor.type_code == 2 and data[88:120].decode("utf-16-be", "ignore").startswith("%/"):
                joliet = True
            root_extent = descriptor.root_extent
            if root_extent and descriptor.root_size:
                handle.seek(root_extent * SECTOR)
                root = handle.read(min(SECTOR, descriptor.root_size))
                if b"SP" in root or b"RR" in root:
                    rock_ridge = True
            if descriptor.type_code == 255:
                break
    if not iso:
        warnings.append("No ISO 9660 volume descriptor found")
    if boot_entries and len({e.platform_id for e in boot_entries}) > 1:
        warnings.append("Multiple El Torito platform entries detected")
    if path.stat().st_size % SECTOR:
        warnings.append("Image size is not a multiple of 2048 bytes")
    return AdvancedInspection(str(path), path.stat().st_size, _hash(path), SECTOR,
                              iso, joliet, rock_ridge, udf, mbr, gpt,
                              tuple(descriptors), tuple(boot_entries), tuple(warnings))


def inspection_dict(path: Path) -> dict:
    return asdict(inspect_advanced(path))
