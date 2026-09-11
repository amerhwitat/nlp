"""Bounded read-only ISO9660/UDF/El Torito inspector (Python 3.8+)."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import struct
from pathlib import Path
from typing import List, Optional, Tuple

SECTOR = 2048
SYSTEM_AREA = 16 * SECTOR
MAX_DESCRIPTOR_SECTORS = 256
MAX_CATALOG_ENTRIES = 32

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
    return {0: "x86 BIOS", 1: "PowerPC", 2: "Mac", 0xEF: "EFI/UEFI"}.get(value, "platform-%02X" % value)

def _parse_descriptor(sector: int, data: bytes) -> Optional[VolumeDescriptor]:
    if len(data) != SECTOR or data[1:6] != b"CD001":
        return None
    return VolumeDescriptor(sector, data[0], data[1:6].decode("ascii", "replace"), data[6], _clean(data[40:72]), struct.unpack_from("<I", data, 132)[0], struct.unpack_from("<I", data, 158)[0], struct.unpack_from("<I", data, 166)[0], _clean(data[7:39]))

def _valid_catalog_checksum(catalog: bytes) -> bool:
    if len(catalog) < 32:
        return False
    return sum(struct.unpack("<16H", catalog[:32])) & 0xFFFF == 0

def _entry(sector: int, offset: int, raw: bytes, platform_id: int) -> BootEntry:
    return BootEntry(sector, offset, platform_id, _platform(platform_id), raw[0] == 0x88, raw[1], struct.unpack_from("<H", raw, 2)[0], raw[4], struct.unpack_from("<H", raw, 6)[0], struct.unpack_from("<I", raw, 8)[0])

def _parse_catalog(handle, sector: int, warnings: List[str]) -> List[BootEntry]:
    if sector < 16:
        warnings.append("El Torito catalog points into the system area")
        return []
    handle.seek(sector * SECTOR)
    catalog = handle.read(SECTOR)
    if len(catalog) < 64:
        warnings.append("El Torito catalog is truncated")
        return []
    if catalog[0] != 1 or catalog[30:32] != b"\x55\xAA":
        warnings.append("El Torito validation entry has an invalid signature")
        return []
    if not _valid_catalog_checksum(catalog):
        warnings.append("El Torito validation entry checksum is invalid")
    default_platform = catalog[1]
    entries: List[BootEntry] = []
    offset = 32
    while offset + 32 <= len(catalog) and len(entries) < MAX_CATALOG_ENTRIES:
        raw = catalog[offset:offset + 32]
        indicator = raw[0]
        if indicator == 0:
            break
        if indicator in (0x90, 0x91):
            platform_id = raw[1]
            count = struct.unpack_from("<H", raw, 2)[0]
            offset += 32
            for _ in range(min(count, MAX_CATALOG_ENTRIES - len(entries))):
                if offset + 32 > len(catalog):
                    warnings.append("El Torito section extends past catalog")
                    return entries
                section = catalog[offset:offset + 32]
                if section[0] in (0x88, 0x00):
                    entries.append(_entry(sector, offset, section, platform_id))
                offset += 32
            continue
        if indicator in (0x88, 0x00):
            entries.append(_entry(sector, offset, raw, default_platform))
        offset += 32
    return entries

def inspect_advanced(path: Path) -> AdvancedInspection:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    warnings: List[str] = []
    descriptors: List[VolumeDescriptor] = []
    boot_entries: List[BootEntry] = []
    iso = joliet = rock_ridge = udf = mbr = gpt = False
    size = path.stat().st_size
    with path.open("rb") as handle:
        system = handle.read(SYSTEM_AREA)
        mbr = len(system) >= 512 and system[510:512] == b"\x55\xAA"
        gpt = len(system) >= 520 and system[512:520] == b"EFI PART"
        for sector in range(16, MAX_DESCRIPTOR_SECTORS):
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
            if descriptor.type_code == 0 and data[7:39].startswith(b"EL TORITO"):
                catalog_sector = struct.unpack_from("<I", data, 71)[0]
                boot_entries.extend(_parse_catalog(handle, catalog_sector, warnings))
            if descriptor.type_code == 2:
                escape = data[88:120]
                joliet = joliet or (escape.startswith(b"%/") and escape[2:3] in (b"@", b"C", b"E"))
            if descriptor.root_extent and descriptor.root_size:
                handle.seek(descriptor.root_extent * SECTOR)
                root = handle.read(min(SECTOR, descriptor.root_size))
                rock_ridge = rock_ridge or b"SP" in root or b"RR" in root
            if descriptor.type_code == 255:
                break
    if not iso:
        warnings.append("No ISO 9660 volume descriptor found")
    if boot_entries and len({e.platform_id for e in boot_entries}) > 1:
        warnings.append("Multiple El Torito platform entries detected")
    if size % SECTOR:
        warnings.append("Image size is not a multiple of 2048 bytes")
    return AdvancedInspection(str(path), size, _hash(path), SECTOR, iso, joliet, rock_ridge, udf, mbr, gpt, tuple(descriptors), tuple(boot_entries), tuple(warnings))

def inspection_dict(path: Path) -> dict:
    return asdict(inspect_advanced(path))
