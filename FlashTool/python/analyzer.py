"""Offline Android artifact inspection.

The analyzer is deliberately non-destructive. It recognizes common Android
artifacts and exposes bounded metadata without applying or modifying images.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import struct
import zipfile
from typing import Dict, List, Optional, Union

ANDROID_BOOT_MAGIC = b"ANDROID!"
AVB_MAGIC = b"AVB0"
AVB_FOOTER_MAGIC = b"AVBf"
SPARSE_MAGIC = 0xED26FF3A
CRAU_MAGIC = b"CrAU"


@dataclass
class Analysis:
    path: str = ""
    kind: str = "unknown"
    size: int = 0
    sha256: str = ""
    android_magic: bool = False
    sparse: bool = False
    avb: bool = False
    ota: bool = False
    payload_version: Optional[int] = None
    manifest_size: Optional[int] = None
    manifest_signature_size: Optional[int] = None
    manifest_offset: Optional[int] = None
    sparse_block_size: Optional[int] = None
    sparse_total_blocks: Optional[int] = None
    avb_required_major: Optional[int] = None
    avb_required_minor: Optional[int] = None
    avb_algorithm: Optional[int] = None
    avb_rollback_index: Optional[int] = None
    avb_rollback_location: Optional[int] = None
    avb_descriptors_size: Optional[int] = None
    notes: Optional[List[str]] = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _crash_safe_unpack(fmt: str, data: bytes, offset: int):
    size = struct.calcsize(fmt)
    if offset < 0 or offset + size > len(data):
        return None
    return struct.unpack_from(fmt, data, offset)


def _parse_avb_header(data: bytes, result: Analysis) -> None:
    # AvbVBMetaImageHeader is big-endian and 256 bytes long.
    if len(data) < 256 or not data.startswith(AVB_MAGIC):
        return
    required = _crash_safe_unpack(">II", data, 4)
    algorithm = _crash_safe_unpack(">I", data, 16)
    descriptors_size = _crash_safe_unpack(">Q", data, 112)
    rollback = _crash_safe_unpack(">Q", data, 128)
    rollback_location = _crash_safe_unpack(">I", data, 140)
    if required:
        result.avb_required_major, result.avb_required_minor = required
    result.avb_algorithm = algorithm[0] if algorithm else None
    result.avb_descriptors_size = descriptors_size[0] if descriptors_size else None
    result.avb_rollback_index = rollback[0] if rollback else None
    result.avb_rollback_location = rollback_location[0] if rollback_location else None
    result.notes.append("bounded AVB header fields parsed; signature was not modified")


def analyze_bytes(data: bytes, path: str = "") -> Analysis:
    result = Analysis(path=path, size=len(data), sha256=hashlib.sha256(data).hexdigest())

    if data.startswith(ANDROID_BOOT_MAGIC):
        result.kind = "android-boot-family"
        result.android_magic = True
        result.notes.append("Android boot image magic detected")

    if len(data) >= 4 and struct.unpack_from("<I", data, 0)[0] == SPARSE_MAGIC:
        result.kind = "sparse"
        result.sparse = True
        header = _crash_safe_unpack("<IHHHHIIII", data, 0)
        if header:
            _, major, minor, file_hdr, chunk_hdr, block_size, total_blocks, total_chunks = header
            result.sparse_block_size = block_size
            result.sparse_total_blocks = total_blocks
            result.notes.append(
                f"sparse v{major}.{minor}, block_size={block_size}, "
                f"blocks={total_blocks}, chunks={total_chunks}, "
                f"file_header={file_hdr}, chunk_header={chunk_hdr}"
            )

    if data.startswith(CRAU_MAGIC) and len(data) >= 24:
        result.kind = "ota-payload"
        result.ota = True
        major = _crash_safe_unpack(">Q", data, 4)
        manifest_size = _crash_safe_unpack(">Q", data, 12)
        sig_size = _crash_safe_unpack(">I", data, 20) if major and major[0] >= 2 else None
        result.payload_version = major[0] if major else None
        result.manifest_size = manifest_size[0] if manifest_size else None
        result.manifest_signature_size = sig_size[0] if sig_size else None
        result.manifest_offset = 24 + (sig_size[0] if sig_size else 0)
        result.notes.append("CrAU update payload header detected; payload was not applied")

    if data.startswith(AVB_MAGIC):
        result.avb = True
        result.kind = "vbmeta"
        _parse_avb_header(data, result)
    elif AVB_FOOTER_MAGIC in data[-4096:]:
        result.avb = True
        result.notes.append("AVB footer marker detected near image tail")

    return result


def analyze_path(path: Union[str, Path]) -> Dict[str, object]:
    p = Path(path)
    data = p.read_bytes()
    result = analyze_bytes(data, str(p))
    if zipfile.is_zipfile(p):
        result.kind = "ota-zip"
        result.ota = True
        with zipfile.ZipFile(p) as zf:
            names = set(zf.namelist())
            if "payload.bin" in names:
                result.notes.append("payload.bin found in OTA ZIP")
            if "payload_properties.txt" in names:
                result.notes.append("payload_properties.txt found in OTA ZIP")
            if "META-INF/com/android/metadata" in names:
                result.notes.append("Android OTA metadata found")
    return result.to_dict()


def preflight_report(device, plan, partition_size: int = 0, image_size: int = 0) -> Dict[str, object]:
    """Return a non-destructive compatibility decision."""
    if device is None or plan is None or not plan.partition or not plan.image_path:
        return {"ok": False, "code": "invalid-input", "reason": "device and plan are required"}
    if device.transport is None or device.transport.value == "none":
        return {"ok": False, "code": "no-transport", "reason": "no supported transport"}
    if not plan.dry_run and not device.bootloader_unlocked:
        return {"ok": False, "code": "locked-write", "reason": "authorized unlocked state is required for writes"}
    if partition_size and image_size and image_size > partition_size:
        return {"ok": False, "code": "image-too-large", "reason": "image exceeds target partition"}
    return {
        "ok": True,
        "code": "ok",
        "dry_run": plan.dry_run,
        "requires_confirmation": True,
        "reason": "preflight passed; no destructive action was performed",
    }
