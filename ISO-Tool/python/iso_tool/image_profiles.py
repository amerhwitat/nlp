"""Declarative ISO/image mastering profiles."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class ImageProfile:
    id: str
    label: str
    filesystem: str = "iso9660+joliet"
    bios_boot: Optional[str] = None
    uefi_boot: Optional[str] = None
    hybrid: bool = True
    udf_version: Optional[str] = None
    reproducible: bool = True
    large_image_boot_order: bool = False
    volume_id_policy: str = "explicit"

PROFILES = {
    "bios-uefi": ImageProfile("bios-uefi", "BIOS + UEFI", bios_boot="boot/bios/first_stage.bin", uefi_boot="EFI/BOOT/BOOTX64.EFI", hybrid=True, udf_version="1.02", large_image_boot_order=True),
    "bios-only": ImageProfile("bios-only", "Legacy BIOS", bios_boot="boot/bios/first_stage.bin", hybrid=False),
    "uefi-only": ImageProfile("uefi-only", "UEFI", uefi_boot="EFI/BOOT/BOOTX64.EFI", hybrid=False, udf_version="1.02"),
    "data": ImageProfile("data", "Data ISO"),
    "udf-hybrid": ImageProfile("udf-hybrid", "ISO 9660 + UDF", filesystem="iso9660+udf", hybrid=True, udf_version="1.02"),
    "reproducible-bios-uefi": ImageProfile("reproducible-bios-uefi", "Reproducible BIOS + UEFI", bios_boot="boot/bios/first_stage.bin", uefi_boot="EFI/BOOT/BOOTX64.EFI", hybrid=True, udf_version="1.02", reproducible=True, large_image_boot_order=True),
}

def get_profile(profile_id: str) -> ImageProfile:
    try:
        return PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(f"Unknown image profile: {profile_id}") from exc
