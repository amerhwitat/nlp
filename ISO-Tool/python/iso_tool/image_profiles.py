"""Declarative ISO/image mastering profiles."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ImageProfile:
    id: str
    label: str
    filesystem: str = "iso9660+joliet"
    bios_boot: str | None = None
    uefi_boot: str | None = None
    hybrid: bool = True
    udf_version: str | None = None

PROFILES = {
    "bios-uefi": ImageProfile("bios-uefi", "BIOS + UEFI", bios_boot="boot/bios/first_stage.bin", uefi_boot="EFI/BOOT/BOOTX64.EFI", hybrid=True, udf_version="1.02"),
    "bios-only": ImageProfile("bios-only", "Legacy BIOS", bios_boot="boot/bios/first_stage.bin", hybrid=False),
    "uefi-only": ImageProfile("uefi-only", "UEFI", uefi_boot="EFI/BOOT/BOOTX64.EFI", hybrid=False, udf_version="1.02"),
    "data": ImageProfile("data", "Data ISO"),
}

def get_profile(profile_id: str) -> ImageProfile:
    try:
        return PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(f"Unknown image profile: {profile_id}") from exc
