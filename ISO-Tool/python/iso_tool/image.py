"""ISO mastering backend selection and command construction."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import os
import shutil
import subprocess
from typing import List, Optional
from .image_profiles import get_profile


def find_backend() -> Optional[str]:
    for name in ("xorriso", "xorrisofs", "oscdimg"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _source_date_epoch() -> Optional[str]:
    value = os.environ.get("SOURCE_DATE_EPOCH")
    if value is None:
        return None
    try:
        return str(int(value))
    except ValueError:
        raise ValueError("SOURCE_DATE_EPOCH must be an integer Unix timestamp")


def build_iso_command(staging: Path, output: Path, label: str = "ISO_TOOL", profile: str = "data", backend: Optional[str] = None) -> List[str]:
    backend = backend or find_backend()
    if not backend:
        raise RuntimeError("No supported ISO backend found (xorriso/xorrisofs/oscdimg).")
    cfg = get_profile(profile)
    staging = Path(staging).resolve()
    output = Path(output).resolve()
    if not staging.is_dir():
        raise NotADirectoryError(str(staging))
    output.parent.mkdir(parents=True, exist_ok=True)
    name = Path(backend).name.lower()
    epoch = _source_date_epoch() if cfg.reproducible else None

    if name in ("xorriso", "xorrisofs"):
        cmd = [backend, "-as", "mkisofs", "-iso-level", "3", "-V", label]
        if cfg.udf_version:
            cmd += ["-udf", "-udf-version", cfg.udf_version]
        bios = staging / cfg.bios_boot if cfg.bios_boot else None
        efi = staging / cfg.uefi_boot if cfg.uefi_boot else None
        if bios and bios.is_file():
            cmd += ["-c", "boot.cat", "-b", cfg.bios_boot, "-no-emul-boot", "-boot-load-size", "4", "-boot-info-table"]
        if efi and efi.is_file():
            cmd += ["-eltorito-alt-boot", "-e", cfg.uefi_boot, "-no-emul-boot"]
        if cfg.large_image_boot_order and (staging / "boot.order").is_file():
            cmd += ["-sort", str(staging / "boot.order")]
        if epoch is not None:
            cmd += ["--modification-date=%s" % epoch, "--set_all_file_dates=%s" % epoch]
        cmd += ["-o", str(output), str(staging)]
        return cmd

    cmd = [backend, "-m", "*", "-l%s" % label, "-o", str(output)]
    bios = staging / cfg.bios_boot if cfg.bios_boot else None
    efi = staging / cfg.uefi_boot if cfg.uefi_boot else None
    if bios and bios.is_file() and efi and efi.is_file():
        cmd.insert(1, "-bootdata:2#p0,e,b%s#pEF,e,b%s" % (cfg.bios_boot, cfg.uefi_boot))
    elif bios and bios.is_file():
        cmd[1:1] = ["-b%s" % cfg.bios_boot, "-p0", "-e"]
    elif efi and efi.is_file():
        cmd[1:1] = ["-b%s" % cfg.uefi_boot, "-pEF", "-e"]
    if cfg.udf_version:
        cmd[1:1] = ["-u1", "-udfver%s" % cfg.udf_version]
    if cfg.large_image_boot_order and (staging / "boot.order").is_file():
        cmd[1:1] = ["-yo%s" % (staging / "boot.order")]
    if epoch is not None:
        stamp = datetime.fromtimestamp(int(epoch), timezone.utc).strftime("%m/%d/%Y,%H:%M:%S")
        cmd[1:1] = ["-t%s" % stamp, "-g"]
    cmd.append(str(staging))
    return cmd


def create_iso(staging: Path, output: Path, label: str = "ISO_TOOL", profile: str = "data"):
    cmd = build_iso_command(staging, output, label, profile)
    return subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3600)
