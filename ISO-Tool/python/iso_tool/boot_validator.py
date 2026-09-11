"""Deterministic boot-entry validation and fallback selection.

The validator is static by default. Optional QEMU/OVMF commands are only
constructed for an isolated test harness; imported boot sectors are never
executed by this module.
Python 3.8 compatible.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Tuple


@dataclass
class BootAttempt:
    entry_id: str
    status: str
    reason: str
    emulator_command: Optional[List[str]] = None


def load_menu(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_entry(entry: dict, staging: Path) -> BootAttempt:
    entry_id = str(entry.get("id", "<unnamed>"))
    artifact = entry.get("artifact") or entry.get("efi") or entry.get("image")
    if artifact:
        relative = str(artifact).lstrip("/")
        candidate = Path(staging) / relative
        if not candidate.is_file():
            return BootAttempt(entry_id, "unavailable", "required artifact missing: %s" % artifact)
    firmware = str(entry.get("firmware", "")).lower()
    if "bios" in firmware and entry.get("biosLoadAddress") != "0x7C00":
        return BootAttempt(entry_id, "invalid", "BIOS entry does not declare conventional 0x7C00 load address")
    if "uefi" in firmware and entry.get("biosInterrupts"):
        return BootAttempt(entry_id, "invalid", "UEFI entry incorrectly declares BIOS interrupts")
    return BootAttempt(entry_id, "eligible", "static validation passed")


def select_with_fallback(menu: dict, staging: Path, preferred: Optional[str] = None) -> Tuple[Optional[str], List[BootAttempt]]:
    entries: Dict[str, dict] = {str(e["id"]): e for e in menu.get("entries", []) if "id" in e}
    current = preferred or menu.get("default")
    attempts: List[BootAttempt] = []
    visited = set()
    while current and current not in visited:
        current = str(current)
        visited.add(current)
        entry = entries.get(current)
        if entry is None:
            attempts.append(BootAttempt(current, "unavailable", "menu entry not found"))
            break
        result = validate_entry(entry, staging)
        attempts.append(result)
        if result.status == "eligible":
            return current, attempts
        fallbacks = entry.get("fallback", [])
        current = next((candidate for candidate in fallbacks if candidate not in visited), None)
    return None, attempts


def qemu_bios_command(image: Path, qemu: str = "qemu-system-x86_64") -> List[str]:
    return [qemu, "-machine", "pc", "-display", "none", "-serial", "stdio", "-no-reboot", "-no-shutdown", "-drive", "format=raw,file=%s" % image]


def qemu_uefi_command(image: Path, ovmf_code: Path, qemu: str = "qemu-system-x86_64") -> List[str]:
    return [qemu, "-machine", "q35", "-display", "none", "-drive", "if=pflash,format=raw,readonly=on,file=%s" % ovmf_code, "-drive", "format=raw,file=%s" % image]


def emulator_available(qemu: str = "qemu-system-x86_64") -> bool:
    return shutil.which(qemu) is not None
