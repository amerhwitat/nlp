"""Deterministic boot-entry validation and fallback selection.

The validator never executes imported boot sectors directly. If QEMU is available,
callers may use the generated command in an isolated VM and report its result.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import shutil
from typing import Iterable


@dataclass
class BootAttempt:
    entry_id: str
    status: str
    reason: str
    emulator_command: list[str] | None = None


def load_menu(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_entry(entry: dict, staging: Path) -> BootAttempt:
    artifact = entry.get("artifact") or entry.get("efi") or entry.get("image")
    if artifact:
        relative = artifact.lstrip("/")
        candidate = Path(staging) / relative
        if not candidate.exists():
            return BootAttempt(entry["id"], "unavailable", f"required artifact missing: {artifact}")
    if entry.get("firmware") and "bios" in entry["firmware"] and entry.get("biosLoadAddress") != "0x7C00":
        return BootAttempt(entry["id"], "invalid", "BIOS entry does not declare conventional 0x7C00 load address")
    if entry.get("firmware") and "uefi" in entry["firmware"] and entry.get("biosInterrupts"):
        # BIOS interrupts are not a UEFI service mechanism; they are permitted only on the BIOS path.
        return BootAttempt(entry["id"], "invalid", "UEFI entry incorrectly declares BIOS interrupts")
    return BootAttempt(entry["id"], "eligible", "static validation passed")


def select_with_fallback(menu: dict, staging: Path, preferred: str | None = None) -> tuple[str | None, list[BootAttempt]]:
    entries = {e["id"]: e for e in menu.get("entries", [])}
    current = preferred or menu.get("default")
    attempts: list[BootAttempt] = []
    visited: set[str] = set()
    while current and current not in visited:
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


def qemu_bios_command(image: Path, qemu: str = "qemu-system-x86_64") -> list[str]:
    return [qemu, "-machine", "pc", "-display", "none", "-serial", "stdio", "-drive", f"format=raw,file={image}"]


def qemu_uefi_command(image: Path, ovmf_code: Path, qemu: str = "qemu-system-x86_64") -> list[str]:
    return [qemu, "-machine", "q35", "-display", "none", "-drive", f"if=pflash,format=raw,readonly=on,file={ovmf_code}", "-drive", f"format=raw,file={image}"]


def emulator_available(qemu: str = "qemu-system-x86_64") -> bool:
    return shutil.which(qemu) is not None
