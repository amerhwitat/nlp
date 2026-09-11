# BIOS/UEFI Boot Entry Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tested first-stage boot workflow to ISO-Tool that supports a BIOS boot-sector entry at physical `0x7C00`, a standards-correct UEFI PE/COFF entry model, automatic boot-entry validation, and fallback to the next menu option when an entry is unavailable or fails validation.

**Architecture:** Keep BIOS and UEFI paths separate. BIOS uses a 512-byte real-mode boot-sector artifact whose conventional load/entry address is `0000:7C00`; UEFI does not use BIOS interrupts or a universal `0x8000` entry address, so ISO-Tool will model UEFI as an EFI application loaded by firmware at an implementation-selected address and entered through the UEFI image entry point. A deterministic validator/QEMU harness tests available entries before selecting the preferred menu entry; failures are recorded and the next eligible entry is tried.

**Tech Stack:** NASM, C/C++ boot stubs, Python orchestration, QEMU where installed, OVMF where installed, shared JSON boot profiles, C#/VC++/Python GUI front ends.

**Spec:** `ISO-Tool/docs/SMART_BOOT_AND_INSTALL.md` and `ISO-Tool/boot/menu.json`.

## Global Constraints

- BIOS first-stage boot sector targets `0x7C00` and may use BIOS interrupt services while in real mode.
- UEFI path must not claim BIOS interrupts; UEFI uses firmware boot/runtime services and a PE/COFF EFI application entry point.
- `0x8000` may be supported only as an optional test/staging address for a custom loader, never as a universal UEFI entry address.
- Imported boot code is never executed merely by importing it.
- Disk writes remain explicit and target-confirmed.
- Boot validation must run only in an isolated emulator/VM when available.
- A failed optional boot entry must fall through to the next configured eligible entry.
- Final ISO publication must still fail when required integrity/safety checks fail.

---

### Task 1: Define boot entry and fallback profiles

**Files:**
- Modify: `ISO-Tool/boot/menu.json`
- Modify: `ISO-Tool/boot/loader_catalog.json`
- Create: `ISO-Tool/boot/boot-profile.schema.json`
- Test: `ISO-Tool/python/tests/test_boot_profiles.py`

- [ ] Write tests for BIOS `0x7C00`, UEFI PE/COFF, optional custom `0x8000`, and fallback ordering.
- [ ] Implement profile schema and menu fields for validation/fallback.
- [ ] Verify JSON structure.
- [ ] Commit.

### Task 2: Add BIOS first-stage boot sector

**Files:**
- Create: `ISO-Tool/boot/bios/first_stage.asm`
- Create: `ISO-Tool/boot/bios/README.md`
- Test: `ISO-Tool/python/tests/test_bios_boot_sector.py`

- [ ] Write tests checking 512-byte size and `55 AA` signature.
- [ ] Implement a minimal NASM real-mode menu at `ORG 0x7C00` using BIOS interrupts for display/input and deterministic fallback selection.
- [ ] Keep the sector independent from imported opaque boot code.
- [ ] Commit.

### Task 3: Add UEFI entry contract

**Files:**
- Create: `ISO-Tool/boot/uefi/entry.c`
- Create: `ISO-Tool/boot/uefi/README.md`
- Test: `ISO-Tool/python/tests/test_uefi_entry_contract.py`

- [ ] Write tests verifying the source declares the UEFI image entry contract and does not use BIOS interrupt instructions.
- [ ] Implement `efi_main(EFI_HANDLE, EFI_SYSTEM_TABLE*)` as a minimal menu/validation entry point.
- [ ] Document that firmware chooses the image load address; `0x8000` is not a generic UEFI entry address.
- [ ] Commit.

### Task 4: Implement boot validation and fallback

**Files:**
- Create: `ISO-Tool/python/iso_tool/boot_validator.py`
- Test: `ISO-Tool/python/tests/test_boot_validator.py`

- [ ] Write tests for unavailable, invalid, successful, and failed entries.
- [ ] Implement deterministic artifact validation and emulator command generation.
- [ ] Implement fallback to the next eligible entry.
- [ ] Record every attempted entry and reason in a report.
- [ ] Commit.

### Task 5: Integrate the validator into image construction

**Files:**
- Modify: `ISO-Tool/python/iso_tool/pipeline.py`
- Modify: `ISO-Tool/python/iso_tool/boot_planner.py`
- Modify: `ISO-Tool/python/main.py`
- Test: `ISO-Tool/python/tests/test_pipeline_boot_validation.py`

- [ ] Write integration tests with synthetic artifacts.
- [ ] Run validation before final ISO publication when an emulator is configured.
- [ ] Emit live GUI events for attempts, failures, and fallback selection.
- [ ] Preserve fail-forward semantics.
- [ ] Commit.

### Task 6: Add C# and VC++ GUI controls

**Files:**
- Modify: `ISO-Tool/dotnet/ISO-Tool/MainWindow.xaml`
- Modify: `ISO-Tool/dotnet/ISO-Tool/MainWindow.xaml.cs`
- Modify: `ISO-Tool/vcpp/ISO-Tool.cpp`

- [ ] Add boot-profile/fallback status controls.
- [ ] Display `BIOS 0x7C00`, `UEFI EFI entry`, and optional custom `0x8000` labels distinctly.
- [ ] Stream validation attempts and fallback decisions into the existing live details pane.
- [ ] Commit.

### Task 7: Document and verify

**Files:**
- Modify: `ISO-Tool/README.md`
- Modify: `ISO-Tool/docs/SMART_BOOT_AND_INSTALL.md`
- Modify: `ISO-Tool/docs/VERIFICATION.md`
- Create: `ISO-Tool/docs/BIOS_UEFI_BOOT_VALIDATION.md`

- [ ] Document BIOS interrupt usage and `0x7C00`.
- [ ] Document the UEFI distinction and reject the premise that UEFI universally enters at `0x8000`.
- [ ] Document QEMU/OVMF validation and fallback behavior.
- [ ] Run Python unit tests and static checks available in the environment.
- [ ] Verify changed files and report exactly what was and was not environment-tested.
- [ ] Commit.
