# Verification matrix

## Static repository verification

- [x] Three separate implementations exist: C++, C#, Python.
- [x] Independent runtime failures are converted into structured failed/skipped job results.
- [x] Failed jobs advance cumulative progress and do not abort unrelated parallel jobs.
- [x] Python GUI exposes a live details log and cumulative progress bar.
- [x] WPF GUI exposes live boot status, details and cumulative progress.
- [x] Win32 GUI exposes native live boot status/log/progress controls.
- [x] BIOS first-stage declares `ORG 0x7C00`, BIOS `INT 10h`/`INT 16h`, and `0x55AA` signature.
- [x] UEFI entry declares the EFI application contract and explicitly does not use BIOS interrupts.
- [x] `0x8000` is restricted to an explicitly configured custom-loader profile rather than presented as a UEFI standard.
- [x] Boot menu contains ordered fallback chains.
- [x] Boot validator records unavailable/invalid entries and selects the next eligible entry.
- [x] QEMU command generation is available for BIOS validation; UEFI validation requires QEMU plus OVMF configuration.
- [x] Boot validation evidence levels are documented: static, assembled, emulated, unverified.

## Tests

Run:

```text
python -m unittest discover ISO-Tool\\python\\tests -v
```

The test suite covers fail-forward execution, boot-image import, firmware-specific menu contracts, BIOS first-stage declarations, UEFI entry declarations, and fallback selection.

## BIOS assembly verification

On a machine with NASM:

```text
nasm -f bin ISO-Tool\\boot\\bios\\first_stage.asm -o first_stage.bin
```

Verify the resulting file is exactly 512 bytes and ends in bytes `55 AA`.

## Emulator verification

When QEMU is installed, use an isolated disposable image/VM to test the BIOS first stage. When OVMF is available, perform the UEFI test with the generated PE/COFF EFI application and architecture-specific EFI boot path. Capture serial/console evidence and classify the result as `emulated` only when the expected handoff evidence is observed.

If QEMU/OVMF is missing, report `unverified`; do not convert static validation into a claim of successful boot.

## Environment-dependent verification

A full Windows build still requires Visual Studio/MSVC, MASM, NASM, Python, and an ISO backend such as xorriso/xorrisofs or Oscdimg. These environment-dependent binaries are not claimed as executed merely from repository edits.
