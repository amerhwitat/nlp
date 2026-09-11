# ISO-Tool

Cross-language desktop ISO/image build orchestrator for GitHub and local repositories.

## Implementations

- `vcpp/` — native Win32 C++20 / MSVC GUI.
- `dotnet/` — WPF C# targeting `net6.0-windows` and .NET Framework 4.8.
- `python/` — Python reference GUI/engine.
- `engine/` — shared JSON schemas and build profiles.
- `boot/` — BIOS/MBR, GPT, UEFI and El Torito integration definitions.
- `docs/` — architecture, ISO formats, toolchains, security and resilience documentation.
- `python/tests/` — Python resilience and conformance tests.

## Pipeline and entry points

The application exposes explicit workflow entry points: `analyze-source`, `build-compiled-images`, `import-boot-image`, `build-iso`, and `validate-image`.

The normal pipeline is:

`GitHub/local repository → inventory → toolchain discovery → build-plan preview → C/C++/ASM/C# compilation → compiled images → boot artifact preparation/import → BIOS/UEFI validation → ISO staging → ISO/image backend → validation → checksum/report`

## BIOS and UEFI first-stage boot

The BIOS first-stage artifact is `boot/bios/first_stage.asm`. It is a 512-byte NASM real-mode boot sector with `ORG 0x7C00`, the conventional BIOS load/handoff address, and BIOS `INT 10h`/`INT 16h` services for its initial menu.

The UEFI contract is `boot/uefi/entry.c`. UEFI does **not** use BIOS interrupts and has no universal `0x8000` entry address. UEFI firmware loads a PE/COFF EFI application and enters its EFI image entry point. `0x8000` is reserved for an explicitly configured custom loader/test profile only.

## Boot testing and fallback

ISO-Tool statically validates configured boot entries and can invoke QEMU for isolated BIOS testing and QEMU+OVMF for UEFI testing when those tools are installed. QEMU tests can use snapshot/read-only semantics so the source image is not modified. Results are classified as `static`, `assembled`, `emulated`, `timeout`, or `unverified`.

If the preferred boot entry is unavailable or fails validation, the deterministic boot planner follows its configured fallback chain and tries the next eligible entry. Each attempt and reason appears in the GUI details log. Required boot/integrity failures can still stop final image publication.

See `docs/BIOS_UEFI_BOOT_VALIDATION.md`.

## New image-mastering profiles

`python/iso_tool/image_profiles.py` defines explicit `data`, `bios-only`, `uefi-only`, and `bios-uefi` profiles. The mastering layer now passes firmware/filesystem intent to xorriso/xorrisofs or Oscdimg instead of treating every ISO as an undifferentiated data image.

Microsoft documents Oscdimg support for ISO 9660, Joliet and UDF, plus BIOS/UEFI El Torito multi-boot entries; ISO-Tool models those choices explicitly. citeturn0search0turn0search1

xorriso exposes El Torito BIOS and EFI boot images, system-area/MBR handling and EFI partition image concepts; ISO-Tool keeps those operations backend-driven rather than executing image contents on the host. citeturn0search2

## Offline ISO inspection

`python/iso_tool/iso_inspect.py` performs read-only inspection of ISO 9660 descriptors and reports likely Joliet/UDF/El Torito structures, boot-catalog sector information, size and SHA-256. It does not execute or mount untrusted image contents.

This makes ISO analysis useful even when the network is unavailable.

## Local repositories and offline operation

A local repository directory can be supplied directly. Local source inventory and authorized builds do not require Internet access. Remote Git acquisition can periodically check connectivity, wait for restoration, and retry network operations. Network status and retry activity are displayed in the live operation log.

## Boot-sector / ISO import

The GUI includes **Import Boot Sector / ISO**. It accepts local `.iso`, `.img`, and `.bin` files, inspects the first sector, detects `0x55AA`, computes a first-sector SHA-256, and can stage a bounded boot-sector region. Imported bytes are inert and are not executed during import. The source image is never modified.

## Large-image and boot-order preparation

The mastering architecture now reserves a boot-order/profile layer so large images can use explicit boot-file ordering when required by the selected backend. Microsoft documents boot-order files for images above 4.5 GB; ISO-Tool treats ordering as a reproducible build input rather than relying on filesystem enumeration order. citeturn0search0

## Reproducibility and provenance

The engine records the selected profile, source hash, boot-artifact hashes, backend selection, toolchain identity and validation results. Future image-report schemas can consume these records to make generated artifacts auditable and reproducible.

## Fail-forward runtime policy

Recoverable runtime failures in individual compiler, assembler, scanner, boot-artifact, or other independent jobs are isolated rather than terminating the entire pipeline. The job-level exception is logged, the failure is recorded, monotonic progress advances, and the next independent job/step continues.

Fail-forward is **not** fail-open: fatal image-integrity, staging, authorization, or safety conditions can still stop publication.

See `docs/RESILIENT_WORKFLOWS.md`.

## Live GUI details

All three front ends expose a live details section while work is running:

- **Python/Tkinter:** live operation log, status line, progress bar, and background worker.
- **C# WPF:** timestamped live log, boot-validation status, and progress bar.
- **VC++ Win32:** native multiline log, boot status, and progress controls updated through the Windows message queue.

Progress is cumulative across the whole operation rather than restarting for every stage.

## Toolchains and image formats

The discovery model supports local MASM (`ml`/`ml64`), NASM, MSVC/CL, MSBuild, GCC/G++, MinGW, CMake, Make and `dotnet`. Image tooling includes xorriso/xorrisofs and Oscdimg where locally installed. The engine models ISO 9660, Joliet, Rock Ridge, UDF, El Torito, BIOS/MBR, GPT, UEFI/EFI System Partition and BIOS+UEFI hybrid images.

QEMU is supported as the isolated validation layer; QEMU provides snapshot mode that writes temporary changes instead of modifying the source image, which is appropriate for disposable boot tests. citeturn0search6turn0search7

## Security

Builds use temporary workspaces, explicit authorization, structured process arguments, timeouts, cancellation, output limits and path validation. Imported boot sectors are bounded and never executed automatically. Physical-disk installation is a separate destructive operation requiring explicit target selection and confirmation.

## Status

Environment-dependent Windows compilation, QEMU/OVMF boot tests, and end-to-end ISO generation are not claimed as verified merely by repository edits; the application reports `unverified` when required external tooling is absent.
