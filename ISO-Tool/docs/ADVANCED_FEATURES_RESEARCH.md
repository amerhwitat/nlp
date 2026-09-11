# Advanced ISO-Tool features — research and implementation record

## Current research basis

The implementation was reviewed against current Microsoft Oscdimg documentation and open-source ISO tooling, including GNU xorriso/libisoburn and PyCdlib. Microsoft documents ISO 9660, Joliet and UDF mastering plus BIOS/UEFI El Torito multi-boot entries and boot-order files for large images. citeturn0search0

PyCdlib is a pure-Python ISO9660 reader/writer with ISO9660-1999, El Torito, Joliet, Rock Ridge and UDF support; its current PyPI release is 1.0.2 from August 2026 and requires Python 3.10+, so ISO-Tool keeps its core implementation standard-library-only and Python 3.8 compatible rather than making PyCdlib mandatory. citeturn0search2turn0search6

GNU xorriso/libisoburn provides mature ISO9660/Rock Ridge mastering and mkisofs-compatible commands. Its current documentation describes EFI El Torito `-e`, platform selection, alternate boot entries and reproducibility through `SOURCE_DATE_EPOCH` and related timestamp controls. citeturn0search1turn1search0turn1search8

## Implemented improvements

1. Python 3.8-compatible boot validation and emulator helpers.
2. Hardened El Torito validation-entry checksum checking.
3. Correct BIOS/UEFI platform interpretation for default and section entries.
4. Bounded El Torito section parsing with a catalog-entry limit.
5. Explicit xorriso EFI `-e` mastering rather than treating an EFI image as a BIOS `-b` image. citeturn1search8turn1search12
6. Reproducible xorriso timestamp flags driven by `SOURCE_DATE_EPOCH`. citeturn1search0turn1search5
7. Oscdimg BIOS+UEFI multi-boot command generation and large-image boot-order support. citeturn0search0
8. MSVC common-controls linker dependency fix.
9. Modernized .NET desktop target to net8.0-windows while retaining net48 compatibility.
10. CI coverage for Python 3.8/3.11/3.12, .NET 8/net48, MSVC x64 and JSON schemas.

## Evidence levels

- **Static**: byte-level or metadata checks performed without executing image contents.
- **Backend-ready**: a validated command line can be generated for xorriso/Oscdimg.
- **Emulated**: a disposable QEMU/OVMF run produced observable evidence.
- **Firmware-verified**: physical or firmware-equivalent boot validation was actually completed.

ISO-Tool must not report static inspection as firmware boot success.

## Next layer

- Full Rock Ridge SUSP/CE parsing and ISO directory traversal.
- GPT/protective-MBR structural validation.
- EFI System Partition FAT metadata validation.
- xorriso `-report_el_torito` and system-area report ingestion.
- QEMU+OVMF boot tests with captured serial/console evidence.
- Deterministic staging manifests, duplicate-file detection and SBOM/provenance output.
- Optional PyCdlib/libarchive read-only backends when installed.

These extensions remain separate from destructive physical-disk operations.

## Safety model

Untrusted boot code is treated as data during import and static inspection. Host execution of imported boot code is prohibited. Dynamic testing belongs inside an isolated emulator/VM with disposable storage. A missing emulator produces `unverified`, not success.
