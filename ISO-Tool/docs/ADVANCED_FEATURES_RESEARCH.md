# Advanced ISO-Tool features — research and implementation record

## Research basis

The current implementation was expanded after reviewing current documentation for Microsoft Oscdimg, xorriso and QEMU.

### Bootable multi-firmware mastering

Microsoft documents BIOS/UEFI multi-boot El Torito images using separate platform IDs and explicit boot entries. Oscdimg supports ISO 9660, Joliet and UDF and exposes boot-order controls for large images. citeturn0search0turn0search1

ISO-Tool therefore adds explicit image profiles instead of a single generic ISO mode:

- `bios-only`
- `uefi-only`
- `bios-uefi`
- `data`

### El Torito and system-area awareness

xorriso documents BIOS El Torito boot images, EFI boot images, MBR/system-area handling and EFI partition image extraction. ISO-Tool adds offline descriptor inspection and records whether an ISO appears to contain ISO 9660, Joliet, UDF and El Torito structures. citeturn0search2

The inspector is deliberately conservative. It does not claim that a detected descriptor proves that the firmware will boot the image.

### Disposable firmware testing

QEMU documents snapshot mode for protecting disk images from write-back and supports CD-ROM/disk image attachment. ISO-Tool uses this model for disposable boot validation and distinguishes `emulated` from `unverified`. citeturn0search6turn0search7

## Features added

1. Declarative image profiles.
2. BIOS+UEFI mastering intent.
3. Offline ISO descriptor inspection.
4. El Torito detection and boot-catalog reporting.
5. SHA-256 provenance for inspected images.
6. Resilient Git acquisition with connectivity-aware retry.
7. Explicit large-image boot-order preparation architecture.
8. QEMU snapshot-oriented validation guidance.
9. Regression tests for new image features.
10. Documentation of evidence levels so static detection is never confused with a real firmware boot test.

## Planned next layer

The next safe extensions are:

- complete El Torito catalog parser and extraction of every BIOS/UEFI entry;
- GPT/protective-MBR structural parser;
- EFI System Partition FAT validation;
- xorriso `-report_el_torito`/system-area report ingestion;
- QEMU+OVMF execution with captured serial/console evidence;
- boot-order generation for large images;
- SBOM/provenance report generation;
- deterministic staging manifests and duplicate-file detection;
- optional ISO mount/read-only inspection backends;
- cross-compiler artifact compatibility matrix;
- automated QEMU boot regression fixtures.

These are intentionally separated from destructive physical-disk operations.

## Safety model

Untrusted boot code is treated as data during import and static inspection. Host execution of imported boot code is prohibited. Dynamic testing belongs inside an isolated emulator/VM with disposable storage. A missing emulator produces `unverified`, not success.
