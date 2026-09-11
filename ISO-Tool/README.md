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

## New capabilities

### Advanced offline ISO inspection

`python/iso_tool/advanced_inspect.py` provides bounded, read-only analysis of ISO images without mounting or executing their contents. It detects ISO 9660 descriptors, Joliet and Rock Ridge hints, UDF markers, MBR/GPT system-area markers, and enumerates El Torito boot entries including BIOS and EFI platform IDs. It also reports malformed/truncated structures and image alignment warnings.

### Reproducible builds and provenance

`python/iso_tool/reproducible.py` provides SHA-256 hashing, `SOURCE_DATE_EPOCH` handling, and machine-readable build provenance. The image profile model now includes reproducibility intent, explicit volume-ID policy, UDF-hybrid images and large-image boot-order requirements.

Microsoft documents Oscdimg support for ISO 9660, Joliet and UDF, together with BIOS/UEFI El Torito multi-boot entries. citeturn0search0 UEFI specifies EFI System Partitions in El Torito no-emulation entries using platform ID `0xEF`. citeturn0search36

libarchive supports ISO 9660 with Rock Ridge/Joliet and its recent releases emphasize bounded parsing and malformed-input hardening; ISO-Tool treats libarchive as an optional inspection backend rather than bundling or executing image contents. citeturn0search1turn0search5

## Pipeline and entry points

The application exposes explicit workflow entry points: `analyze-source`, `build-compiled-images`, `import-boot-image`, `build-iso`, and `validate-image`.

The normal pipeline is:

`GitHub/local repository → inventory → toolchain discovery → build-plan preview → C/C++/ASM/C# compilation → compiled images → boot artifact preparation/import → BIOS/UEFI validation → ISO staging → ISO/image backend → advanced inspection → validation → checksum/provenance report`

## BIOS and UEFI first-stage boot

The BIOS first-stage artifact is `boot/bios/first_stage.asm`. It is a 512-byte NASM real-mode boot sector with `ORG 0x7C00`, the conventional BIOS load/handoff address, and BIOS `INT 10h`/`INT 16h` services for its initial menu.

The UEFI contract is `boot/uefi/entry.c`. UEFI does **not** use BIOS interrupts and has no universal `0x8000` entry address. UEFI firmware loads a PE/COFF EFI application and enters its EFI image entry point. `0x8000` is reserved for an explicitly configured custom loader/test profile only.

## Boot testing and fallback

ISO-Tool statically validates configured boot entries and can invoke QEMU for isolated BIOS testing and QEMU+OVMF for UEFI testing when those tools are installed. QEMU tests can use snapshot/read-only semantics so the source image is not modified. Results are classified as `static`, `assembled`, `emulated`, `timeout`, or `unverified`.

If the preferred boot entry is unavailable or fails validation, the deterministic boot planner follows its configured fallback chain and tries the next eligible entry. Each attempt and reason appears in the GUI details log. Required boot/integrity failures can still stop final image publication.

See `docs/BIOS_UEFI_BOOT_VALIDATION.md`.

## Image-mastering profiles

`python/iso_tool/image_profiles.py` defines `data`, `bios-only`, `uefi-only`, `bios-uefi`, `udf-hybrid`, and `reproducible-bios-uefi` profiles. The mastering layer passes firmware/filesystem intent to xorriso/xorrisofs or Oscdimg instead of treating every ISO as an undifferentiated data image.

For large images, profiles can require explicit boot-file ordering. Microsoft documents boot-order files for images above 4.5 GB; ISO-Tool treats ordering as a reproducible build input rather than relying on filesystem enumeration order. citeturn0search0

## Security and resilience

Builds use temporary workspaces, explicit authorization, structured process arguments, timeouts, cancellation, output limits and path validation. Imported boot sectors are bounded and never executed automatically. Physical-disk installation is a separate destructive operation requiring explicit target selection and confirmation.

All image parsing is intended to be bounded and read-only. A malformed ISO can produce validation errors, but it must not cause automatic mounting, execution or host filesystem modification.

## Status

Environment-dependent Windows compilation, QEMU/OVMF boot tests, and end-to-end ISO generation are not claimed as verified merely by repository edits; the application reports `unverified` when required external tooling is absent.
