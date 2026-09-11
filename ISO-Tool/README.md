# ISO-Tool

Cross-language desktop ISO/image build orchestrator for GitHub and local repositories.

## Implementations

- `vcpp/` — native Win32 C++20 / MSVC GUI.
- `dotnet/` — WPF C# targeting .NET 8 Windows and .NET Framework 4.8.
- `python/` — Python reference GUI/engine, kept Python 3.8 compatible.
- `engine/` — shared JSON schemas and build profiles.
- `boot/` — BIOS/MBR, GPT, UEFI and El Torito integration definitions.
- `docs/` — architecture, ISO formats, toolchains, security and resilience documentation.
- `python/tests/` — Python conformance and regression tests.

## Build/error fixes in the current enhancement branch

- Removed Python 3.10+ union-type syntax from boot validation/test modules so the declared Python 3.8 compatibility is real.
- Added the MSVC `Comctl32.lib` dependency required by the native progress-control implementation.
- Modernized the WPF target from .NET 6 to .NET 8 while retaining net48 compatibility.
- Expanded CI to compile Python, run Python tests, build .NET 8/net48 and build the real MSVC Release x64 solution.
- Corrected xorriso EFI mastering to use the EFI El Torito `-e` path instead of treating the EFI image as a BIOS `-b` entry.
- Added El Torito validation checksums and section-platform handling.

## Advanced offline ISO inspection

`python/iso_tool/advanced_inspect.py` provides bounded, read-only analysis without mounting or executing image contents. It detects ISO 9660 descriptors, Joliet/Rock Ridge hints, UDF markers, MBR/GPT system-area markers, and El Torito BIOS/EFI entries. Malformed/truncated structures and image-alignment problems are reported as warnings.

## Reproducible mastering

`python/iso_tool/image.py` now builds explicit xorriso/xorrisofs or Oscdimg commands. Reproducible profiles use `SOURCE_DATE_EPOCH` where supported and can require deterministic boot ordering. xorriso documents `SOURCE_DATE_EPOCH` and file-date controls for reproducible ISO output. citeturn1search0turn1search5

Microsoft documents Oscdimg support for ISO 9660, Joliet and UDF, BIOS/UEFI El Torito multi-boot entries, and explicit boot-order files for large images. citeturn0search0

PyCdlib is tracked as an optional reference/backend because it supports ISO9660, El Torito, Joliet, Rock Ridge and UDF; its current release requires Python 3.10+, so it is not a mandatory dependency of the Python 3.8-compatible core. citeturn0search2turn0search6

## Pipeline and entry points

The application exposes explicit workflow entry points: `analyze-source`, `build-compiled-images`, `import-boot-image`, `inspect-iso`, `build-iso`, and `validate-image`.

Normal pipeline:

`GitHub/local repository → inventory → toolchain discovery → build-plan preview → C/C++/ASM/C# compilation → compiled images → boot artifact preparation/import → BIOS/UEFI validation → ISO staging → ISO/image backend → advanced inspection → validation → checksum/provenance report`

## BIOS and UEFI

`boot/bios/first_stage.asm` is a 512-byte NASM real-mode boot sector with `ORG 0x7C00`, the conventional BIOS load/handoff address, and BIOS interrupt services for its initial menu.

`boot/uefi/entry.c` is a PE/COFF EFI application entry contract. UEFI does not use BIOS interrupts and has no universal `0x8000` entry address. `0x8000` remains reserved for an explicitly configured custom loader/test profile.

## Boot testing and fallback

ISO-Tool statically validates configured boot entries and can construct QEMU/OVMF commands for isolated testing when those tools are installed. Results are never promoted from `unverified` to boot success without evidence. The fallback planner follows configured alternatives when an entry is unavailable or invalid.

## Security and resilience

Imported boot sectors are treated as inert data. No automatic host execution, mounting, firmware-security bypass, or destructive physical-disk operation is performed. Dynamic testing belongs inside a disposable VM/emulator environment.

## Verification status

The repository now contains expanded CI intended to catch the Python, .NET and MSVC build failures described above. Repository edits themselves do not constitute a successful Windows/MSVC/QEMU run; those results depend on the external CI/toolchain environment.
