# ISO-Tool

Cross-language desktop ISO/image build orchestrator for GitHub and local repositories, with a native Visual Studio 2022/MSVC front end.

## Implementations

- `vcpp/` — native Win32 C++20 / MSVC GUI, embedded application icon and Windows resource script.
- `dotnet/` — WPF C# implementation.
- `python/` — Python reference GUI/engine.
- `engine/` — shared JSON schemas and build profiles.
- `boot/` — BIOS/MBR, GPT, UEFI and El Torito integration definitions.
- `icons/` — ISO/CD/DVD application icon resources.
- `docs/` — architecture, ISO formats, toolchains, security, resilience and configuration documentation.

## Build pipeline

The native GUI now treats compilation/linking and ISO staging as one visible workflow:

`repository → dependency scan → build/compile/link → artifact collection → source staging → application staging → boot-image export → ISO mastering → validation`

During artifact collection the live log explicitly reports messages such as:

- `Adding executable: ...`
- `Adding library: ...`
- `Adding binary image: ...`
- `Added N executable/library/binary artifacts to ISO`
- `Added source tree to ISO /src`

CMake is preferred when a repository has a `CMakeLists.txt`; the native ISO-Tool Visual Studio project falls back to MSBuild when appropriate. Existing artifacts are still staged when no supported build entry point is available.

## CD and DVD profiles

The user chooses **CD** or **DVD** before mastering. The output is named `chimera-cd.iso` or `chimera-dvd.iso` by default.

The GUI exposes:

- ISO 9660
- ISO 9660 + Joliet + Rock Ridge
- UDF
- BIOS
- UEFI
- BIOS + UEFI
- dependency scan / installation policy
- output directory
- boot-image export

Microsoft documents Oscdimg support for ISO 9660, Joliet and UDF and El Torito CD/DVD boot options. The implementation therefore keeps filesystem and boot intent as explicit settings instead of assuming that every ISO is the same. See Microsoft Oscdimg documentation: https://learn.microsoft.com/windows-hardware/manufacture/desktop/oscdimg-command-line-options.

## ISO staging hierarchy

Generated staging follows an optical-image-oriented hierarchy:

```text
ISO root/
├── boot/
│   ├── bios/
│   ├── uefi/
│   └── spitfire/
├── efi/
│   └── boot/
├── bin/
├── lib/
├── src/
├── include/
├── applications/
│   ├── linux/
│   └── windows/
├── tools/
├── docs/
└── metadata/
```

The complete selected repository source is staged under `/src`. Build products are collected into `/bin` and `/lib` according to file type. Locally authorized free applications can be staged under `/applications/linux` and `/applications/windows` without automatically redistributing proprietary binaries.

## Chimera II Spit Fire export

**Build Boot Image** exports `spitfire-boot.img` to the user-selected output directory. **Boot Image + ISO** performs both operations. If an already assembled Chimera boot artifact exists, it is preferred over a generated placeholder/export container.

The ISO Tool's boot model remains compatible with BIOS/MBR and UEFI profiles and the Chimera II Spit Fire/Jasper boot architecture.

## Dependencies

Startup performs a dependency scan. The GUI provides two policies:

1. `Scan only`
2. `Install missing dependencies`

Automatic installation is restricted to trusted package-manager mechanisms such as Windows Package Manager/WinGet. Missing tools remain visible in the live log when they cannot be safely installed automatically.

Typical dependencies include NASM, MSBuild/CMake, xorriso or Oscdimg, and QEMU for optional boot validation.

## Embedded application icon

`icons/ISO-Tool.ico` is compiled into the Windows executable through `ISO-Tool.rc`. The icon is a CD/DVD-inspired optical-media symbol and does not require an external icon file at runtime.

`vcpp/resource.h` contains the resource identifier and `vcpp/ISO-Tool.rc` binds the icon into the PE application resource section.

## Boot validation

The BIOS first-stage artifact is `boot/bios/first_stage.asm`. It is a 512-byte NASM real-mode boot sector with `ORG 0x7C00`.

The UEFI contract is `boot/uefi/entry.c`. UEFI loads a PE/COFF EFI application rather than using BIOS interrupts. `0x8000` is reserved for explicitly configured custom loader/test profiles.

QEMU and QEMU+OVMF can be used for isolated BIOS/UEFI validation when installed. Results are classified as `static`, `assembled`, `emulated`, `timeout`, or `unverified`.

## Image formats and backend options

The engine models:

- ISO 9660
- Joliet
- Rock Ridge
- UDF
- El Torito
- BIOS/MBR
- GPT
- UEFI/EFI System Partition
- BIOS + UEFI hybrid images

Backends include xorriso/xorrisofs and Microsoft Oscdimg when available.

## Offline inspection and import

`python/iso_tool/iso_inspect.py` performs read-only ISO inspection. The GUI can import local `.iso`, `.img`, and `.bin` files and stage bounded boot-sector data as inert input.

## Reproducibility and provenance

The generated staging directory contains `metadata/iso-tool-manifest.txt` with media type, filesystem, boot mode and hierarchy information. The build pipeline records toolchain/backend decisions in the live operation log.

## Security

ISO-Tool does not execute imported boot sectors. Third-party package/application sources remain explicit and are not silently treated as trusted. Physical-disk operations are outside this image-mastering workflow.

## Status

Windows compilation and end-to-end ISO generation remain environment-dependent. The application reports missing tools and failed backend operations in its live log rather than claiming an ISO was produced when it was not.
