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

The native GUI treats compilation/linking and ISO staging as one visible workflow:

`repository → dependency scan → build/compile/link → artifact collection → source staging → application staging → boot-image export → ISO mastering → validation`

During artifact collection the live log explicitly reports messages such as:

- `Adding executable: ...`
- `Adding library: ...`
- `Adding binary image: ...`
- `Added N executable/library/binary artifacts to ISO`
- `Added source tree to ISO /src`

CMake is preferred when a repository has a `CMakeLists.txt`; the native ISO-Tool Visual Studio project falls back to MSBuild when appropriate. Existing artifacts are still staged when no supported build entry point is available.

## User-selected output location

Before final ISO mastering, ISO-Tool asks the user to choose the save directory. The GUI provides a Browse control and displays the active destination; it never silently redirects the final ISO into the repository.

The default suggestion is:

```text
%USERPROFILE%\\Downloads\\Chimera-II-ISO-Tool
```

The user can select any writable directory. Generated output is organized as:

```text
<selected-output>/
├── iso/                         final ISO files
├── boot-images/                 generated BIOS/UEFI/Spit Fire images
├── binaries/
│   ├── executables/             EXE and executable binary artifacts
│   └── libraries/               DLL/LIB/A/SO artifacts
├── logs/                        build/dependency/validation logs
└── manifests/                   artifact and reproducibility manifests
```

The Python engine exposes `suggested_output_dir()`, `prepare_output_layout()` and `dependency_cache_dir()` for the same policy.

## Dependency downloads and search

Dependency discovery is separated from final ISO output. Trusted package-manager discovery may identify missing tools, but downloaded installers/packages are cached under:

```text
%USERPROFILE%\\Downloads\\Chimera-II-ISO-Tool\\dependencies
```

The native front end must show the missing dependency, trusted source/package manager and download/cache location before installation. `Scan only` performs no installation. `Install missing dependencies` requires explicit user authorization and never executes arbitrary downloaded scripts.

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
- user-selected output directory
- boot-image export and validation evidence

Microsoft documents Oscdimg support for ISO 9660, Joliet and UDF and El Torito CD/DVD boot options. The implementation therefore keeps filesystem and boot intent as explicit settings instead of assuming that every ISO is the same.

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

## Chimera II Spit Fire export and boot images

**Build Boot Image** exports `spitfire-boot.img` to the user-selected `boot-images/` directory. **Boot Image + ISO** performs both operations. If an already assembled Chimera boot artifact exists, it is preferred over a generated placeholder/export container.

All generated `.bin`, `.img`, and `.efi` boot artifacts are retained outside the source tree and are also staged into the ISO when appropriate. Optional QEMU/OVMF validation saves its evidence/logs beside the selected output. Static generation is never presented as proof that the image booted.

## Dependencies

Startup performs a dependency scan. The GUI provides two policies:

1. `Scan only`
2. `Install missing dependencies`

Automatic installation is restricted to trusted package-manager mechanisms such as Windows Package Manager/WinGet. Missing tools remain visible in the live log when they cannot be safely installed automatically. Dependency installers remain in the profile Downloads cache.

Typical dependencies include NASM, MSBuild/CMake, xorriso or Oscdimg, and QEMU for optional boot validation.

## Embedded application icon

`icons/ISO-Tool.ico` is compiled into the Windows executable through `ISO-Tool.rc`. The icon is a CD/DVD-inspired optical-media symbol and does not require an external icon file at runtime.

## Boot validation

The BIOS first-stage artifact is `boot/bios/first_stage.asm`. It is a 512-byte NASM real-mode boot sector with `ORG 0x7C00`.

The UEFI contract is `boot/uefi/entry.c`. UEFI loads a PE/COFF EFI application rather than using BIOS interrupts. QEMU and QEMU+OVMF can be used for isolated BIOS/UEFI validation when installed. Results are classified as `static`, `assembled`, `emulated`, `timeout`, or `unverified`.

## Image formats and backend options

The engine models ISO 9660, Joliet, Rock Ridge, UDF, El Torito, BIOS/MBR, GPT, UEFI/EFI System Partition, and BIOS + UEFI hybrid images. Backends include xorriso/xorrisofs and Microsoft Oscdimg when available.

## Offline inspection and import

`python/iso_tool/iso_inspect.py` performs read-only ISO inspection. The GUI can import local `.iso`, `.img`, and `.bin` files and stage bounded boot-sector data as inert input.

## Reproducibility and provenance

The generated staging directory contains `metadata/iso-tool-manifest.txt` with media type, filesystem, boot mode and hierarchy information. The build pipeline records toolchain/backend decisions in the live operation log and keeps generated artifacts outside the source tree.

## Security

ISO-Tool does not execute imported boot sectors. Third-party package/application sources remain explicit and are not silently treated as trusted. Physical-disk operations are outside this image-mastering workflow.

## Status

Windows compilation and end-to-end ISO generation remain environment-dependent. The application reports missing tools and failed backend operations in its live log rather than claiming an ISO was produced when it was not.
