# ISO-Tool

Cross-language desktop ISO/image build orchestrator for GitHub and local repositories, with native Visual Studio 2022/MSVC, WPF/.NET, and Python front ends.

## Chimera II workspace integration

ISO-Tool has a first-class multi-repository build path for the related Chimera II workspace:

- `amerhwitat/ChimeraIIOS` — operating system, kernel, boot and system components.
- `amerhwitat/BizX` — application, commerce and wallet platform.
- `amerhwitat/BizXtreme` — game, crypto, WebGL/Three.js and application integration.

The standard profile is `engine/repository-profiles.json`. `python/build_workspace.py` acquires, recursively analyzes, builds, links, stages and masters these repositories into one workspace while preserving each source tree and build result.

## Converted Windows compiler/assembler detector

The legacy Windows batch detector is now implemented in the Python ISO-Tool engine as `python/iso_tool/toolchain_detector.py`, with `python/detect_toolchains.py` as its standalone entry point. It detects GCC/MinGW, MSVC, NASM, MASM, Go, Rust, Java, Python, LLVM/Clang, LLD, CMake, Ninja, MSBuild, Git and ISO mastering tools.

Detection checks the current `PATH`, bounded common installation locations and Visual Studio registry installation roots. It writes `manifests/windows-toolchains.json`. The detector runs **before dependency resolution/package checks**, so dependency and build planning can consume the host toolchain state. It never persists environment changes unless `--apply-user-env` is explicitly supplied. The default is process-local and side-effect free.

```text
cd ISO-Tool/python
python detect_toolchains.py --output <selected-output>\manifests\windows-toolchains.json
python detect_toolchains.py --apply-user-env
```

The converted implementation also fixes brittle batch-script behavior: PATH entries are deduplicated, `%USERPROFILE%` is expanded safely, Visual Studio installations can be discovered through the registry, and the process does not use `setx` by default (avoiding truncation and unintended persistent changes).

## Deep recursive repository scan

ISO-Tool accepts Git/GitHub repositories, generic Git URLs, direct ZIP/TAR archives, local archives, and local source directories. The acquisition engine records provenance and SHA-256 values, safely extracts archives, and never executes a discovered script merely because it exists.

The recursive scanner walks the complete source hierarchy and writes `knowledge/repository-tree.json`. Every file is classified by source language, build system, image/artifact type, documentation type, or script requiring review. Nested projects are independently considered for compilation.

## Implementations

- `vcpp/` — native Win32 C++20 / MSVC GUI.
- `dotnet/` — WPF C# implementation.
- `python/` — Python reference GUI/engine and command-line build entry points.
- `engine/` — shared JSON schemas, build profiles and repository profiles.
- `boot/` — BIOS/MBR, GPT, UEFI and El Torito integration definitions.
- `docs/` — architecture, ISO formats, toolchains, security, resilience, configuration and related-repository documentation.

## Build pipeline

`source URL/archive → acquire → toolchain detection → deep recursive tree scan → discover applications → dependency graph → deterministic build plan → registered build adapters → compile/link → artifact collection → boot-image construction → filesystem staging → bootable ISO/IMG mastering → verification`

Registered build adapters cover CMake, Make, Meson, Cargo, npm, Maven, Gradle, .NET and Autotools when their required toolchain is available. Unsupported or unavailable systems are recorded rather than treated as successful. CMake and actual build-system dependency information remain authoritative; AI/RNN/LLM planning is advisory.

Independent projects may build in parallel. Failures are isolated and recorded while unrelated jobs continue. Compiler-specific build directories are kept separate so GNU and MSVC objects/CRT assumptions are never mixed.

## Multi-repository command

From `ISO-Tool/python`:

```text
python build_workspace.py --output <selected-output> --compiler auto
```

Use `--compiler gnu` or `--compiler msvc` to force a toolchain policy, or override/add sources with `--repos ID=URL`. The resulting workspace contains repository-specific source trees and collected artifacts, followed by a combined bootable ISO/IMG build when the required backend is available.

## Spit Fire bootable ISO

The bundled `boot/bios/first_stage.asm` is the Spit Fire first-stage BIOS bootloader. The boot builder produces `boot-images/first_stage.bin`, verifies that it is exactly 512 bytes and ends in `0x55AA`, then inserts it into the ISO staging tree.

The BIOS El Torito profile passes the boot sector explicitly to xorriso/xorrisofs or Oscdimg. Therefore the generated BIOS ISO is boot-configured rather than being only a data ISO.

Generated executables, libraries and binary/EFI/image artifacts are merged into the ISO staging hierarchy under `/bin`, `/lib`, and `/boot-images` before mastering. In the multi-repository workflow, repository-specific source and artifact provenance is retained in the combined manifest.

## Optional applications and package managers

Application discovery scans `package.json`, Python packaging files, `Cargo.toml`, Maven/Gradle files, .NET projects, Go modules, CMake and Make files. It reports required/recommended/optional evidence and package managers including APT, DNF, Zypper, pacman, apk, XBPS, Portage, Homebrew, Flatpak, Snap, WinGet, Chocolatey and Scoop.

Discovery never installs packages. Installation requires explicit authorization (`--yes`) and a registered package-manager command. Arbitrary downloaded installers/scripts are never executed automatically.

## User-selected output locations

The GUI exposes separate text fields and Browse controls for build root, final ISO file, IMG file, boot-image directory, and executable/library destination.

```text
<selected-output>/
├── sources/
├── repositories/<repository>/
├── knowledge/
│   └── repository-tree.json
├── build/
├── staging/
├── iso/
├── img/
├── boot-images/
├── executables/
├── libraries/
├── manifests/
└── logs/
```

The default build root remains `%USERPROFILE%\\Downloads\\Chimera-II-ISO-Tool` on Windows.

## ISO staging hierarchy

```text
ISO root/
├── boot/{bios,uefi,spitfire}/
├── efi/boot/
├── bin/
├── lib/
├── src/{chimera-ii-os,bizx,bizxtreme}/
├── include/
├── applications/{linux,windows}/
├── tools/
├── docs/
└── metadata/
```

The complete selected source tree is preserved under `/src`; generated executables and libraries are collected under `/bin` and `/lib`, and boot artifacts under `/boot-images`.

## ISO/image generation

ISO mastering remains based on xorriso/xorrisofs and Oscdimg where available. The `.img` output produced by the GUI is an exact copy of the generated ISO image and is intentionally documented as an optical-image-compatible IMG, not a raw partitioned hard-disk image.

## Dependencies and reproducibility

Toolchain detection is a separate first-stage operation and its manifest is available to later dependency/build stages. Cached tools remain under `%USERPROFILE%\\Downloads\\Chimera-II-ISO-Tool\\dependencies`. Generated manifests record source type, URLs, hashes, build systems, compilers, detected tools, artifacts, boot images and output paths.

## Security

ISO-Tool does not execute imported boot sectors or arbitrary downloaded scripts. Package installation and network acquisition are visible operations governed by registered adapters and explicit authorization. Archive path traversal and symbolic/hard-link extraction attacks are rejected. Environment persistence from the converted detector is also opt-in.

## Verification

The detector has focused unit coverage for PATH discovery and process-local environment application. CI runs the Python deep-scan/Spit Fire fallback tests plus native Windows Visual Studio 2022 build verification. Native compiler and ISO backend availability remains environment-dependent; the tool reports missing tools and failed adapters instead of claiming an artifact exists when it does not.

See `docs/RELATED_REPOSITORY_INTEGRATION.md` for the complete Chimera II OS/BizX/BizXtreme integration contract and `docs/VERIFICATION.md` for the verification matrix.
