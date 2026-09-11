# ISO-Tool

Cross-language desktop ISO/image build orchestrator for GitHub and local repositories, with a native Visual Studio 2022/MSVC front end.

## New source acquisition workflow

The Python engine now accepts a Git/GitHub repository URL, generic Git URL, direct ZIP/TAR archive URL, local source archive, or local source directory. The GUI provides a source textbox and Browse Source control. A Git repository is cloned with submodules; archives are downloaded, SHA-256 recorded, safely extracted, and then scanned.

Archive extraction rejects absolute/traversal paths and symbolic/hard links. Downloaded source scripts are not executed merely because they exist in the archive.

## Implementations

- `vcpp/` — native Win32 C++20 / MSVC GUI.
- `dotnet/` — WPF C# implementation.
- `python/` — Python reference GUI/engine.
- `engine/` — shared JSON schemas and build profiles.
- `boot/` — BIOS/MBR, GPT, UEFI and El Torito integration definitions.
- `docs/` — architecture, ISO formats, toolchains, security, resilience and configuration documentation.

## Build pipeline

`source URL/archive → acquire → verify → scan documents/source → discover applications → dependency graph → deterministic build plan → registered build adapters → artifact collection → filesystem staging → ISO/IMG mastering → verification`

Registered build adapters cover CMake, Make, Meson, Cargo, npm, Maven, Gradle, .NET and Autotools when their required toolchain is available. Unsupported or unavailable build systems are recorded rather than treated as silently successful. CMake target/dependency ordering remains authoritative; AI/RNN/LLM planning is advisory.

## Optional applications and package managers

Application discovery scans manifests such as `package.json`, Python packaging files, `Cargo.toml`, Maven/Gradle files, .NET projects, Go modules, CMake and Make files. It produces `applications.json` with required/recommended/optional evidence and reports package managers including APT, DNF, Zypper, pacman, apk, XBPS, Portage, Homebrew, Flatpak, Snap, WinGet, Chocolatey and Scoop.

Discovery never installs packages. Installation requires explicit authorization (`--yes`) and a registered package-manager command. Arbitrary downloaded installers/scripts are never executed automatically.

## User-selected output locations

The GUI exposes separate text fields and Browse controls for build root, final ISO file, IMG file, boot-image directory, and executable/library destination. If no explicit ISO/IMG path is supplied, the selected build root is used with the normal `iso/` and `img/` layout.

```text
<selected-output>/
├── sources/
├── downloads/
├── extracted/
├── knowledge/
├── build/
├── staging/
├── iso/
├── img/
├── boot-images/
├── binaries/
│   ├── executables/
│   └── libraries/
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
├── src/
├── include/
├── applications/{linux,windows}/
├── tools/
├── docs/
└── metadata/
```

The complete selected source tree is preserved under `/src`; build artifacts are collected under `/bin` and `/lib`. Proprietary applications are not silently redistributed.

## ISO/image generation

The existing ISO mastering backends remain xorriso/xorrisofs and Oscdimg where available. ISO generation uses the staged hierarchy and explicit boot/filesystem profile. The GUI's IMG output is an ISO9660-compatible image copy when selected; raw disk-writing operations remain outside this workflow.

## Dependencies and reproducibility

Dependency discovery is separate from final output. Cached tools remain under `%USERPROFILE%\\Downloads\\Chimera-II-ISO-Tool\\dependencies`. Generated manifests record source type, URLs, hashes, build systems, compilers, artifacts and output paths.

## Security

ISO-Tool does not execute imported boot sectors or arbitrary downloaded scripts. Package installation and network acquisition are visible operations governed by registered adapters and explicit authorization.

## Status

Native compiler, package-manager, and ISO backend availability remains environment-dependent. ISO-Tool reports missing tools and failed adapters in its live log instead of claiming success without an artifact.
