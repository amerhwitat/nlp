# ISO-Tool

Cross-language desktop ISO/image build orchestrator for GitHub and local repositories.

## Implementations

- `vcpp/` — native Win32 C++20 / MSVC GUI and Visual Studio solution.
- `codeblocks/` — GNU Code::Blocks / MinGW project using the same native Win32 C++ implementation.
- `dotnet/` — WPF C# targeting .NET 8 Windows and .NET Framework 4.8.
- `python/` — Python reference GUI/engine, kept Python 3.8 compatible.
- `engine/` — shared JSON schemas and build profiles.
- `boot/` — BIOS/MBR, GPT, UEFI and El Torito integration definitions.
- `icons/` — Chimera II OS-inspired deterministic vector application branding.
- `docs/` — architecture, ISO formats, toolchains, security, portability and resilience documentation.
- `python/tests/` — Python conformance and regression tests.

## Recursive whole-repository analysis and build

ISO-Tool can now acquire a GitHub/Git repository or accept an existing checkout, walk it recursively, inventory source files and build manifests, discover available toolchains, compile supported projects, and collect/link compatible native artifacts.

The implementation is `python/iso_tool/recursive_build.py`. It recognizes C/C++/assembly, Rust, Go, Java, C#, F#, Python, JavaScript/TypeScript and common build manifests including CMake, Make, Meson, Cargo, Go modules, Maven, Gradle, MSBuild, .NET, npm and Python projects.

Build execution is deliberately project-aware. Every discovered supported manifest is built in its own directory rather than concatenating unrelated projects. Direct C/C++ sources are compiled recursively; when a compatible native source set has exactly one `main()`/`wWinMain()` entry point, successful objects are linked into `recursive-native` or `recursive-native.exe`. Multiple entry points are reported instead of producing an invalid executable. Python is byte-compiled, while managed and other language projects are built by their native build systems when the corresponding toolchain is installed.

### Planner mode

```text
python -m iso_tool.recursive_build C:\src\repository
```

Planner mode inventories the whole checkout and produces no compiler/build side effects.

### Execute a local checkout

```text
python -m iso_tool.recursive_build C:\src\repository --execute
```

### Acquire and execute a GitHub repository

```text
python -m iso_tool.recursive_build https://github.com/owner/repository.git --execute
```

Executed builds write `recursive-build.log` and `recursive-build-report.json` under the build output directory. The detailed design is documented in `docs/RECURSIVE_REPOSITORY_BUILD.md`.

## Native Windows build parity

The native Win32 GUI is intentionally shared by MSVC and GNU Code::Blocks/MinGW.

### Visual C++

Open `vcpp/ISO-Tool.sln` in Visual Studio 2022 and build **Release | x64**. The main native source explicitly contains `#pragma comment(lib, "comctl32.lib")`, and the project also declares `Comctl32.lib` for deterministic MSVC linkage of `InitCommonControlsEx`.

### GNU Code::Blocks

Open `codeblocks/ISO-Tool.cbp` in Code::Blocks with MinGW-w64/GNU GCC. The project builds the same `../vcpp/ISO-Tool.cpp` source with C++17, `-mwindows`, `-municode` and `-lcomctl32`.

This keeps the application architecture identical while accommodating the different compiler/linker conventions. GCC/MinGW ignores the MSVC `#pragma comment` directive and receives the library through the project linker settings.

## Build/error fixes in the current enhancement branch

- Removed Python 3.10+ union-type syntax from boot validation/test modules so the declared Python 3.8 compatibility is real.
- Added the explicit `comctl32.lib` pragma to the main Win32 source and retained the project-level linker dependency.
- Added a Code::Blocks/MinGW project for the native GUI, including Unicode and Windows-subsystem entry-point handling.
- Added recursive repository acquisition, inventory, build-manifest discovery, native compilation and compatible native linking.
- Added regression tests for recursive traversal, manifest detection and planner mode.
- Modernized the WPF target from .NET 6 to .NET 8 while retaining net48 compatibility.
- Expanded CI to compile Python, run Python tests, build .NET 8/net48 and build the real MSVC Release x64 solution.
- Corrected xorriso EFI mastering to use the EFI El Torito `-e` path instead of treating the EFI image as a BIOS `-b` entry.
- Added El Torito validation checksums and section-platform handling.

## Chimera II OS application branding

`icons/ISO-Tool-logo.svg` is the vector master for the ISO-Tool application identity. Its geometry follows the existing Chimera II OS visual language: circular Chimera/griffin seal, infinity/boot-disc geometry, dark technical foundation, and gold/cyan/violet accents. The design direction is based on the existing Chimera II Library artwork rather than a newly generated raster image.

The SVG is suitable for documentation and web UI. Native Windows `.ico` packaging remains an explicit rasterization step so the repository does not falsely claim a binary icon has been produced when only the vector master is present.

## Advanced offline ISO inspection

`python/iso_tool/advanced_inspect.py` provides bounded, read-only analysis without mounting or executing image contents. It detects ISO 9660 descriptors, Joliet/Rock Ridge hints, UDF markers, MBR/GPT system-area markers, and El Torito BIOS/EFI entries. Malformed/truncated structures and image-alignment problems are reported as warnings.

## Reproducible mastering

`python/iso_tool/image.py` now builds explicit xorriso/xorrisofs or Oscdimg commands. Reproducible profiles use `SOURCE_DATE_EPOCH` where supported and can require deterministic boot ordering.

## Pipeline and entry points

The application exposes explicit workflow entry points: `analyze-source`, `recursive-build`, `build-compiled-images`, `import-boot-image`, `inspect-iso`, `build-iso`, and `validate-image`.

Normal pipeline:

`GitHub/local repository → recursive inventory → build-manifest/dependency discovery → toolchain discovery → build-plan preview → recursive C/C++/ASM/project compilation → compatible native linking → compiled/runtime artifacts → boot artifact preparation/import → BIOS/UEFI validation → ISO staging → ISO/image backend → advanced inspection → validation → checksum/provenance report`

## BIOS and UEFI

`boot/bios/first_stage.asm` is a 512-byte NASM real-mode boot sector with `ORG 0x7C00`, the conventional BIOS load/handoff address, and BIOS interrupt services for its initial menu.

`boot/uefi/entry.c` is a PE/COFF EFI application entry contract. UEFI does not use BIOS interrupts and has no universal `0x8000` entry address. `0x8000` remains reserved for an explicitly configured custom loader/test profile.

## Boot testing and fallback

ISO-Tool statically validates configured boot entries and can construct QEMU/OVMF commands for isolated testing when those tools are installed. Results are never promoted from `unverified` to boot success without evidence. The fallback planner follows configured alternatives when an entry is unavailable or invalid.

## Security and resilience

Imported boot sectors are treated as inert data. No automatic host execution, mounting, firmware-security bypass, or destructive physical-disk operation is performed. Dynamic testing belongs inside a disposable VM/emulator environment.

Repository build scripts are executable code. For that reason planner mode is the default and `--execute` is explicit; users should build untrusted repositories only in disposable environments.

## Verification status

The repository now contains expanded CI intended to catch the Python, .NET, MSVC and MinGW build failures described above. Repository edits themselves do not constitute a successful Windows/MSVC/MinGW/QEMU run; those results depend on the external CI/toolchain environment.
