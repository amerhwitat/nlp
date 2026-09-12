# ISO-Tool

Cross-language desktop ISO/image build orchestrator for GitHub and local repositories, with native Visual Studio 2022/MSVC, WPF/.NET, and Python front ends.

## Python source parity

Every Python module under the ISO-Tool source root is now recursively inspected with `ast` during the build preflight. ISO-Tool generates deterministic C, C++, C#/.NET and Java parity units under the selected output's `generated/python-parity` directory and records source SHA-256 identities in `parity-manifest.json`. The generated units are a safe coverage layer: dynamic Python semantics are not silently guessed and native implementations must pass behavioral contract tests before being declared equivalent.

```text
python -m iso_tool.source_translator <python-root> --output <output>/generated/python-parity
```

See `docs/PYTHON_TO_C_CPP_CSHARP_JAVA.md` for target baselines and semantic-parity rules.

## Chimera II workspace integration

ISO-Tool has a first-class multi-repository build path for the related Chimera II workspace:

- `amerhwitat/ChimeraIIOS` — operating system, kernel, boot and system components.
- `amerhwitat/BizX` — application, commerce and wallet platform.
- `amerhwitat/BizXtreme` — game, crypto, WebGL/Three.js and application integration.

The standard profile is `engine/repository-profiles.json`. `python/build_workspace.py` acquires, recursively analyzes, builds, links, stages and masters these repositories into one workspace while preserving each source tree and build result.

## Converted Windows compiler/assembler detector

The legacy Windows batch detector is now implemented in the Python ISO-Tool engine as `python/iso_tool/toolchain_detector.py`, with `python/detect_toolchains.py` as its standalone entry point. It detects GCC/MinGW, MSVC, NASM, MASM, Go, Rust, Java, Python, LLVM/Clang, LLD, CMake, Ninja, MSBuild, Git and ISO mastering tools.

Detection checks the current `PATH`, bounded common installation locations and Visual Studio registry installation roots. It writes `manifests/windows-toolchains.json`. The detector runs **before dependency resolution/package checks**, so dependency and build planning can consume the host toolchain state. It never persists environment changes unless `--apply-user-env` is explicitly supplied. The default is process-local and side-effect free.

## Deep recursive repository scan

ISO-Tool accepts Git/GitHub repositories, generic Git URLs, direct ZIP/TAR archives, local archives, and local source directories. The acquisition engine records provenance and SHA-256 values, safely extracts archives, and never executes a discovered script merely because it exists.

The recursive scanner walks the complete source hierarchy and writes `knowledge/repository-tree.json`. Every file is classified by source language, build system, image/artifact type, documentation type, or script requiring review. Nested projects are independently considered for compilation.

## Implementations

- `vcpp/` — native Win32 C++20 / MSVC GUI.
- `dotnet/` — WPF C# implementation.
- `java/` — Java 8+ implementation.
- `python/` — Python reference GUI/engine and command-line build entry points.
- `engine/` — shared JSON schemas, build profiles and repository profiles.
- `boot/` — BIOS/MBR, GPT, UEFI and El Torito integration definitions.
- `docs/` — architecture, ISO formats, toolchains, security, resilience, configuration and language-parity documentation.

## Build pipeline

`source URL/archive → acquire → toolchain detection → dependency inventory → Python AST parity generation → deep recursive tree scan → discover applications → dependency graph → deterministic build plan → registered build adapters → compile/link → artifact collection → boot-image construction → filesystem staging → bootable ISO/IMG mastering → verification`

Registered build adapters cover CMake, Make, Meson, Cargo, npm, Maven, Gradle, .NET and Autotools when their required toolchain is available. Unsupported or unavailable systems are recorded rather than treated as successful. CMake and actual build-system dependency information remain authoritative; AI/RNN/LLM planning is advisory.

Independent projects may build in parallel. Failures are isolated and recorded while unrelated jobs continue. Compiler-specific build directories are kept separate so GNU and MSVC objects/CRT assumptions are never mixed.

## Multi-repository command

From `ISO-Tool/python`:

```text
python build_workspace.py --output <selected-output> --compiler auto
```

## Spit Fire bootable ISO

The bundled `boot/bios/first_stage.asm` is the Spit Fire first-stage BIOS bootloader. The boot builder produces `boot-images/first_stage.bin`, verifies that it is exactly 512 bytes and ends in `0x55AA`, then inserts it into the ISO staging tree.

## Dependencies and reproducibility

Toolchain and dependency detection are first-stage operations. Generated manifests record source type, URLs, hashes, build systems, compilers, detected tools, Python parity coverage, artifacts, boot images and output paths.

## Security

ISO-Tool does not execute imported boot sectors or arbitrary downloaded scripts. Package installation and network acquisition are visible operations governed by registered adapters and explicit authorization. Archive path traversal and symbolic/hard-link extraction attacks are rejected.

## Verification

CI covers Python scanning/parity metadata and native Windows build verification where the required toolchains are available. Native compiler and ISO backend availability remains environment-dependent; the tool reports missing tools and failed adapters instead of claiming an artifact exists when it does not.
