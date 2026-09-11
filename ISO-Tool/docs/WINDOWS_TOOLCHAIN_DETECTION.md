# Windows Toolchain Detection and Selection

ISO-Tool now treats the local Windows build environment as a first-class build input.

## Discovery sources

The native Windows front end checks:

- `PATH` using Windows executable search.
- Build-related environment variables including `VCINSTALLDIR`, `VCToolsInstallDir`, `VSINSTALLDIR`, `LLVMInstallDir`, `NASM_PREFIX`, `MINGW_HOME`, `MINGW64_HOME`, and `CODEBLOCKS`.
- Visual Studio VC toolset registry locations under `HKLM\SOFTWARE\Microsoft\VisualStudio\SxS\VC7` and the 32-bit compatibility view.

The detector identifies C/C++ compilers, linkers and assemblers and records the executable path, family, discovery source and reported version.

## Supported families

C/C++: MSVC `cl.exe`, GNU/MinGW `g++`/`gcc`, LLVM Clang/Clang-CL.

Linkers: MSVC `link.exe`, LLVM `lld-link.exe`, GNU `ld.exe` and compatible LLVM/GNU alternatives when present.

Assemblers: MASM `ml.exe`/`ml64.exe`, NASM, YASM, LLVM MC and GNU `as`.

## User selection

The native GUI presents three independent selectors:

1. C/C++ compiler
2. Linker
3. Assembler

Each entry displays the detected executable, version and discovery source. Refresh re-scans the environment and registry. The selected paths are passed to the recursive build engine and persisted in `toolchain-selection.json`.

ISO-Tool does not silently modify the registry or PATH. It only reads these locations for discovery.

## Progress dashboard

The GUI reports machine-readable progress from the Python build engine and maps it to five visible stages:

- 🔧 Assembling
- 💾 Building boot sector
- ⚙ Compilation
- 🔗 Linking
- 🏁 Finishing up

A live log shows the underlying build output. A failure leaves the report and log available for diagnosis rather than being presented as a successful build.

## Resource branding

The Windows resource file embeds the ISO-Tool icon bytes and stage labels so the executable remains branded without requiring an external image at runtime. The Chimera II OS SVG logo remains available in `icons/` as the source artwork.

## Compatibility boundary

A selected linker is used only when it is explicitly supplied to the native direct-build path. Project-native manifests (Visual Studio, CMake, Make, etc.) remain authoritative for their own ABI, runtime and library requirements. ISO-Tool never combines unrelated application entry points into one executable merely because they occur in the same repository.
