# Native Windows Portability: MSVC and GNU Code::Blocks

## Shared native application

ISO-Tool uses one Win32 C++ implementation with separate compiler/project descriptions:

| Environment | Project | Standard | Entry point | Required libraries |
|---|---|---|---|---|
| Visual Studio / MSVC | `vcpp/ISO-Tool.sln` | C++20 | `wWinMain` | `Comctl32.lib`, `Comdlg32.lib`, `Advapi32.lib` |
| Code::Blocks / MinGW | `codeblocks/ISO-Tool.cbp` | C++17 | `wWinMain` | `-lcomctl32`, `-lcomdlg32`, `-ladvapi32` |

## Professional build dashboard

The native window presents independent progress bars for 🔧 Assembling, 💾 Building boot sector, ⚙ Compilation, 🔗 Linking and 🏁 Finishing up, plus an overall progress bar and live child-process log. State emoji identify ready, running, success and failure conditions.

## Toolchain discovery and selection

`ToolchainDetection.hpp` scans executable search paths, Windows environment variables and Visual Studio registry locations. It detects C/C++ compilers, linkers and assemblers, probes a reported version, and records the discovery source.

The GUI exposes separate C/C++, linker and assembler selectors. Refresh performs a new read-only environment/registry scan. The chosen executable paths are passed to the Python build engine and recorded in `toolchain-selection.json`.

Environment variables include `VCINSTALLDIR`, `VCToolsInstallDir`, `VSINSTALLDIR`, `LLVMInstallDir`, `NASM_PREFIX`, `MINGW_HOME`, `MINGW64_HOME` and `CODEBLOCKS`. Visual Studio discovery uses the standard `SxS\\VC7` registry locations.

## Embedded resources and branding

`ISO-Tool.rc` embeds the native icon bytes and stage strings into the executable resource section. The editable Chimera II OS-inspired SVG remains in `icons/ISO-Tool-logo.svg`.

## Windows API linkage

The source explicitly links `Comctl32.lib`, `Comdlg32.lib` and `Advapi32.lib`. `Advapi32.lib` is required for read-only registry discovery. The Save dialog uses the common-dialog API. Code::Blocks/MinGW supplies the equivalent libraries explicitly.

## ISO output selection

**Build ISO…** opens a native Windows Save dialog. The user chooses both directory and filename. The selected path is handed to `python/build_iso.py --output <selected-path>` without substituting a fixed filename.

## Recursive compilation/linking

The native GUI is the frontend for the Python recursive build engine. The engine inventories the complete repository, creates an external-reference graph, resolves supported project-managed dependencies, builds each project independently, compiles direct native sources and assembly sources, and links only compatible targets. Independent applications are never flattened into one executable.

## Verification boundary

CI performs actual MSVC and MinGW build attempts on Windows, including compilation of the embedded `.rc` resource through `windres` for the MinGW job. Source edits and project XML checks alone do not prove an executable build.
