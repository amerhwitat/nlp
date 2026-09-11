# Native Windows Portability: MSVC and GNU Code::Blocks

## Shared native application

ISO-Tool uses one Win32 C++ implementation with separate compiler/project descriptions:

| Environment | Project | Standard | Entry point | Required GUI libraries |
|---|---|---|---|---|
| Visual Studio / MSVC | `vcpp/ISO-Tool.sln` | C++20 | `wWinMain` | `Comctl32.lib`, `Comdlg32.lib` |
| Code::Blocks / MinGW | `codeblocks/ISO-Tool.cbp` | C++17 | `wWinMain` | `-lcomctl32`, `-lcomdlg32` |

## Professional build dashboard

The native window now presents a live dashboard with independent progress bars for:

- 🔧 Assembling
- 💾 Building boot sector
- ⚙ Compilation
- 🔗 Linking
- 🏁 Finishing up

Child-process stdout/stderr is streamed into the live log. The overall status indicator uses Unicode state symbols/emoji for ready, running, success and failure states.

## Toolchain discovery and selection

`ToolchainDetection.hpp` scans executable search paths, selected Windows environment variables and Visual Studio registry locations. It detects C/C++ compilers, linkers and assemblers, probes a reported version, and records the discovery source.

The GUI exposes separate C/C++, linker and assembler selectors. A refresh performs a new environment/registry scan. The chosen executable paths are passed to the Python build engine, which records them in `toolchain-selection.json`.

The registry and environment are read-only discovery sources; ISO-Tool does not modify either one during detection.

## Embedded resources and branding

`ISO-Tool.rc` embeds the native icon bytes and stage strings into the executable resource section. This keeps the core branding available at runtime without requiring an external icon file. The vector Chimera II OS logo remains in `icons/ISO-Tool-logo.svg` for source/design use.

## Linker directives

The main source contains:

```cpp
#pragma comment(lib, "comctl32.lib")
```

MSVC consumes this directive, while project files also declare the dependency explicitly. The application uses `GetSaveFileNameW` for the user-selected ISO destination, so `Comdlg32.lib` is explicitly linked as well. MinGW does not consume MSVC linker pragmas and therefore receives both libraries through the Code::Blocks/MSYS2 linker settings.

## ISO output selection

**Build ISO…** opens a native Windows Save dialog. The user chooses both directory and filename. `OFN_OVERWRITEPROMPT` protects an existing output. The selected path is handed to `python/build_iso.py --output <selected-path>` without substituting a fixed filename.

## Recursive compilation/linking

The native GUI is the frontend for the Python recursive build engine. The engine inventories the complete repository, creates an external-reference graph, resolves supported project-managed dependencies, builds each project independently, compiles direct native sources, and links only compatible targets. Independent applications are never flattened into one executable.

## Verification boundary

CI is configured to perform actual MSVC and MinGW builds on Windows. Source edits and project XML checks alone do not prove an executable build. Repository build execution is explicit because project build scripts are arbitrary executable code.
