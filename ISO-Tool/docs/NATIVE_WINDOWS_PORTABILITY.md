# Native Windows Portability: MSVC and GNU Code::Blocks

## Shared native application

ISO-Tool uses one Win32 C++ implementation with separate compiler/project descriptions:

| Environment | Project | Standard | Entry point | Required GUI libraries |
|---|---|---|---|---|
| Visual Studio / MSVC | `vcpp/ISO-Tool.sln` | C++20 | `wWinMain` | `Comctl32.lib`, `Comdlg32.lib` |
| Code::Blocks / MinGW | `codeblocks/ISO-Tool.cbp` | C++17 | `wWinMain` | `-lcomctl32`, `-lcomdlg32` |

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

`external-reference-report.json` records local include relationships, `#pragma comment(lib, ...)` references, project dependency metadata and unresolved references. `recursive-build-report.json` records every resolution, compile, link, build, skip and error decision.

## Unicode and startup

Visual Studio supplies Unicode macros through project settings. Code::Blocks supplies `-DUNICODE -D_UNICODE`. The source uses `wWinMain` and wide Windows APIs. MinGW uses `-mwindows -municode`.

## Verification boundary

CI is configured to perform actual MSVC and MinGW builds on Windows. Source edits and project XML checks alone do not prove an executable build. Repository build execution is explicit because project build scripts are arbitrary executable code.
