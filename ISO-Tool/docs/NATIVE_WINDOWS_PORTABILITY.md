# Native Windows Portability: MSVC and GNU Code::Blocks

## Goal

ISO-Tool's native GUI has one shared Win32 C++ implementation and two supported Windows build descriptions:

| Environment | Project | Standard | Entry point | Common controls |
|---|---|---|---|---|
| Visual Studio / MSVC | `vcpp/ISO-Tool.sln` | C++20 | `wWinMain` | `Comctl32.lib` |
| Code::Blocks / MinGW | `codeblocks/ISO-Tool.cbp` | C++17 | `wWinMain` | `-lcomctl32` |

The source is deliberately conservative so the application does not require compiler-specific application logic.

## Common-controls pragma

The primary native source contains the explicit directive:

```cpp
#pragma comment(lib, "comctl32.lib")
```

MSVC consumes the directive directly. Code::Blocks/MinGW keeps the explicit `-lcomctl32` project setting, so the same source remains portable across the two toolchains.

## Unicode

Visual Studio supplies `UNICODE` and `_UNICODE` through `<CharacterSet>Unicode</CharacterSet>`. Code::Blocks supplies the equivalent `-DUNICODE -D_UNICODE` compiler definitions. The C++ source does not redefine either macro.

## Linkage

`InitCommonControlsEx` belongs to the Windows common-controls library. MSVC receives `Comctl32.lib` from both the source pragma and the `.vcxproj`; MinGW receives `comctl32` from the Code::Blocks linker configuration.

## WinMain handling

The GUI remains a Unicode `wWinMain` application. Code::Blocks/MinGW uses `-mwindows` to select the Windows subsystem and `-municode` to select the Unicode startup wrapper.

## Recursive repository build integration

The Python reference engine now provides the cross-language recursive build graph used by the application workflow. It can acquire a GitHub repository, recursively inventory source and project manifests, invoke compatible native build systems, compile direct C/C++ sources, and link a native target only when the entry-point model is unambiguous. It records managed/interpreted artifacts separately rather than producing a misleading single executable from incompatible languages.

See `RECURSIVE_REPOSITORY_BUILD.md` for the full build graph and execution model.

## Icon/branding

`../icons/ISO-Tool-logo.svg` is the vector master. It follows the existing Chimera II OS Library artwork's circular seal, infinity geometry and dark gold/cyan/violet visual language. The SVG can be converted into Windows icon sizes as part of a packaging job; no binary `.ico` is claimed until that conversion has actually been performed.

## Verification boundary

Repository XML/project validation can be performed independently, but a genuine MSVC or MinGW executable build must run on Windows with the corresponding toolchain installed. CI should report those builds separately from source-level validation. Recursive repository execution is opt-in because repository build scripts are arbitrary executable code.
