# ISO-Tool — GNU Code::Blocks / MinGW

This directory provides a GNU Code::Blocks project for the native Win32 ISO-Tool implementation.

## Requirements

- Code::Blocks with MinGW-w64/GNU GCC
- Windows SDK/MinGW Windows headers and libraries
- C++17-capable GCC
- Python 3.8+ when using the recursive repository analysis/build workflow from the application

## Build

Open `ISO-Tool.cbp` in Code::Blocks and select **Debug** or **Release**.

The project uses the shared native implementation at `../vcpp/ISO-Tool.cpp` and links `comctl32` for the Windows common-controls API.

The linker settings use `-mwindows` and `-municode` so the existing `wWinMain` entry point is handled by MinGW without changing the application architecture.

The shared C++ source explicitly contains:

```cpp
#pragma comment(lib, "comctl32.lib")
```

GCC/MinGW does not need that MSVC directive because `-lcomctl32` remains explicit in the Code::Blocks linker settings.

## Recursive repository workflow

The native GUI presents the recursive analysis/build workflow. The cross-language build engine is implemented under `../python/iso_tool/recursive_build.py` and can acquire a GitHub repository, walk it recursively, discover project manifests, compile compatible native sources, invoke project-native build systems, and collect/link compatible artifacts.

The native executable remains a Win32 frontend; language-specific build execution is intentionally delegated to the corresponding toolchain rather than attempting to turn every language into one invalid native binary.

## Compatibility model

The implementation is intentionally kept portable between MSVC and MinGW:

- Windows API calls use the wide-character `W` variants.
- The explicit `#pragma comment(lib, "comctl32.lib")` is retained in the main source for MSVC.
- Unicode macros are supplied by each build system instead of being redefined in source.
- `Comctl32.lib` is supplied by the Visual C++ project and `-lcomctl32` by Code::Blocks/MinGW.

## Output

- Debug: `bin/Debug/ISO-Tool.exe`
- Release: `bin/Release/ISO-Tool.exe`

The GUI is the same native Win32 application used by the Visual C++ solution; the build system is the only intentional difference.
