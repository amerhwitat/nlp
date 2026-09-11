# ISO-Tool — GNU Code::Blocks / MinGW

This directory provides a GNU Code::Blocks project for the native Win32 ISO-Tool implementation.

## Requirements

- Code::Blocks with MinGW-w64/GNU GCC
- Windows SDK/MinGW Windows headers and libraries
- C++17-capable GCC

## Build

Open `ISO-Tool.cbp` in Code::Blocks and select **Debug** or **Release**.

The project uses the shared native implementation at `../vcpp/ISO-Tool.cpp` and links `comctl32` for the Windows common-controls API.

The linker settings use `-mwindows` and `-municode` so the existing `wWinMain` entry point is handled by MinGW without changing the application architecture.

## Compatibility model

The implementation is intentionally kept portable between MSVC and MinGW:

- Windows API calls use the wide-character `W` variants.
- MSVC-only `#pragma comment` directives are guarded with `_MSC_VER`.
- Unicode macros are supplied by each build system instead of being redefined in source.
- `Comctl32.lib` is supplied by the Visual C++ project and `-lcomctl32` by Code::Blocks/MinGW.

## Output

- Debug: `bin/Debug/ISO-Tool.exe`
- Release: `bin/Release/ISO-Tool.exe`

The GUI is the same native Win32 application used by the Visual C++ solution; the build system is the only intentional difference.
