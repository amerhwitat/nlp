# ISO-Tool — GNU Code::Blocks / MinGW

This directory provides a GNU Code::Blocks project for the native Win32 ISO-Tool implementation.

## Requirements

- Code::Blocks with MinGW-w64/GNU GCC
- Windows SDK/MinGW Windows headers and libraries
- C++17-capable GCC
- Python 3.8+ for the recursive build/ISO helper
- An ISO backend such as xorriso/xorrisofs or Oscdimg for final mastering

## Build

Open `ISO-Tool.cbp` and select **Debug** or **Release**. The project uses the shared native source `../vcpp/ISO-Tool.cpp` and links both `comctl32` and `comdlg32`.

`-mwindows` and `-municode` preserve the `wWinMain` Windows GUI entry point. The shared source explicitly contains `#pragma comment(lib, "comctl32.lib")`; GCC ignores this MSVC-specific directive, so Code::Blocks supplies the equivalent libraries through linker settings.

## User-selected ISO destination

Click **Build ISO…** in the native GUI. The Windows Save dialog lets the user choose the destination directory and filename. The exact selected path is passed to the recursive `build_iso.py` helper, so the tool does not impose a fixed ISO filename.

## Recursive dependency/build workflow

The frontend delegates repository compilation to `../python/iso_tool/recursive_build.py`. It recursively inventories sources and manifests, creates an external-reference graph, resolves project-managed dependencies through native package/build systems, compiles compatible native sources, invokes each independent project build, and records artifacts and failures.

The end-to-end helper is `../python/build_iso.py`. It stages source plus compiled artifacts and calls the ISO mastering backend with the user-selected output path.

## Linking model

Independent applications remain independent. ISO-Tool never concatenates all repository `main()` functions into one executable. Direct C/C++ sources are linked only when one compatible entry point exists. Project-defined libraries and external dependencies remain attached to their owning build target.

## Output

- Debug: `bin/Debug/ISO-Tool.exe`
- Release: `bin/Release/ISO-Tool.exe`
- Recursive reports: `external-reference-report.json`, `recursive-build-report.json`, `recursive-build.log`
- ISO staging: `<selected-name>.iso-tool-build/iso-staging/`
