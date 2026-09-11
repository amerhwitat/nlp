# Dual GNU C++ / MSVC build policy

ISO-Tool supports a two-compiler validation build on Windows:

1. **GNU C++**: prefer MSYS2 UCRT64 MinGW-w64 GCC for a native Windows GNU toolchain.
2. **Microsoft Visual C++**: use the installed Visual Studio 2022/MSVC x64 toolchain.

The UCRT64 choice is intentional: MSYS2 documents UCRT64 as its recommended environment when uncertain and notes that UCRT provides better compatibility with MSVC than the legacy MSVCRT target. The two CRT families must not be mixed within the same object/static-library boundary.

## Toolchain detection before dependency checks

`python/iso_tool/toolchain_detector.py` is the converted implementation of the legacy Windows compiler/assembler batch detector. It runs before dependency/build planning and records `manifests/windows-toolchains.json`.

Detection order is:

1. Current process `PATH` (`shutil.which`).
2. Bounded common installation hints.
3. Visual Studio registry installation roots for MSVC/MASM.

The detector covers GCC/MinGW, G++, MSVC, MASM, NASM, LLVM/Clang, LLD, Go, Rust, Java, Python, CMake, Ninja, MSBuild, Git and ISO mastering backends. Environment persistence is opt-in; normal ISO-Tool builds do not mutate the user's permanent PATH.

## Missing GCC

If `g++` is absent, ISO-Tool records an acquisition plan for the official MSYS2 UCRT64 package `mingw-w64-ucrt-x86_64-gcc`. Installation is an explicit, auditable dependency operation; ISO-Tool does not execute arbitrary scripts downloaded from search results.

Dependencies are cached beneath the user's profile Downloads directory:

`%USERPROFILE%\\Downloads\\Chimera-II-ISO-Tool\\dependencies\\msys2`

After installation, ISO-Tool discovers `ucrt64/bin/g++.exe`, records its version, and uses it for the GNU build.

## Build matrix

Each build records:

- compiler identity and version
- target architecture
- C++ standard
- compiler flags
- linker identity
- detected toolchain manifest
- output executable/library paths
- success/failure status
- source repository identifier in the combined workspace manifest

The MSVC and GNU builds are separate build trees so compiler-specific generated files and CRT assumptions cannot contaminate one another.

## Related repositories

The same compiler policy is applied independently to compatible C/C++ projects discovered recursively in ChimeraIIOS, BizX and BizXtreme. A missing GNU or MSVC toolchain does not convert that repository into a successful build; the result is recorded as unavailable/failed and independent repositories continue under the fail-forward policy.

## Compatibility selection

For the existing Windows ISO-Tool C++ sources, the default GNU target is x86_64-w64-mingw32/UCRT64. MSVC remains the authoritative native Windows build because the GUI uses Win32 APIs and MSVC-specific Windows SDK integration may be present. GNU C++ is a second independent build/compatibility validation.
