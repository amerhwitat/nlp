# Verification matrix

## Static repository verification

- [x] Three separate implementations exist: C++, C#, Python.
- [x] C++ solution uses the standard MSVC C++ project type GUID.
- [x] C# project targets .NET Framework 4.8 and .NET 6 Windows.
- [x] Python package imports its pipeline and GUI entry point.
- [x] JSON schemas define project/build-report structures.
- [x] Toolchain discovery uses executable lookup and bounded version probes.
- [x] Process execution uses argument vectors and explicit timeouts.
- [x] Parallel build primitive uses a bounded worker pool.

## Environment-dependent verification

A full compile and bootable-image test must run on Windows with Visual Studio/MSVC, MASM, NASM, a Python installation, and at least one ISO backend (xorriso/xorrisofs or Oscdimg). This repository operation cannot truthfully report those external binaries as executed here.

Recommended commands on Windows:

```text
msbuild ISO-Tool\\vcpp\\ISO-Tool.sln /m /p:Configuration=Release /p:Platform=x64
 dotnet build ISO-Tool\\dotnet\\ISO-Tool\\ISO-Tool.csproj -c Release -f net6.0-windows
 python -m unittest discover ISO-Tool\\python\\tests
```

Then exercise the GUI against a small trusted fixture containing C, C++, ASM and an EFI/BIOS boot artifact, and verify the resulting image with the selected backend and firmware/emulator.
