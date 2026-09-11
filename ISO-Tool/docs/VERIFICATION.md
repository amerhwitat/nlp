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
- [x] Independent runtime failures are converted into structured failed/skipped job results.
- [x] Failed jobs advance cumulative progress and do not abort unrelated parallel jobs.
- [x] Python GUI exposes a live details log and cumulative progress bar.
- [x] WPF GUI exposes timestamped live details, status, and cumulative progress.
- [x] Win32 GUI exposes a native live log/status/progress view and performs work off the UI thread.
- [x] Fail-forward behavior is documented in `docs/FAIL_FORWARD_PROGRESS_AND_LIVE_LOGGING.md`.

## Fresh verification performed for this change

The Python resilience regression tests were executed from a source-equivalent checkout of the updated pipeline and test file:

```text
python -m unittest discover -s tests -v
Ran 2 tests in 0.005s
OK
```

The two tests cover:

1. an independent parallel job raising a runtime exception while another job still completes;
2. an unavailable external command being returned as a structured failure instead of escaping from `run_safe`.

## Environment-dependent verification

A full compile and bootable-image test must run on Windows with Visual Studio/MSVC, MASM, NASM, a Python installation, and at least one ISO backend (xorriso/xorrisofs or Oscdimg). This repository operation cannot truthfully report those external Windows binaries as executed here.

Recommended commands on Windows:

```text
msbuild ISO-Tool\\vcpp\\ISO-Tool.sln /m /p:Configuration=Release /p:Platform=x64
dotnet build ISO-Tool\\dotnet\\ISO-Tool\\ISO-Tool.csproj -c Release -f net6.0-windows
python -m unittest discover ISO-Tool\\python\\tests -v
```

Then exercise the GUI against a small trusted fixture containing C, C++, ASM and an EFI/BIOS boot artifact, and verify the resulting image with the selected backend and firmware/emulator.
