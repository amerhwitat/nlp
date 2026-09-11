# ISO-Tool

Cross-language desktop ISO/image build orchestrator for GitHub repositories.

## Implementations

- `vcpp/` — native Win32 C++20 / MSVC GUI.
- `dotnet/` — WPF C# targeting `net6.0-windows` and .NET Framework 4.8.
- `python/` — Python reference GUI/engine.
- `engine/` — shared JSON schemas and build profiles.
- `boot/` — MBR, GPT, UEFI and El Torito integration definitions.
- `docs/` — architecture, ISO formats, toolchains and security documentation.
- `tests/` — cross-language conformance fixtures.

## Pipeline

GitHub repository → inventory → toolchain discovery → build-plan preview → C/C++/ASM/C# compilation → boot artifact preparation → ISO staging → ISO/image backend → validation → checksum/report.

Repository build commands are **not executed silently**. The application has Analyze Only, Trusted Build and Custom Build modes.

## Toolchains

The discovery model supports local MASM (`ml`/`ml64`), NASM, MSVC/CL, MSBuild, GCC/G++, MinGW, CMake, Make and `dotnet`. Multiple locally installed versions can be enumerated and selected by the build plan.

## Image formats

The engine models ISO 9660 levels, Joliet, Rock Ridge, UDF, El Torito, BIOS/MBR, GPT, UEFI/EFI System Partition and BIOS+UEFI hybrid images. Image creation is delegated to a validated local backend such as xorriso/xorrisofs or Windows Oscdimg when available.

## Security

Builds occur in a temporary workspace with explicit authorization, structured process arguments, timeouts, cancellation, output limits and path validation. ISO-Tool is a build orchestrator, not a sandbox: users should only build repositories they trust.

## Status

The directory is designed as three independent implementations sharing machine-readable schemas. Windows compilation and end-to-end ISO generation require the corresponding local Windows toolchains and image utilities; they cannot be verified on this development environment.
