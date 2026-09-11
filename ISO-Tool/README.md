# ISO-Tool

Cross-language desktop ISO/image build orchestrator for GitHub repositories.

## Implementations

- `vcpp/` — native Win32 C++20 / MSVC GUI.
- `dotnet/` — WPF C# targeting `net6.0-windows` and .NET Framework 4.8.
- `python/` — Python reference GUI/engine.
- `engine/` — shared JSON schemas and build profiles.
- `boot/` — MBR, GPT, UEFI and El Torito integration definitions.
- `docs/` — architecture, ISO formats, toolchains and security documentation.
- `python/tests/` — Python resilience and conformance tests.

## Pipeline

GitHub repository → inventory → toolchain discovery → build-plan preview → C/C++/ASM/C# compilation → boot artifact preparation → ISO staging → ISO/image backend → validation → checksum/report.

Repository build commands are **not executed silently**. The application has Analyze Only, Trusted Build and Custom Build modes.

## Fail-forward runtime policy

A runtime failure in an individual compiler, assembler, scanner, boot-artifact, or other independent job is isolated rather than terminating the entire pipeline. The application:

1. catches the job-level runtime/process exception;
2. writes the exception type and message to the live details log;
3. marks the job failed/skipped;
4. advances the single monotonic overall progress bar;
5. continues to the next independent job/step; and
6. preserves the failure in the final report.

Fail-forward is **not** fail-open: fatal image-integrity, staging, authorization, or safety conditions can still stop the pipeline.

See `docs/FAIL_FORWARD_PROGRESS_AND_LIVE_LOGGING.md`.

## Live GUI details

All three front ends expose a details section while work is running:

- **Python/Tkinter:** live operation log, status line, and progress bar updated from a worker thread.
- **C# WPF:** timestamped live log, status text, and progress bar.
- **VC++ Win32:** native multiline log and status/progress controls updated through the Windows message queue so the UI remains responsive.

Progress is cumulative across the whole operation rather than restarting for every stage. This is consistent with Microsoft guidance for lengthy operations.

## Toolchains

The discovery model supports local MASM (`ml`/`ml64`), NASM, MSVC/CL, MSBuild, GCC/G++, MinGW, CMake, Make and `dotnet`. Multiple locally installed versions can be enumerated and selected by the build plan.

## Image formats

The engine models ISO 9660 levels, Joliet, Rock Ridge, UDF, El Torito, BIOS/MBR, GPT, UEFI/EFI System Partition and BIOS+UEFI hybrid images. Image creation is delegated to a validated local backend such as xorriso/xorrisofs or Windows Oscdimg when available.

## Security

Builds occur in a temporary workspace with explicit authorization, structured process arguments, timeouts, cancellation, output limits and path validation. ISO-Tool is a build orchestrator, not a sandbox: users should only build repositories they trust.

## Status

The directory is designed as three independent implementations sharing machine-readable schemas. Windows compilation and end-to-end ISO generation require the corresponding local Windows toolchains and image utilities; they cannot be verified on this development environment.
