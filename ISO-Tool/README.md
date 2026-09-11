# ISO-Tool

Cross-language desktop ISO/image build orchestrator for GitHub and local repositories.

## Implementations

- `vcpp/` — native Win32 C++20 / MSVC GUI.
- `dotnet/` — WPF C# targeting `net6.0-windows` and .NET Framework 4.8.
- `python/` — Python reference GUI/engine.
- `engine/` — shared JSON schemas and build profiles.
- `boot/` — MBR, GPT, UEFI and El Torito integration definitions.
- `docs/` — architecture, ISO formats, toolchains, security and resilience documentation.
- `python/tests/` — Python resilience and conformance tests.

## Pipeline and entry points

The application exposes explicit workflow entry points: `analyze-source`, `build-compiled-images`, `import-boot-image`, `build-iso`, and `validate-image`.

The normal pipeline is:

`GitHub/local repository → inventory → toolchain discovery → build-plan preview → C/C++/ASM/C# compilation → compiled images → boot artifact preparation/import → ISO staging → ISO/image backend → validation → checksum/report`

Repository build commands are **not executed silently**. The application has Analyze Only, Trusted Build and Custom Build modes.

## Local repositories and offline operation

A local repository directory can be supplied directly. Local source inventory and authorized builds do not require Internet access.

For remote acquisition, the resilience layer periodically checks connectivity and can wait for connectivity to return before retrying network operations. Retry count may be bounded or indefinite. Cancellation remains available. Network status and retry activity are displayed in the live operation log.

## Boot-sector / ISO import

The GUI includes **Import Boot Sector / ISO**. It accepts local `.iso`, `.img`, and `.bin` files, inspects the first sector, detects the conventional `0x55AA` signature when present, computes a first-sector SHA-256, and can stage a bounded boot-sector region. Imported bytes are inert data and are not executed during import. The source image is never modified.

## Fail-forward runtime policy

A recoverable runtime failure in an individual compiler, assembler, scanner, boot-artifact, or other independent job is isolated rather than terminating the entire pipeline. The application catches the job-level exception, writes its type/message to the live details log, records the failure, advances monotonic progress, and continues to the next independent job/step.

Fail-forward is **not** fail-open: fatal image-integrity, staging, authorization, or safety conditions can still stop publication of an image.

See `docs/RESILIENT_WORKFLOWS.md`.

## Live GUI details

All three front ends expose a details section while work is running:

- **Python/Tkinter:** live operation log, status line, progress bar, and background worker.
- **C# WPF:** timestamped live log, status text, and progress bar.
- **VC++ Win32:** native multiline log and status/progress controls updated through the Windows message queue.

Progress is cumulative across the whole operation rather than restarting for every stage.

## Toolchains

The discovery model supports local MASM (`ml`/`ml64`), NASM, MSVC/CL, MSBuild, GCC/G++, MinGW, CMake, Make and `dotnet`. Image tooling includes xorriso/xorrisofs and Oscdimg where locally installed.

## Image formats

The engine models ISO 9660 levels, Joliet, Rock Ridge, UDF, El Torito, BIOS/MBR, GPT, UEFI/EFI System Partition and BIOS+UEFI hybrid images. Image creation is delegated to a validated local backend when available.

## Security

Builds use temporary workspaces, explicit authorization, structured process arguments, timeouts, cancellation, output limits and path validation. Imported boot sectors are bounded and never executed automatically. Physical-disk installation is a separate destructive operation requiring explicit target selection and confirmation.

## Status

Windows compilation and end-to-end ISO generation require the corresponding local Windows toolchains, boot firmware/emulator and image utilities; those environment-dependent operations are not claimed as verified merely by repository edits.
