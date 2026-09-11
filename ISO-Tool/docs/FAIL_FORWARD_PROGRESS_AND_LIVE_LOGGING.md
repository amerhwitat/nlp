# ISO-Tool — Fail-Forward Runtime Errors, Progress, and Live Details

## Purpose

ISO-Tool is designed as a long-running multi-stage build/image pipeline. A runtime failure in one job must not automatically terminate unrelated work.

The default execution rule is **fail forward**:

1. Start a job or pipeline step.
2. Catch expected process/toolchain/runtime failures at the job boundary.
3. Record the exception type and message in the operation log.
4. Mark that job as failed/skipped.
5. Advance the overall progress counter.
6. Continue with the next independent job or step.
7. Preserve all warnings/errors in the final build report.

This is not an instruction to ignore failures. It means failures are isolated, visible, and actionable while independent work continues.

## Failure classes

The engine treats these as recoverable at an individual-job boundary:

- missing executable (`OSError` / process launch failure)
- non-zero compiler/assembler/ISO-tool exit (`subprocess.SubprocessError`)
- process timeout
- malformed optional artifact
- per-file inventory/read exception
- optional boot-loader artifact failure
- individual parallel worker exception

A fatal pipeline condition can still stop the operation when continuing would make the requested image unsafe or meaningless, for example:

- output path cannot be controlled or validated
- required boot architecture has no valid boot path
- image staging directory cannot be created
- integrity/format validation proves the final image is invalid
- explicit user cancellation/stop

## GUI progress model

All three implementations expose a persistent operation-details area:

- **Python/Tkinter:** live `Text` log plus a determinate progress bar and status line.
- **C# WPF:** timestamped read-only log, status text, and determinate `ProgressBar`.
- **VC++ Win32:** native multiline log, status label, and common-controls progress bar updated from a worker thread through window messages.

The progress value is monotonic: a completed/skipped job advances the single overall bar instead of resetting it for each stage. This follows Windows progress guidance for long-running operations. citeturn0search0turn0search1

## Logging requirements

Each operation should make the following visible while it is happening:

- stage name
- job/step name
- start/completion message
- recoverable error type and message
- skipped/failed job identifier
- cumulative progress
- final status

The GUI log is a user-visible operational log, not a replacement for the machine-readable build report.

## Security boundary

Fail-forward must never become fail-open. ISO-Tool must not silently:

- execute arbitrary repository commands through a shell
- suppress compiler diagnostics
- bypass signature/security checks
- choose a physical disk automatically
- overwrite a disk after a recoverable error
- treat a failed boot validation as a successful image

Errors are bypassed only at the smallest independent job boundary. The final report must preserve the failure.

## Testing

Python regression coverage is in `python/tests/test_pipeline_resilience.py`. It verifies that a failing parallel job does not prevent a successful independent job from completing and that external command failures can be returned as structured results.

Windows builds remain environment-dependent. Visual Studio/MSBuild, .NET, NASM/MASM, GCC/Clang, ISO backends, and QEMU/OVMF must be installed on the machine where those paths are exercised.
