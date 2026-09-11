# Architecture

ISO-Tool has three independent front ends—native Win32 C++, WPF C#, and Python—over a common machine-readable project model.

## Pipeline stages

1. Acquire GitHub source at an immutable ref when supplied.
2. Inventory source, project files, submodules and build metadata.
3. Detect local compiler/assembler/image-generator capabilities.
4. Generate a reviewable build plan.
5. Compile/assemble independent jobs in parallel subject to dependencies.
6. Link/package boot artifacts.
7. Construct a staging filesystem tree.
8. Generate ISO/IMG through a selected capable backend.
9. Validate filesystem, boot metadata and output size.
10. Calculate SHA-256 and emit a reproducibility/build report.

## Fail-forward execution boundary

Independent jobs are isolated. A runtime/process failure is caught at the smallest job boundary, converted into a failed/skipped result, logged, and allowed to yield the next independent job. The global progress counter still advances because the job has reached a terminal state.

A failure must remain visible in both the live GUI details area and the final machine-readable report. Fail-forward never means suppressing diagnostics.

Pipeline-level fatal conditions can still stop the operation when continuing would create an unsafe or invalid result, including authorization failure, uncontrolled output paths, unusable staging, invalid required boot metadata, or failed final image integrity checks.

## GUI progress and details

The progress model is event-based so GUI implementations can display global progress without coupling UI code to build execution. Each front end also maintains a live operation-details log containing stage/job messages and recoverable errors. Progress is cumulative and monotonic across the complete operation rather than resetting for each stage.

## Trust modes

`analyze`: no repository build commands.

`trusted`: user explicitly authorizes recognized build operations.

`custom`: user reviews and edits the generated plan before execution.

All process launches use argument vectors rather than shell interpolation where supported.
