# Architecture

ISO-Tool has three independent front ends—native Win32 C++, WPF C#, and Python—over a common machine-readable project model. The Python layer also provides the canonical multi-repository workspace entry point.

## Workflow entry points

1. `analyze-source`
2. `build-compiled-images`
3. `import-boot-image`
4. `build-iso`
5. `validate-image`
6. `build-workspace` — acquire and build the related Chimera II OS, BizX and BizXtreme repositories into one staged image.

These entry points can be invoked separately so a compiled-image workflow can feed a later ISO build.

## Related repository workspace

The standard workspace profile is `engine/repository-profiles.json`. It identifies:

- `amerhwitat/ChimeraIIOS` — OS/kernel/boot/system source;
- `amerhwitat/BizX` — application/commerce/wallet source; and
- `amerhwitat/BizXtreme` — game/crypto/WebGL/Three.js/application source.

`python/build_workspace.py` is the single command-line orchestration point. Each repository is acquired through the safe source layer, recursively analyzed, built using the registered adapters, and staged without executing imported artifacts. Source trees are preserved per repository, while compatible executables, libraries and boot/image artifacts are collected into common output areas.

## Pipeline stages

1. Acquire GitHub source at an immutable ref when supplied, with connectivity recovery for network failures.
2. Accept a local repository directly when supplied.
3. Inventory source, project files, submodules and build metadata.
4. Detect local compiler/assembler/image-generator capabilities.
5. Generate a reviewable build plan.
6. Compile/assemble independent jobs in parallel subject to dependencies.
7. Link/package boot artifacts.
8. Optionally inspect/import bounded boot-sector data from local ISO/IMG/BIN media.
9. Construct a staging filesystem tree.
10. Generate ISO/IMG through a selected capable backend.
11. Validate filesystem, boot metadata and output size.
12. Calculate SHA-256 and emit a reproducibility/build report.

The multi-repository workflow applies the same stages independently to each repository before combining their source and compatible artifacts.

## Fail-forward execution boundary

Independent jobs are isolated. A runtime/process failure is caught at the smallest job boundary, converted into a failed/skipped result, logged, and allowed to yield the next independent job. The global progress counter still advances because the job has reached a terminal state.

A failure remains visible in both the live GUI details area and the final machine-readable report. Fail-forward never means suppressing diagnostics.

Pipeline-level fatal conditions can still stop the operation when continuing would create an unsafe or invalid result, including authorization failure, uncontrolled output paths, unusable staging, invalid required boot metadata, or failed final image integrity checks.

## Offline and connectivity recovery

Local repositories require no Internet. Remote acquisition uses a connectivity monitor and retry policy. The monitor periodically checks connectivity and can wait for restoration before retrying a network operation. Retry count can be bounded or indefinite; cancellation is always supported.

## GUI progress and details

The progress model is event-based. Each front end maintains a live operation-details log containing stage/job messages, errors, network transitions, imported-image metadata and recovery actions. Progress is cumulative and monotonic across the complete operation rather than resetting for each stage.

## Trust modes

`analyze`: no repository build commands.

`trusted`: user explicitly authorizes recognized build operations.

`custom`: user reviews and edits the generated plan before execution.

All process launches use argument vectors rather than shell interpolation where supported.

## Artifact and language boundaries

Compiled executables are staged under `/bin`, libraries under `/lib`, and binary/boot/EFI images under `/boot-images`. Existing packaged APK/WebGL outputs remain artifacts. ISO-Tool never converts a binary artifact into source code in another language. Each repository retains its native C/C++, C#, Java, Python, Node.js, JavaScript, TypeScript, Unity and other source boundaries.

## Boot import boundary

Imported boot/image bytes are opaque input. Inspection may detect a conventional boot signature and calculate hashes. Import does not execute imported code, and only explicitly bounded regions are copied into staging.
