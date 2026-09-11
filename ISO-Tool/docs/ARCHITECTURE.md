# Architecture

ISO-Tool has three independent front ends—native Win32 C++, WPF C#, and Python—over a common machine-readable project model.

## Workflow entry points

1. `analyze-source`
2. `build-compiled-images`
3. `import-boot-image`
4. `build-iso`
5. `validate-image`

## Pipeline stages

1. Acquire GitHub source at an immutable ref when supplied, with connectivity recovery for network failures.
2. Accept a local repository directly when supplied.
3. Inventory source, project files, submodules and build metadata.
4. Detect local compiler/linker/assembler/image-generator capabilities from PATH, environment and Windows registry sources.
5. Present detected toolchains and versions for user selection.
6. Generate a reviewable build plan containing the selected toolchain paths.
7. Compile/assemble independent jobs subject to dependencies.
8. Link/package compatible artifacts.
9. Optionally inspect/import bounded boot-sector data from local ISO/IMG/BIN media.
10. Construct a staging filesystem tree.
11. Generate ISO/IMG through a selected capable backend.
12. Validate filesystem, boot metadata and output size.
13. Calculate SHA-256 and emit a reproducibility/build report.

## Native progress dashboard

The Windows front end exposes five visible stage bars and one overall bar:

- 🔧 Assembling
- 💾 Building boot sector
- ⚙ Compilation
- 🔗 Linking
- 🏁 Finishing up

The Python engine emits machine-readable progress events; the native frontend maps those events to the stage bars and streams child stdout/stderr into the live log. State symbols make ready/running/success/failure immediately visible.

## Toolchain discovery boundary

Environment variables and registry keys are read-only discovery sources. Detection never changes PATH, registry values, compiler installations or SDK configuration. A detected version is informational and the selected executable path is recorded in `toolchain-selection.json`.

## Fail-forward execution boundary

Independent jobs are isolated. A runtime/process failure is caught at the smallest job boundary, converted into a failed/skipped result, logged, and allowed to yield the next independent job. Progress still reaches a terminal state for that job. Fatal conditions can stop the operation when continuing would create an unsafe or invalid result.

## Offline and connectivity recovery

Local repositories require no Internet. Remote acquisition uses a connectivity monitor and retry policy. Cancellation remains supported.

## GUI details

The native resource file embeds core icon bytes and progress-stage labels. The editable Chimera II OS-inspired SVG remains in `icons/`. This keeps the executable's basic branding self-contained while preserving an editable vector source.

## Trust modes

`analyze`: no repository build commands.

`trusted`: user explicitly authorizes recognized build operations.

`custom`: user reviews and edits the generated plan before execution.

## Boot import boundary

Imported boot/image bytes are opaque input. Inspection may detect a conventional boot signature and calculate hashes. Import does not execute imported code, and only explicitly bounded regions are copied into staging.
