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

The progress model is event-based so GUI implementations can display global and per-stage progress without coupling UI code to build execution.

## Trust modes

`analyze`: no repository build commands.

`trusted`: user explicitly authorizes recognized build operations.

`custom`: user reviews and edits the generated plan before execution.

All process launches use argument vectors rather than shell interpolation where supported.
