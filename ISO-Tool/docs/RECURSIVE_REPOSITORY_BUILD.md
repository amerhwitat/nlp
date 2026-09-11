# Recursive Repository Analysis, Dependency Resolution and Build

ISO-Tool treats a repository as a recursive build graph rather than a flat list of files.

## Input

The builder accepts a local Git checkout or an HTTPS/Git repository URL. Remote repositories are acquired with `git clone --recursive --depth 1`, so available Git submodules are included.

## Recursive inventory

The inventory walks the checkout while excluding generated/vendor output such as `.git`, `node_modules`, `build`, `dist`, `target`, `bin`, `obj` and Python cache/virtual-environment directories. It records source language, entry points, file sizes, project manifests and build systems.

## External-reference graph

`python/iso_tool/external_refs.py` discovers C/C++ include relationships and `#pragma comment(lib, ...)` references, and records package/build manifests. Local includes are resolved against the including file and repository root. System headers remain toolchain dependencies rather than being copied into the source tree.

The report is written as `external-reference-report.json`. Unresolved references are explicitly reported.

## Dependency resolution

When execution is enabled, project-native dependency mechanisms are used before each supported build:

- npm: `npm ci` when a lockfile exists, otherwise `npm install`;
- Cargo: `cargo fetch`;
- Go: `go mod download`;
- .NET/MSBuild: restore;
- Maven: dependency resolution;
- Gradle: dependency resolution.

CMake external projects/FetchContent remain controlled by the CMake configuration. Python environments are not implicitly modified.

## Compile and link model

Every discovered project manifest is built in its own directory. Direct C/C++ sources are compiled recursively into isolated object paths. If exactly one compatible native entry point exists, successful objects may be linked into `recursive-native`/`recursive-native.exe`. Multiple entry points are reported instead of incorrectly producing one executable.

This is intentional: a repository may contain many independent applications and libraries. ISO-Tool compiles and links each according to its build system rather than flattening unrelated targets.

For C/C++, `#pragma comment(lib, "comctl32.lib")` is an explicit native dependency. MSVC consumes it, while MinGW receives the equivalent `-lcomctl32` project/link setting. The same mechanism is used to carry the Windows common-dialog dependency needed by the ISO Save dialog.

## End-to-end ISO build

`python/build_iso.py` is the end-to-end entry point. It:

1. recursively acquires/inventories the repository;
2. resolves external dependencies;
3. compiles/assembles supported targets;
4. links compatible native artifacts;
5. writes build/dependency reports and logs;
6. stages source plus generated artifacts;
7. invokes xorriso/xorrisofs/Oscdimg with an explicit output path.

The native Windows **Build ISO…** command opens a Save dialog. The user chooses both the destination directory and the ISO filename, and that exact path is passed to `build_iso.py --output`. There is no fixed output filename.

Example:

```text
python build_iso.py C:\src\repository --output D:\Images\ChimeraII.iso
```

## Reports

Executed builds produce `recursive-build.log`, `recursive-build-report.json`, `external-reference-report.json`, isolated native objects and an `iso-staging` tree.

## Safety

Planner mode does not execute repository build commands. Execution is explicit because build scripts and package hooks are executable code. Untrusted repositories should be processed in disposable environments. ISO-Tool does not automatically execute imported boot sectors or bypass host security controls.

## Verification

Source/project changes are validated through repository/CI checks. A real MSVC, MinGW, package-manager, ISO-backend or boot-emulator success claim requires that corresponding external toolchain run to produce evidence.
