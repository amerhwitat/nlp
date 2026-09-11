# Recursive Repository Analysis and Build

ISO-Tool now treats a repository as a recursive build graph rather than a flat list of files.

## Input

The recursive builder accepts either:

- a local Git checkout; or
- an HTTPS/Git repository URL such as a GitHub repository.

Remote repositories are acquired with `git clone --recursive --depth 1`, so nested Git submodules are included when the source repository exposes them.

## Recursive inventory

The inventory walks the complete checkout while excluding generated/dependency trees such as `.git`, `node_modules`, `build`, `dist`, `target`, `bin`, `obj`, and Python cache/virtual-environment directories.

It records:

- C/C++/headers and assembly;
- Rust, Go, Java, C#, F#/VB-style managed sources where detectable;
- Python, JavaScript/TypeScript;
- CMake, Make, Meson, Cargo, Go, Maven, Gradle, MSBuild, .NET, npm and Python manifests;
- native entry points such as `main()` and `wWinMain()`;
- file sizes and relative paths.

## Build and link model

`python/iso_tool/recursive_build.py --execute` invokes every discovered supported project manifest in its own directory. It does **not** concatenate unrelated projects or languages into one invalid executable.

Native C/C++ sources are also compiled recursively when a direct build is appropriate. If a compatible native source set contains exactly one entry point, its successful object files are linked into `recursive-native` (or `recursive-native.exe` on Windows). Multiple independent entry points are reported rather than incorrectly linked together.

Python is byte-compiled with `compileall`; it is recorded as Python bytecode rather than mislabeled as a native executable. Managed, Rust, Go, Java, and JavaScript projects are built through their native project systems when those tools are installed.

Every executed build is logged to `ISO-Tool-build/recursive-build.log` and summarized in `recursive-build-report.json`.

## Examples

Inventory only:

```text
python -m iso_tool.recursive_build C:\src\repository
```

Build a local checkout:

```text
python -m iso_tool.recursive_build C:\src\repository --execute
```

Acquire and build a GitHub repository:

```text
python -m iso_tool.recursive_build https://github.com/owner/repository.git --execute
```

## Safety and reproducibility

The planner mode is the default and never executes a compiler. `--execute` is explicit because repository build scripts are arbitrary programs and may run package managers, generators, custom scripts, or other commands.

ISO-Tool records failures and unavailable toolchains instead of claiming that an artifact exists. A final ISO/IMG should therefore be created only from artifacts whose build status is `built` and whose validation stage succeeds.

## Windows native source linkage

The primary Win32 source explicitly contains:

```cpp
#pragma comment(lib, "comctl32.lib")
```

MSVC consumes this directive directly. Code::Blocks/MinGW continues to specify `comctl32` in its project linker settings, keeping the source portable.
