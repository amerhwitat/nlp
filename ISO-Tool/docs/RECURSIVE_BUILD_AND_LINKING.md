# Recursive build, external references and linking architecture

## Scope

ISO-Tool treats a repository as a graph of projects rather than as one giant source file. It recursively inventories the checkout, discovers project manifests and source-level references, resolves dependencies through native project tooling, builds each compatible target, and records the resulting artifacts.

## Stages

1. Acquire a local checkout or clone a Git repository with submodules.
2. Walk the complete tree while excluding generated/vendor build output.
3. Inventory languages, source files, entry points and build manifests.
4. Parse C/C++ include references and `#pragma comment(lib, ...)` library references.
5. Record package/build-system dependency metadata.
6. Resolve dependencies using native tooling: npm, Cargo, Go modules, dotnet/MSBuild, Maven and Gradle.
7. Configure/build CMake and execute other discovered build manifests in their own project directories.
8. Compile direct C/C++ sources recursively into isolated object paths.
9. Link only a compatible native target when an unambiguous entry point exists.
10. Preserve independent applications and libraries as separate artifacts.
11. Stage source and generated artifacts for ISO mastering.
12. Pass the user-selected final ISO path and filename to the ISO backend.
13. Validate the image and emit reports/logs.

## External references

`external_refs.py` creates `external-reference-report.json`. Local headers are resolved against the source directory and repository root. C/C++ library pragmas are retained as explicit link references. System headers are not copied into the repository merely because they are included; they remain toolchain dependencies.

Package-managed dependencies are not blindly compiled as unrelated source files. Their native package manager/build system is responsible for fetching/configuring the correct versions and ABI. This prevents duplicate symbols and incompatible runtime combinations.

## Native linking

For direct C/C++ compilation, all source objects are isolated under the build output. If exactly one native entry point (`main` or `wWinMain`) is detected, the successful objects can be linked into `recursive-native`/`recursive-native.exe`. If multiple entry points are present, ISO-Tool refuses to invent a single executable and reports the ambiguity.

For MSVC, `#pragma comment(lib, "comctl32.lib")` is consumed by the compiler/linker and `Comctl32.lib`/`Comdlg32.lib` are also explicit project dependencies. For GNU/MinGW, equivalent libraries are specified in Code::Blocks/project linker settings; the recursive GNU linker translates discovered pragma library names when appropriate.

## User-selected ISO output

The Win32 GUI uses the Windows Save dialog for **Build ISO…**. The complete path, including the user-selected filename, becomes the `--output` value of `build_iso.py`. The CLI exposes the same behavior with `--output <path>`. There is no fixed output filename.

## Failure isolation

A missing toolchain, failed dependency resolution, failed project, ambiguous entry point or unresolved external reference is recorded rather than silently presented as a successful executable. This is especially important for repositories containing multiple independent applications.

## Security

Executing a repository's build scripts executes repository-controlled code. Planner mode remains side-effect free; execution is explicit. Untrusted repositories should be built in a disposable environment. No imported boot code is automatically executed by ISO-Tool.
