# ISO-Tool

Cross-language desktop ISO/image build orchestrator for GitHub and local repositories.

## Implementations

- `vcpp/` — native Win32 C++20 / MSVC GUI.
- `codeblocks/` — GNU Code::Blocks / MinGW project sharing the native source.
- `dotnet/` — WPF C# implementation.
- `python/` — Python reference engine and CLI.
- `engine/` — shared JSON schemas and build profiles.
- `boot/` — BIOS/MBR, GPT, UEFI and El Torito integration definitions.
- `icons/` — Chimera II OS-inspired deterministic vector branding.
- `docs/` — architecture, build, ISO, toolchain, security and portability documentation.

## End-to-end repository build

ISO-Tool recursively analyzes a complete local checkout or Git repository. It inventories source files and build manifests, discovers the dependency/build graph, resolves external dependencies through the project's native package/build system, compiles supported projects, and links compatible native targets.

The authoritative orchestration is `python/iso_tool/recursive_build.py`. External references are recorded by `python/iso_tool/external_refs.py` in `external-reference-report.json`. The end-to-end `python/build_iso.py` entry point then stages source plus compiled artifacts and invokes the selected ISO backend with the exact output path supplied by the user.

Supported project/build families include C/C++ and assembly, Rust, Go, Java, C#, F#, Python, JavaScript/TypeScript, CMake, Make, Meson, Cargo, Go modules, Maven, Gradle, MSBuild/.NET and npm. Dependency resolution uses project-native mechanisms such as `npm ci/install`, `cargo fetch`, `go mod download`, `dotnet restore`, MSBuild Restore, Maven dependency resolution and Gradle dependency resolution. Python environments are not mutated implicitly.

### Important linking rule

The repository is **not** flattened into one invalid executable. Independent applications remain independent build targets. Direct C/C++ sources are compiled recursively; if exactly one compatible native entry point exists, successful objects can be linked into a native executable. Multiple entry points are reported rather than incorrectly combined. Project manifests are built using their own dependency and linker model.

C/C++ `#pragma comment(lib, ...)` references are discovered as link-library references. MSVC consumes the pragma directly; GNU/MinGW receives corresponding library names through the linker command when the environment can resolve them.

## Choosing the ISO destination and filename

The native Windows **Build ISO…** command opens a Save dialog. The user chooses both the destination directory and the final filename (for example `D:\Images\ChimeraII.iso`). Existing files are protected by the overwrite prompt. The selected path is passed unchanged to the end-to-end ISO mastering entry point.

CLI users have the same control:

```text
python build_iso.py C:\src\repository --output D:\Images\MyRepository.iso
```

The output filename is never silently replaced by a fixed repository name.

## Native Windows linkage

The main Win32 source explicitly contains:

```cpp
#pragma comment(lib, "comctl32.lib")
```

and also links `Comctl32.lib` at the MSVC project level. The output Save dialog uses the Windows common-dialog API and therefore also links `Comdlg32.lib`. Code::Blocks/MinGW links `comctl32` and `comdlg32` explicitly because GCC does not consume MSVC linker pragmas.

## Recursive reports and artifacts

Each executed recursive build writes:

- `external-reference-report.json` — source includes, link-library pragmas, project dependency metadata and unresolved references.
- `recursive-build-report.json` — sources, manifests, dependency-resolution jobs, build jobs, artifacts, skips and errors.
- `recursive-build.log` — complete command/output transcript.
- `objects/` — directly compiled native object files.
- `iso-staging/` — source and compiled-artifact tree supplied to the ISO backend.

## Pipeline

`GitHub/local repository → recursive inventory → external-reference graph → toolchain discovery → dependency resolution → per-project compile/assemble → compatible native linking → artifact collection → ISO staging → user-selected ISO path/name → xorriso/xorrisofs/Oscdimg → validation → checksum/provenance report`

## Planner versus execute

Planner mode is the default and does not execute repository build scripts. Use `--execute` for recursive compilation. End-to-end ISO mastering is an explicit execution operation.

```text
python -m iso_tool.recursive_build C:\src\repository
python -m iso_tool.recursive_build C:\src\repository --execute
python build_iso.py C:\src\repository --output D:\Images\repository.iso
```

Build scripts from untrusted repositories execute code. Use a disposable VM or isolated build environment for untrusted source.

## BIOS and UEFI

`boot/bios/first_stage.asm` uses the conventional BIOS `0x7C00` first-stage load/handoff address. `boot/uefi/entry.c` describes a PE/COFF EFI application; UEFI does not use BIOS interrupts and has no universal `0x8000` entry address. A custom `0x8000` address is only used when explicitly configured.

## ISO inspection and reproducibility

`python/iso_tool/advanced_inspect.py` performs bounded, read-only ISO 9660/UDF/MBR/GPT/El Torito inspection. `python/iso_tool/image.py` constructs xorriso/xorrisofs/Oscdimg commands and supports reproducible profiles using `SOURCE_DATE_EPOCH` where supported.

## Security and verification boundary

Imported boot sectors are treated as inert data. ISO-Tool does not automatically execute imported code, bypass firmware security, mount untrusted images, or perform destructive physical-disk operations. Static validation and optional isolated VM/emulator testing are kept separate from claims of successful boot.

Repository edits and CI configuration do not by themselves prove that a Windows/MSVC/MinGW executable or ISO was successfully produced; those claims require an actual toolchain/CI run with recorded evidence.
