# Git Source, Archive, Application, and Output Pipeline Design

## Purpose

Extend ISO-Tool so a user can enter a Git/GitHub URL, direct source archive, or local source path; acquire it; scan all safe-to-read source and documentation; discover build systems and optional applications; compile recognized projects; and write ISO, IMG, boot-image, and binary outputs to explicitly selected destinations.

## Source inputs

1. GitHub repository URL or `owner/repository`.
2. Generic Git URL, including SSH form when Git is installed.
3. Direct ZIP/TAR/TAR.GZ/TGZ source archive.
4. Local source archive.
5. Local source directory.

Acquisition records URL/input, source type, SHA-256 for downloaded archives, extracted root, and Git revision when available.

## Safety

Archive extraction rejects absolute paths and `..` traversal. Source files are scanned before build execution. Downloaded scripts are never executed merely because they exist. Build execution is limited to registered build-system adapters. Package installation requires explicit user authorization and registered package-manager commands.

## Repository intelligence

The scanner reads text source, documentation, manifests, build descriptions, linker scripts, assembly, C/C++, Rust, Go, Java, Python, Node, C#, Fortran, Swift, Make, CMake, Meson, Autotools, Ninja, Cargo, npm, Maven/Gradle, Visual Studio project/solution files, and image metadata. It ignores common VCS/cache/vendor/build trees unless explicitly requested.

## Application discovery

Detect application/package manifests and available package managers. Produce `applications.json` with `required`, `recommended`, `optional`, and `unavailable` entries. Supported package-manager families include APT, DNF, Zypper, pacman, apk, XBPS, Portage, Homebrew, Flatpak, Snap, WinGet, Chocolatey, and Scoop. Discovery does not install anything.

## Build pipeline

`acquire -> verify -> extract -> scan -> application discovery -> dependency graph -> deterministic build plan -> registered build adapters -> artifact staging -> filesystem merge -> ISO/IMG mastering -> verification`.

CMake target/dependency ordering remains authoritative. AI/RNN/LLM components can annotate or propose plan refinements but cannot bypass policy or dependency validation.

## Outputs

The user can independently select directories for:

- ISO files
- IMG files
- boot images
- executables/libraries
- source/download cache

The output manifest records every produced artifact and SHA-256 checksum.
