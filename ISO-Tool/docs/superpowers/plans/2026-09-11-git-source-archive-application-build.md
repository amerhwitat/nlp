# Git Source, Archive, Application, and Output Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let ISO-Tool acquire Git/GitHub repositories or source archives, scan them, discover optional applications, compile recognized source, and write ISO/IMG artifacts to user-selected locations.

**Architecture:** A source-acquisition layer normalizes Git/GitHub/archive/local inputs and records provenance. A scanner produces repository/application knowledge; the existing deterministic build planner remains authoritative while AI only proposes refinements. The GUI exposes source and separate output-directory fields and runs acquisition, scanning, compilation, staging, and image generation.

**Tech Stack:** Python 3, Tkinter, Git, urllib/zipfile/tarfile, CMake, GNU/MSVC, existing ISO mastering engine, JSON manifests.

**Spec:** `ISO-Tool/docs/superpowers/specs/2026-09-11-git-source-archive-application-build-design.md`

## Global Constraints

- Never execute arbitrary downloaded scripts solely because they exist in a source archive.
- Record source URL/type, archive hash, Git revision when available, detected build systems, applications, and output paths.
- Preserve deterministic dependency/build ordering; AI output is advisory.
- Package-manager installation requires explicit authorization and uses registered package-manager commands only.
- ISO/IMG creation consumes staged artifacts rather than concatenating unrelated image bytes.

---

### Task 1: Source acquisition

**Files:** `ISO-Tool/python/iso_tool/github_source.py`, `ISO-Tool/python/iso_tool/source_archive.py`, tests.

- [ ] Add tests for Git URLs, direct archives, ZIP extraction, path traversal rejection, and provenance manifest generation.
- [ ] Implement archive download/extraction with SHA-256 recording and safe member paths.
- [ ] Extend repository input detection beyond GitHub to generic Git URLs and local archives.
- [ ] Run focused tests, then commit.

### Task 2: Application discovery

**Files:** `ISO-Tool/python/iso_tool/application_discovery.py`, tests, package documentation.

- [ ] Test detection of package-manager manifests and target package-manager availability.
- [ ] Implement required/recommended/optional application classification from repository metadata and system package sources.
- [ ] Generate `applications.json` without executing package installation.
- [ ] Add explicit `--yes`/authorization gate for registered package-manager operations.
- [ ] Run focused tests, then commit.

### Task 3: Build orchestration

**Files:** `ISO-Tool/python/iso_tool/build_entrypoint.py`, `ISO-Tool/python/iso_tool/source_archive.py`, tests.

- [ ] Test source-type dispatch and output manifest generation.
- [ ] Compile recognized CMake sources with GNU/MSVC as currently supported and add recognized Make/Meson/Autotools/Cargo/npm/Gradle/Visual Studio command plans where toolchains exist.
- [ ] Keep unsupported systems in a report rather than failing the whole acquisition scan.
- [ ] Stage executables, libraries, EFI, BIN, and IMG outputs.
- [ ] Run focused tests, then commit.

### Task 4: GUI

**Files:** `ISO-Tool/python/main.py`, tests/docs.

- [ ] Add source URL/archive textbox and Browse Source control.
- [ ] Add separate ISO output, IMG output, boot-image, and artifact destination controls.
- [ ] Add Acquire/Scan, Discover Applications, Compile All Recognized, and Build ISO+IMG actions.
- [ ] Display provenance, discovered build systems, applications, and final output paths.
- [ ] Run syntax/import checks and commit.

### Task 5: Mirror and documentation

**Files:** corresponding `ISO-Tool` files in `amerhwitat/ChimeraIIOS` plus README/docs/tests.

- [ ] Mirror the implementation and documentation while retaining Chimera II OS defaults.
- [ ] Document supported inputs, optional application policy, output layout, reproducibility, and safety rules.
- [ ] Run repository-level syntax/tests available in the environment and report any unavailable native toolchains honestly.
