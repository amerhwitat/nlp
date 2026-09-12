# Multi-Repository Build Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible, cross-platform build/release orchestration layer across the user's GitHub repositories, with visible dependency/build/link progress, Python PyInstaller packaging, Java launch/build scripts, web/mobile application targets, database automation, and repository-specific documentation.

**Architecture:** Each repository receives only the build targets appropriate to its detected source tree, while a shared `build-tools/` convention provides common logging, dependency detection, failure handling, artifact manifests, and database runners. GitHub Actions provides authoritative CI across Windows/Linux/macOS because native toolchains must run on their target platforms; local `.bat`, PowerShell, POSIX shell, and macOS-oriented scripts call the same repository build entry points. Python packaging uses PyInstaller on each target OS rather than cross-compilation. Web targets use Next.js/React/Vue/Angular where the repository has a web application, and React Native only where mobile UI is appropriate.

**Tech Stack:** Bash/POSIX sh, PowerShell 7+, Windows CMD `.bat`, Python 3.8+, PyInstaller, Java/JDK, Maven/Gradle where detected, CMake/Ninja/MSBuild/Visual Studio where detected, Node.js/npm/pnpm, Next.js, React, Vue, Angular, React Native, GitHub Actions, SQL runners, Docker Compose where useful, repository-specific existing toolchains.

**Spec:** This plan implements the user's approved cross-repository build, packaging, application-target, SQL automation, and documentation expansion request.

## Global Constraints

- Do not claim a repository builds successfully until a target-native CI/local verification provides evidence.
- Do not cross-compile PyInstaller artifacts; build each OS artifact on that OS.
- Preserve existing source structure and behavior unless a build integration requires a focused change.
- Detect dependencies from existing manifests before adding new dependencies.
- Every generated runner must show repository, stage, command, dependency installation, compiler/linker stage, artifact location, elapsed time, and exit status.
- Never hard-code secrets, credentials, personal tokens, database passwords, or signing keys.
- Database scripts remain parameterized and support dry-run/validation where the database engine permits it.
- Web/mobile frameworks are added only to repositories with an appropriate application surface; shared build infrastructure must not force a framework into unrelated repositories.
- Generated artifacts belong in ignored build/dist directories or CI artifacts, not source control unless explicitly required.
- Existing Ancient Scripts/RNN-LLM database work remains compatible with the new orchestration layer.

---

## Repository Scope

Primary repositories identified from the user's GitHub account:

- `amerhwitat/nlp`
- `amerhwitat/ChimeraIIOS`
- `amerhwitat/BizX`
- `amerhwitat/BizXtreme`
- `amerhwitat/PDFreaderPY`
- `amerhwitat/bruteforce`
- `amerhwitat/keygen`
- `amerhwitat/CPU4096`
- `amerhwitat/CPU4096Simulator`
- `amerhwitat/general`
- `amerhwitat/test`
- `amerhwitat/amerhwitat.github.io`

Repository capabilities must be detected rather than assumed. For example, Java scripts are generated when Java sources/manifests exist; React Native is generated only for mobile projects; database runners are generated when SQL/database configuration exists.

---

## Task 1: Shared Build Contract and Repository Manifest

**Files:**
- Create: `build-tools/repository-manifest.schema.json`
- Create: `build-tools/repositories.json`
- Create: `build-tools/README.md`
- Create: `docs/BUILD_ORCHESTRATION.md`

**Interfaces:**
- Manifest fields: `repository`, `language`, `build_system`, `entrypoints`, `dependencies`, `databases`, `web_targets`, `mobile_targets`, `artifacts`, `supported_os`.
- Orchestrator consumes this manifest and emits structured progress events.

- [ ] **Step 1: Define the JSON schema** for repository capabilities, commands, dependency managers, artifacts, and target operating systems.
- [ ] **Step 2: Populate repository manifests** from actual repository trees and existing build files; never invent a language target solely because it was requested.
- [ ] **Step 3: Document capability detection and override rules** so maintainers can add a target without changing the global runner.
- [ ] **Step 4: Validate representative manifests** with JSON parsing and schema checks.
- [ ] **Step 5: Commit** with message `build: add cross-repository build manifest contract`.

---

## Task 2: Cross-Platform Local Runner

**Files:**
- Create: `build-tools/build_all.sh`
- Create: `build-tools/build_all.ps1`
- Create: `build-tools/build_all.bat`
- Create: `build-tools/build_all.zsh`
- Create: `build-tools/build_all.cmd`
- Create: `build-tools/build_common.py`
- Create: `build-tools/logging.py`
- Create: `build-tools/dependency_probe.py`

**Interfaces:**
- Shell/PowerShell/CMD entry points invoke `python build-tools/build_common.py` where Python is available, or fall back to native commands for bootstrap stages.
- Common runner emits `BUILD START`, `DEPENDENCY`, `COMPILE`, `LINK`, `TEST`, `PACKAGE`, `DATABASE`, and `BUILD END` events.

- [ ] **Step 1: Write failing tests** for command classification, dependency reporting, exit-code propagation, and elapsed-time formatting.
- [ ] **Step 2: Run the tests** and verify failure before implementation.
- [ ] **Step 3: Implement the Python orchestration core** with subprocess streaming so compiler output remains visible in real time.
- [ ] **Step 4: Implement POSIX shell, zsh, PowerShell, and CMD wrappers** that forward arguments and preserve exit codes.
- [ ] **Step 5: Add dry-run mode** that prints the exact commands without executing them.
- [ ] **Step 6: Run unit tests and shell syntax checks** on available tooling.
- [ ] **Step 7: Commit** with message `build: add cross-platform repository orchestrator`.

---

## Task 3: Dependency Installation and Toolchain Probes

**Files:**
- Create: `build-tools/install_dependencies.ps1`
- Create: `build-tools/install_dependencies.bat`
- Create: `build-tools/install_dependencies.sh`
- Create: `build-tools/install_dependencies.zsh`
- Create: `build-tools/toolchain_matrix.json`

**Interfaces:**
- Probe Git, Python, Java/JDK, Node/npm/pnpm, Rust/Cargo, Go, CMake/Ninja, GCC/Clang, MSVC/MSBuild, .NET, Android SDK, Xcode, and database clients when applicable.
- Never silently install a missing system compiler; report the exact package/tool required and continue only when the repository can be built without it.

- [ ] **Step 1: Implement read-only dependency probing.**
- [ ] **Step 2: Add package-manager installation hooks** for Chocolatey/winget on Windows and apt/dnf/pacman/brew where appropriate, gated behind explicit `--install`.
- [ ] **Step 3: Add version reporting and compatibility checks.**
- [ ] **Step 4: Add CI dependency setup matching the local probes.**
- [ ] **Step 5: Test `--check-only` and `--install --dry-run`.**
- [ ] **Step 6: Commit** with message `build: add dependency and toolchain detection`.

---

## Task 4: Python Build and PyInstaller Packaging

**Files:**
- Create: `build-tools/python/build_all_python.py`
- Create: `build-tools/python/build_pyinstaller.ps1`
- Create: `build-tools/python/build_pyinstaller.bat`
- Create: `build-tools/python/build_pyinstaller.sh`
- Create: `build-tools/python/build_pyinstaller.zsh`
- Create: `build-tools/python/pyinstaller_common.py`

**Interfaces:**
- Discover `pyproject.toml`, `requirements*.txt`, `setup.py`, and Python entry points.
- Prefer `python -m PyInstaller` for environment correctness.
- Generate both `onedir` and optional `onefile` artifacts when configured.

- [ ] **Step 1: Discover Python entry points** and write a deterministic build manifest.
- [ ] **Step 2: Create failing tests** for script discovery and safe executable naming.
- [ ] **Step 3: Implement virtual-environment setup and dependency installation.**
- [ ] **Step 4: Implement PyInstaller spec generation/building.**
- [ ] **Step 5: Add Windows `.exe`, Linux executable, and macOS executable/app packaging jobs.**
- [ ] **Step 6: Add artifact checksums and manifest generation.**
- [ ] **Step 7: Verify with target-native CI; do not cross-build.**
- [ ] **Step 8: Commit** with message `build: add Python PyInstaller packaging matrix`.

---

## Task 5: Java Build and Run Automation

**Files:**
- Create: `build-tools/java/build_all_java.ps1`
- Create: `build-tools/java/build_all_java.bat`
- Create: `build-tools/java/build_all_java.sh`
- Create: `build-tools/java/build_all_java.zsh`
- Create: `build-tools/java/run_java_apps.ps1`
- Create: `build-tools/java/run_java_apps.bat`
- Create: `build-tools/java/run_java_apps.sh`

**Interfaces:**
- Detect Maven `pom.xml`, Gradle `build.gradle*`, or plain Java source trees.
- Build each repository independently and label output by repository.
- Run configured main classes/JARs only when a runnable application is detected.

- [ ] **Step 1: Discover Java projects and entry points.**
- [ ] **Step 2: Add Maven/Gradle/plain-JDK command selection.**
- [ ] **Step 3: Stream compilation and linking/packaging output.**
- [ ] **Step 4: Add test and JAR execution commands.**
- [ ] **Step 5: Verify each detected Java repository in CI.**
- [ ] **Step 6: Commit** with message `build: automate Java repository builds and runs`.

---

## Task 6: Native C/C++/.NET Build Automation

**Files:**
- Create: `build-tools/native/build_native.ps1`
- Create: `build-tools/native/build_native.bat`
- Create: `build-tools/native/build_native.sh`
- Create: `build-tools/native/build_native.zsh`
- Create: `build-tools/dotnet/build_all_dotnet.ps1`
- Create: `build-tools/dotnet/build_all_dotnet.bat`
- Create: `build-tools/dotnet/build_all_dotnet.sh`

**Interfaces:**
- CMake projects use configure/build/test/install.
- Visual Studio projects use MSBuild/`dotnet` where applicable.
- Make/autotools projects use their native build systems.

- [ ] **Step 1: Detect CMake, solution/project, Make, and autotools roots.**
- [ ] **Step 2: Add architecture/configuration selection such as Debug/Release and x64/ARM64.**
- [ ] **Step 3: Stream compiler/linker output and preserve logs.**
- [ ] **Step 4: Add test execution and artifact indexing.**
- [ ] **Step 5: Verify ChimeraIIOS and native CPU repositories using their actual manifests.**
- [ ] **Step 6: Commit** with message `build: automate native and dotnet repositories`.

---

## Task 7: Web Application Targets

**Files:**
- Create: `build-tools/web/build_nextjs.ps1`
- Create: `build-tools/web/build_nextjs.bat`
- Create: `build-tools/web/build_nextjs.sh`
- Create: `build-tools/web/build_react.ps1`
- Create: `build-tools/web/build_vue.ps1`
- Create: `build-tools/web/build_angular.ps1`
- Create: `build-tools/web/build_web.sh`
- Create: `build-tools/web/README.md`

**Interfaces:**
- Detect framework from package manifests and existing source directories.
- Next.js uses the repository's package manager and its existing `build` script.
- React/Vue/Angular use existing project scripts rather than replacing application architecture.
- Backend uses the repository's strongest existing backend language; only add a backend scaffold when the repository explicitly has an application surface requiring one.

- [ ] **Step 1: Detect Next.js, React, Vue, and Angular projects.**
- [ ] **Step 2: Add dependency installation and production builds.**
- [ ] **Step 3: Add backend build/run hooks for detected Node, Python, Java, .NET, Go, or Rust services.**
- [ ] **Step 4: Add static artifact manifests and local preview commands.**
- [ ] **Step 5: Verify web builds in GitHub Actions.**
- [ ] **Step 6: Commit** with message `build: add web application target automation`.

---

## Task 8: React Native and Mobile Build Automation

**Files:**
- Create: `build-tools/mobile/build_react_native.ps1`
- Create: `build-tools/mobile/build_react_native.bat`
- Create: `build-tools/mobile/build_react_native.sh`
- Create: `build-tools/mobile/README.md`

**Interfaces:**
- Detect React Native/Expo projects.
- Android builds run on Windows/Linux/macOS where Android tooling is available.
- iOS builds run on macOS with Xcode/CocoaPods; no script should falsely promise iOS builds on Windows/Linux.

- [ ] **Step 1: Detect mobile project manifests.**
- [ ] **Step 2: Add Android dependency and Gradle build commands.**
- [ ] **Step 3: Add macOS iOS/Xcode/CocoaPods commands.**
- [ ] **Step 4: Add Metro/dev-server run commands with visible logs.**
- [ ] **Step 5: Verify platform-specific CI jobs.**
- [ ] **Step 6: Commit** with message `build: add React Native mobile automation`.

---

## Task 9: SQL Database Automation

**Files:**
- Create: `build-tools/database/run_all_sql.ps1`
- Create: `build-tools/database/run_all_sql.bat`
- Create: `build-tools/database/run_all_sql.sh`
- Create: `build-tools/database/README.md`
- Modify: `sql/**` as required for idempotency/engine-specific compatibility.

**Interfaces:**
- Run PostgreSQL, MySQL/MariaDB, SQLite, SQL Server, Oracle, Db2, SAP HANA, BigQuery, Snowflake, DuckDB and Access scripts through native clients where configured.
- Respect `NLP_DB_*` environment variables and never commit credentials.
- Produce a database migration/build report.

- [ ] **Step 1: Inventory every existing SQL directory and classify OLTP/OLAP/MDM/Chimera integration scripts.**
- [ ] **Step 2: Add engine-specific command runners.**
- [ ] **Step 3: Add validation-only mode.**
- [ ] **Step 4: Add ordered migration execution and failure-stop behavior.**
- [ ] **Step 5: Add database health/status output.**
- [ ] **Step 6: Verify syntax with available clients and CI service containers where licensing permits.**
- [ ] **Step 7: Commit** with message `build: automate multi-database initialization`.

---

## Task 10: GitHub Actions Multi-OS CI/CD

**Files:**
- Create: `.github/workflows/build-matrix.yml`
- Create: `.github/workflows/python-packaging.yml`
- Create: `.github/workflows/java-matrix.yml`
- Create: `.github/workflows/web-matrix.yml`
- Create: `.github/workflows/database-validation.yml`
- Create: `.github/workflows/native-matrix.yml`

**Interfaces:**
- Use OS/language matrices where the repository supports them.
- Jobs publish build logs, manifests, checksums and artifacts.
- Packaging jobs depend on successful tests/builds.

- [ ] **Step 1: Create the generic matrix workflow.**
- [ ] **Step 2: Add Python packaging matrix.**
- [ ] **Step 3: Add Java matrix.**
- [ ] **Step 4: Add native/.NET matrix.**
- [ ] **Step 5: Add web/mobile matrix.**
- [ ] **Step 6: Add database validation.**
- [ ] **Step 7: Verify workflow YAML structure and inspect resulting workflow runs.**
- [ ] **Step 8: Commit** with message `ci: add cross-platform build matrix`.

---

## Task 11: Repository-Specific Integration

**Files:**
- Modify each scoped repository's README/build documentation.
- Create repository-local `build/`, `scripts/`, or `tools/` entry points only where that repository needs them.

- [ ] **Step 1: Update `nlp` with ancient-script, RNN/LLM, SQL, Python, Java, native, web and database build instructions.**
- [ ] **Step 2: Update `ChimeraIIOS` with desktop/mobile microkernel, RNN/LLM database, native, ISO and emulator build instructions.**
- [ ] **Step 3: Update `BizX` and `BizXtreme` with Node.js, Java, Python and application build/run entry points.**
- [ ] **Step 4: Update `PDFreaderPY` and other Python repositories with PyInstaller packaging.**
- [ ] **Step 5: Update native CPU repositories with C/C++ build automation.**
- [ ] **Step 6: Update `amerhwitat.github.io` with web build/deploy instructions if its tree contains an applicable frontend.**
- [ ] **Step 7: Update remaining repositories only for detected applicable languages/build systems.**
- [ ] **Step 8: Commit each repository's coherent change set separately.**

---

## Task 12: End-to-End Verification and Release Reports

**Files:**
- Create: `build-tools/verify_all.py`
- Create: `build-tools/generate_build_report.py`
- Create: `docs/BUILD_MATRIX.md`
- Create: `docs/ARTIFACTS.md`

- [ ] **Step 1: Run dependency probes.**
- [ ] **Step 2: Run repository-specific tests.**
- [ ] **Step 3: Run native compilation/linking checks.**
- [ ] **Step 4: Run Python packaging checks on Windows/Linux/macOS runners.**
- [ ] **Step 5: Run Java builds/runs.**
- [ ] **Step 6: Run applicable web/mobile builds.**
- [ ] **Step 7: Run SQL validation.**
- [ ] **Step 8: Generate a machine-readable and human-readable build report with pass/fail/skip and exact reasons.**
- [ ] **Step 9: Perform code review and fix review findings.**
- [ ] **Step 10: Verify again before claiming completion.**

---

## Verification Standards

A repository is marked **PASS** only when its applicable tests/build/package steps complete successfully on a compatible native runner. A target is **SKIPPED** when required proprietary or platform-specific tooling is unavailable, with the exact dependency reported. A target is **FAIL** when its commands execute and return a non-zero status. The global report must distinguish all three states.

PyInstaller packaging follows the documented model of building on the target operating system; it is not treated as a cross-compiler. GitHub Actions matrices are used for OS/language variations, and jobs can run in parallel when independent. Next.js is treated as a full-stack React framework rather than as a separate backend technology, while React Native mobile builds remain platform-dependent.
