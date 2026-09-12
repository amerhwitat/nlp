# Cross-platform build automation

The repository now exposes one deterministic entry point for Windows CMD, PowerShell, Linux and macOS:

- `build-tools\build.bat`
- `powershell -ExecutionPolicy Bypass -File build-tools\build.ps1`
- `./build-tools/build.sh`

Examples:

- Everything detected: `python build-tools/build.py`
- Python + PyInstaller: `python build-tools/build.py --only python --onefile`
- Java: `python build-tools/build.py --only java`
- Node/Next.js/React/Vue/Angular targets: `python build-tools/build.py --only node`
- C/C++/CMake/Make/.NET: `python build-tools/build.py --only native`
- SQL inventory/validation: `python build-tools/build.py --only sql`
- Preview commands without execution: add `--dry-run`.

The runner detects existing project manifests and does not manufacture a framework into an unrelated repository. PyInstaller artifacts are built on the native target OS because PyInstaller output is OS/Python-version specific. Java uses Maven, Gradle, or `javac` according to the detected project. Web projects use their existing package manager and `build` script. Database execution remains credential-free in source control and uses `NLP_DB_*` environment variables with native clients.

Artifacts belong under `build/artifacts/` and `build/` and should remain ignored.
