# Cross-Platform Script Addendum

This addendum extends the approved web-first build/deployment architecture with first-class automation in **POSIX shell, Microsoft PowerShell, Python 3, Perl 5, and Windows CMD batch**.

## Canonical responsibilities

- **PowerShell:** canonical Windows orchestration and database/deployment logic.
- **CMD batch:** compatibility launcher for Windows users, Visual Studio, Code::Blocks and CI.
- **POSIX shell:** Linux/macOS-compatible orchestration entrypoint.
- **Python:** portable orchestration for environments where Python is already the primary runtime.
- **Perl:** portable orchestration for Unix/legacy automation environments.

The implementations expose the same logical operations rather than duplicating business logic.

## Required operations

Each automation family must support, where applicable:

- prerequisite/dependency installation;
- web dependency restore and build;
- Python environment validation;
- native/.NET/WASM delegation;
- database prerequisite check;
- database initialization;
- deterministic SQL ordering;
- failure propagation;
- repository-root resolution independent of current directory.

## Dependency safety

Scripts never embed secrets. PostgreSQL credentials are provided through `NLP_DATABASE_URL` or `DATABASE_URL`; local SQLite uses `artifacts/database/nlp.sqlite`.

PowerShell 7 is cross-platform, while Windows PowerShell 5.1 remains available side-by-side on Windows. Microsoft documents supported PowerShell installation paths for Windows and Linux. citeturn0search0turn0search2 Perl's standard execution model supports direct script execution and portable shebang-based invocation. citeturn0search4turn0search14

## Verification policy

The GitHub connector can author and inspect repository files but does not execute the repository's local interpreters. Runtime validation therefore belongs to CI/local checkout execution. Scripts must fail closed when required tools are absent and print actionable diagnostics.
