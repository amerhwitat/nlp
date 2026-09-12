# Cross-Platform Build, Deploy, Dependency, and Database Scripts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide one source-preserving automation layer callable from Windows batch, Microsoft PowerShell, POSIX shell, Python, and Perl for web/native builds, dependency setup, and database lifecycle operations.

**Architecture:** PowerShell is the canonical Windows orchestration layer; POSIX shell is the canonical Unix entrypoint; Python and Perl provide portable orchestration equivalents; `.bat` files are stable Windows launchers. Database scripts prefer SQLite for zero-install development and PostgreSQL when `NLP_DATABASE_URL`/`DATABASE_URL` is supplied.

**Tech Stack:** PowerShell 7+/Windows PowerShell fallback, CMD batch, POSIX sh/bash, Python 3, Perl 5, npm, .NET CLI, SQLite, PostgreSQL client, existing C/C++/WASM toolchains.

**Spec:** `docs/superpowers/specs/2026-09-12-web-first-build-deploy-database-batch-architecture-design.md`

## Global Constraints

- Preserve `python/`, `cpp/`, `vcpp/`, `dotnet/`, existing web code, data, database and application directories.
- Never commit passwords, API keys, database credentials, or cloud tokens.
- Generated output belongs under `artifacts/` and must not overwrite source.
- Scripts resolve the repository root from their own location.
- Windows defaults to x64 and Release unless overridden.
- Database initialization is deterministic and ordered by SQL filename.
- Optional tools produce actionable diagnostics rather than silently claiming success.

---

### Task 1: Cross-platform bootstrap

**Files:**
- Create: `scripts/bootstrap/install-dependencies.ps1`
- Create: `scripts/bootstrap/install-dependencies.bat`
- Create: `scripts/bootstrap/install-dependencies.sh`
- Create: `scripts/bootstrap/install-dependencies.py`
- Create: `scripts/bootstrap/install-dependencies.pl`

- [x] Implement platform-aware prerequisite checks.
- [x] Create `.venv` and install `python/requirements.txt` when present.
- [x] Install web dependencies using `npm ci` when a lockfile exists, otherwise `npm install`.
- [x] Keep database initialization separate from dependency installation while allowing a prerequisite check.

### Task 2: Build orchestration

**Files:**
- Create: `scripts/build/build-all.ps1`
- Create: `scripts/build/build-all.bat`
- Create: `scripts/build/build-all.sh`
- Create: `scripts/build/build-all.py`
- Create: `scripts/build/build-all.pl`

- [x] Resolve the repository root independently of the caller's working directory.
- [x] Build the web application when `web/package.json` exists.
- [x] Validate Python and build .NET when those targets exist.
- [x] Delegate native/WASM/database operations to dedicated scripts when present.
- [x] Propagate failures through non-zero exit status.

### Task 3: Database lifecycle entrypoints

**Files:**
- Create: `scripts/database/init-database.ps1`
- Create: `scripts/database/init-database.bat`
- Create: `scripts/database/init-database.sh`
- Create: `scripts/database/init-database.py`
- Create: `scripts/database/init-database.pl`

- [x] Provide `check`/`Check` and `init`/`Init` modes.
- [x] Detect SQLite and PostgreSQL clients.
- [x] Use `NLP_DATABASE_URL` or `DATABASE_URL` for PostgreSQL without storing credentials.
- [x] Initialize SQLite under `artifacts/database/` for local development.
- [x] Apply SQL files in deterministic filename order.

### Task 4: Documentation and verification

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-09-12-web-first-build-deploy-database-batch-architecture-design.md`

- [x] Document all five script families.
- [x] Document supported invocation patterns and environment variables.
- [ ] Run syntax checks in environments where the interpreters are available.
- [ ] Run CI build/test workflows after workflow execution is available.

### Verification commands

```text
python scripts/bootstrap/install-dependencies.py developer
python scripts/build/build-all.py --configuration Release --skip-database
python scripts/database/init-database.py check
perl -c scripts/build/build-all.pl
perl -c scripts/bootstrap/install-dependencies.pl
perl -c scripts/database/init-database.pl
pwsh -NoProfile -File scripts/database/init-database.ps1 -Mode Check
```

The GitHub connector cannot execute these interpreters inside the repository runtime, so interpreter execution must be verified by the repository's CI runner or a local checkout. The scripts themselves return failures rather than masking them.
