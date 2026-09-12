# Web-First Build, Deployment, Dependency, and Database Automation Architecture

**Date:** 2026-09-12  
**Repository:** `amerhwitat/nlp`  
**Branch:** `plans/ancient-language-coptic-greek-aramaic`

## 1. Purpose

Extend the approved web-first/source-preserving architecture with a complete, repeatable automation layer for development, compilation, WebAssembly, testing, packaging, deployment, dependency installation, and database provisioning. Native source implementations remain in their existing language directories; web implementations are added alongside them rather than replacing them.

The automation layer must support Windows-first development while remaining usable from Linux/macOS where practical. It must expose predictable entry points for developers, CI, release packaging, local installation, and database initialization.

## 2. Source-preservation rule

The existing native implementations are authoritative source assets and must remain in place:

- `python/`
- `cpp/`
- `vcpp/`
- `dotnet/`
- existing `web/javascript/`
- existing `data/`
- existing `db/`
- existing application-specific directories

The web migration adds new web targets without deleting or silently relocating native files. Every conversion target must retain a mapping to its original implementation and document whether the web version is a direct port, WebAssembly adapter, TypeScript implementation, or browser-only integration.

## 3. Target automation layout

The following structure is the target architecture. Existing files with the same responsibility should be enhanced rather than duplicated:

```text
scripts/
├── bootstrap/
│   ├── install-dependencies.bat
│   ├── install-dependencies.ps1
│   ├── install-dependencies.sh
│   ├── check-prerequisites.bat
│   └── check-prerequisites.ps1
├── build/
│   ├── build-all.bat
│   ├── build-all.ps1
│   ├── build-all.sh
│   ├── build-web.bat
│   ├── build-web.ps1
│   ├── build-native.bat
│   ├── build-native.ps1
│   ├── build-wasm.bat
│   ├── build-wasm.ps1
│   ├── build-dotnet.bat
│   ├── build-python.bat
│   └── build-databases.bat
├── test/
│   ├── test-all.bat
│   ├── test-all.ps1
│   └── test-web.bat
├── database/
│   ├── install-postgresql.bat
│   ├── install-sqlite.bat
│   ├── init-database.bat
│   ├── migrate-database.bat
│   ├── seed-database.bat
│   ├── backup-database.bat
│   └── restore-database.bat
├── deploy/
│   ├── deploy-web.bat
│   ├── deploy-web.ps1
│   ├── package-release.bat
│   └── package-release.ps1
└── clean/
    ├── clean-all.bat
    └── clean-all.ps1

web/
├── package.json
├── package-lock.json
├── tsconfig.json
├── vite.config.*
├── app/
├── components/
├── workers/
├── wasm/
├── speech/
├── ocr/
├── ancient-languages/
├── archaeology/
├── research/
├── similarity/
├── neural/
├── api/
└── assets/

config/
├── development.env.example
├── test.env.example
└── production.env.example

artifacts/
├── web/
├── wasm/
├── native/
├── dotnet/
├── python/
├── database/
└── releases/
```

If the repository already has an equivalent directory, the implementation should consolidate into it instead of creating parallel competing layouts.

## 4. Canonical commands

The automation layer must provide these logical operations:

| Operation | Canonical responsibility |
|---|---|
| bootstrap | verify host tools and install declared project dependencies |
| build:web | compile TypeScript/JavaScript and package browser assets |
| build:wasm | compile supported C/C++ modules to WebAssembly and copy runtime artifacts |
| build:native | build C++ and Visual C++ targets without modifying source layout |
| build:dotnet | build supported .NET projects |
| build:python | create/test Python environments and validate Python modules |
| build:db | validate and package database schema/migrations/seeds |
| build:all | execute the supported build graph in deterministic order |
| test:all | run unit, integration, schema, web and native checks that are available |
| deploy:web | package and deploy the static/web application using configured provider adapters |
| install:db | install or verify local database prerequisites and initialize a project database |
| migrate:db | apply schema migrations in order |
| seed:db | load reproducible registry/reference seed data |
| backup:db | create a timestamped backup without exposing credentials |
| restore:db | restore a selected backup after an explicit confirmation gate |
| package:release | create a versioned release bundle containing web, WASM, native, data and database artifacts as configured |
| clean | remove generated artifacts only, never source or authoritative data |

The root `package.json` scripts must mirror the canonical web operations so developers can use `npm run <operation>` in addition to the platform scripts. npm lifecycle hooks should be used only where they add deterministic value; dependency installation should remain explicit and auditable. npm documentation recommends keeping build work in package scripts and avoiding unnecessary install-time scripts. citeturn0search1turn0search8

## 5. Dependency installation

Dependency installation must be explicit, idempotent, and platform-aware.

### Windows

The primary path is:

1. Check for Git, Node.js/npm, Python, CMake, a supported C++ compiler, .NET SDK, and database client/server tools.
2. Prefer already-installed system tools rather than overwriting them.
3. Install only missing project dependencies through declared package managers or documented installers.
4. Use a lockfile for Node dependencies.
5. Create/activate a Python virtual environment for Python tooling.
6. Record tool versions into a generated build manifest.

Visual Studio/MSVC builds must select a matching architecture environment. Microsoft toolchains provide command-line environment setup through the Visual Studio developer environment; the database automation must not mix incompatible 32-bit and 64-bit dependency trees. citeturn0search3

### Cross-platform

PowerShell and POSIX shell entry points should call the same logical operations. Platform scripts must fail with actionable diagnostics when a required tool is absent.

### Security

No script may contain passwords, API keys, cloud credentials, database secrets, or personal access tokens. Secrets are loaded from environment variables, CI secret stores, or user-managed local configuration files excluded from Git.

## 6. Database architecture

The application database automation must support the repository's normalized ER/MADM architecture and preserve the existing SQL targets.

Primary development options:

- SQLite for zero-install local development and browser-adjacent/test workflows.
- PostgreSQL for the full relational research/knowledge-graph application profile.
- Additional SQL targets remain schema-generation/compatibility targets where supported by the existing project.

Database scripts must:

1. verify connectivity;
2. create the configured database/schema when authorized;
3. apply migrations in deterministic order;
4. create required extensions only when explicitly enabled and available;
5. load language, script, Unicode, provenance and other safe reference seeds;
6. run schema validation checks;
7. produce a machine-readable migration/build report;
8. support backup and restore operations.

For Windows, the project should prefer supported PostgreSQL binary distributions for normal developer installation rather than requiring users to build PostgreSQL from source. PostgreSQL documentation explicitly recommends binary distributions for most Windows users and documents Visual C++/Windows SDK source builds separately. citeturn0search0

## 7. Web build pipeline

The web build must execute in this order unless dependency analysis proves a safe parallel alternative:

```text
prerequisite check
      |
      +--> npm dependency restore
      |
      +--> Python environment validation
      |
      +--> native/WASM prerequisite validation
      |
      +--> database schema validation
      |
      v
TypeScript/static web build
      |
      +--> Web Workers
      +--> WASM runtime integration
      +--> speech adapters
      +--> OCR/ancient-language modules
      +--> archaeology/research/similarity modules
      |
      v
unit/integration tests
      |
      v
production bundle
      |
      v
release manifest + checksums
```

Existing `web/javascript/` functionality must be migrated incrementally into the web architecture without breaking its current entry points.

## 8. WebAssembly boundary

C/C++ functionality suitable for browser execution must be exposed through narrow WebAssembly interfaces rather than copying arbitrary native internals into browser code. The boundary must define:

- memory ownership;
- UTF-8 input/output;
- binary/image buffer handling;
- error/status codes;
- cancellation where practical;
- deterministic build flags;
- worker-safe invocation rules.

WebAssembly artifacts must be versioned and checksummed. Native builds remain independently buildable.

## 9. Speech, OCR, neural and research adapters

The build/deploy layer must package optional adapters for:

- browser Web Speech API fallback;
- Whisper.cpp-compatible WASM/native ASR integration;
- sherpa-ONNX local/server speech integration;
- TTS backends;
- VAD/audio processing;
- OCR and glyph classification;
- neural embeddings/similarity;
- archaeological/historical research APIs.

Third-party engines and models are not copied into the repository merely to automate builds. The repository stores adapters, manifests, license/provenance metadata, version constraints and checksums where appropriate.

## 10. Deployment

Deployment scripts must support a provider-neutral static-web deployment interface:

```text
build:web
  -> test:web
  -> package:web
  -> generate manifest/checksums
  -> deploy adapter
  -> verify deployed health/version endpoint
```

Provider-specific credentials are supplied only through environment variables or CI secrets. A local deployment script must support a dry-run/package-only mode so users can validate output without publishing it.

Deployment must preserve source maps only when explicitly configured for a release profile and must never publish `.env` files, database credentials, private model tokens, or local development databases.

## 11. Batch script requirements

Every `.bat` script must:

- use `@echo off` and clear error handling;
- resolve the repository root relative to the script location;
- avoid assuming the current working directory;
- use `setlocal`/`endlocal`;
- return non-zero exit codes on required failures;
- quote paths containing spaces;
- print the selected configuration and tool versions;
- delegate complex logic to PowerShell or project-native package commands rather than duplicating large implementations;
- avoid destructive operations outside generated directories;
- support `Debug` and `Release` where meaningful;
- support `x64` as the default Windows architecture while allowing an explicit override;
- be safe to call from CI.

PowerShell scripts are the canonical implementation for complex Windows orchestration; batch files are stable entry points for Visual Studio, Code::Blocks, CI runners and users who want one-click commands.

## 12. Installation profiles

Provide explicit profiles rather than one opaque installer:

- `minimal`: web runtime + SQLite + declared frontend dependencies;
- `developer`: minimal + Python + C/C++ + .NET + database tooling + tests;
- `research`: developer + optional speech/OCR/neural model adapters and research-data tooling;
- `server`: production web/API/database prerequisites without desktop-only tools;
- `ci`: noninteractive, locked dependencies, deterministic builds, no interactive database/password prompts.

## 13. Reproducibility

Each successful build/package should emit `artifacts/*/build-manifest.json` containing:

- repository commit/ref;
- build timestamp;
- OS/architecture;
- compiler/interpreter/runtime versions;
- dependency lockfile hash;
- enabled feature flags;
- database schema/migration version;
- WASM/native artifact hashes;
- web bundle hash;
- optional model/runtime identifiers;
- deployment target identifier without credentials.

## 14. CI integration

GitHub Actions must invoke the same canonical scripts used locally rather than maintaining a second build implementation. CI should cover, as applicable:

- dependency installation;
- web type/build checks;
- Python tests;
- C++/MSVC builds;
- .NET builds/tests;
- WASM compilation;
- database schema/migration tests;
- packaging;
- deployment dry-run.

Actual deployment remains gated by repository environment/secret configuration.

## 15. Testing requirements

Tests must include:

- missing-prerequisite diagnostics;
- idempotent dependency installation;
- clean/rebuild behavior;
- Debug/Release build paths;
- x64 Windows path;
- web production bundle;
- WASM load and UTF-8 round-trip;
- Python/native regression tests;
- database migration from empty state;
- repeat migration with no unintended changes;
- seed validation;
- backup/restore smoke test;
- package manifest/checksum validation;
- deployment dry-run;
- secret exclusion from release artifacts.

## 16. Non-goals

This automation design does not:

- replace native source implementations;
- force every native algorithm into WebAssembly when browser execution is inappropriate;
- embed third-party proprietary models or credentials;
- automatically publish research claims as authoritative historical facts;
- make destructive database migrations without an explicit migration policy;
- install software outside the user's configured environment without an explicit installer/profile action.

## 17. Acceptance criteria

The implementation is accepted when:

1. the existing source directories remain intact;
2. a clean checkout can run the documented bootstrap path on supported Windows tooling;
3. `build-all` reaches web/native/.NET/Python/WASM/database targets that are enabled and reports unsupported optional targets clearly;
4. database initialization and migration are repeatable;
5. release packaging creates a manifest and checksums;
6. web deployment can be performed through a configured adapter or validated through dry-run;
7. CI invokes the same canonical automation commands;
8. all generated files are kept out of source directories unless explicitly designated as generated assets;
9. no credentials are committed or packaged;
10. documentation explains every script and profile.

## 18. Implementation sequencing

After this specification is reviewed and approved, the implementation plan will be created separately. The plan will cover the automation foundation first, then web/WASM builds, database lifecycle, dependency profiles, testing/CI, deployment packaging, and final documentation. Each task will have an independently testable deliverable and frequent commits.
