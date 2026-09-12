# NLP / Ancient Language Research Toolkit

This repository contains Python, C++, .NET, Visual C++ and desktop/web implementations for Thamudic, Ancient North Arabian and the expanding ArchaeologicalKnowledgeSystem ancient-language research platform.

## Ancient-language intelligence expansion

The registry-driven architecture is being extended across Mesopotamian cuneiform, Egyptian writing, Ancient North/South Arabian, Coptic, Greek, Latin, Ancient/Modern Aramaic, Syriac, Mandaic and Ancient/Modern Hebrew.

### Hebrew

Hebrew historical stages are represented separately:

- Ancient/Biblical Hebrew
- Second Temple/historical Hebrew
- Masoretic Hebrew
- Medieval Hebrew
- Modern Hebrew
- Paleo-Hebrew historical script context
- Samaritan Hebrew related/distinct tradition

Hebrew registry data preserves RTL direction, final forms, niqqud, cantillation and combining behavior.

## Unicode, UTF-8 and alphabet/sign registry

The application ships structured alphabet/sign tables and a versioned offline Unicode snapshot. Each character/sign record can retain:

- language and historical stage
- script/script variant
- Unicode code point
- Unicode name
- exact UTF-8 bytes
- normalization and bidi metadata
- transliteration
- pronunciation profile
- provenance/source/version/checksum

`tools/generate_ancient_unicode_registry.py` rebuilds the Unicode snapshot from a pinned Unicode Character Database. This is intentionally separate from language identity: Unicode Script does not by itself identify a language.

For cuneiform and Egyptian hieroglyphs, the registry models signs and values rather than pretending that every writing system is an alphabet.

## Pronunciation and voice

Transliteration and target-language translation fields now have browser speech controls for:

- Listen
- Pause
- Resume
- Stop
- Replay
- voice selection
- locale
- rate
- pitch
- volume
- pronunciation mode

Modern speech, scholarly pronunciation, reconstructed pronunciation and reference pronunciation are distinct. Ancient reconstructed speech is explicitly labeled and is never represented as an authenticated historical recording without appropriate evidence.

## Cross-platform automation

The repository now provides source-preserving build, dependency, database and release entrypoints under `scripts/`. PowerShell is the canonical Windows orchestration layer, POSIX shell is the Unix entrypoint, and Python/Perl provide portable orchestration equivalents. Windows `.bat` files are stable launchers for users and CI.

### Bootstrap

- `scripts/bootstrap/install-dependencies.bat`
- `scripts/bootstrap/install-dependencies.ps1`
- `scripts/bootstrap/install-dependencies.sh`
- `scripts/bootstrap/install-dependencies.py`
- `scripts/bootstrap/install-dependencies.pl`

Profiles include `minimal`, `developer`, `research`, `server`, and `ci` where supported. Web dependencies use `npm ci` when `package-lock.json` is present. Python dependencies use an isolated `.venv` when a requirements file exists.

### Build

- `scripts/build/build-all.bat`
- `scripts/build/build-all.ps1`
- `scripts/build/build-all.sh`
- `scripts/build/build-all.py`
- `scripts/build/build-all.pl`

The build orchestration preserves native source directories and delegates specialized native/WASM targets to dedicated scripts when they are present. Windows defaults to `Release`; `Debug` can be selected through the PowerShell/Python interfaces.

### Database

- `scripts/database/init-database.bat`
- `scripts/database/init-database.ps1`
- `scripts/database/init-database.sh`
- `scripts/database/init-database.py`
- `scripts/database/init-database.pl`

Use `check`/`Check` to inspect available clients. Use `init`/`Init` to initialize SQL in deterministic filename order. SQLite is the zero-install local path; PostgreSQL is selected through `NLP_DATABASE_URL` or `DATABASE_URL`. Credentials are never stored in scripts.

### Typical commands

```text
Windows CMD:      scripts\\bootstrap\\install-dependencies.bat developer
Windows PowerShell: pwsh scripts\\build\\build-all.ps1 -Configuration Release
Linux/macOS:      bash scripts/build/build-all.sh
Python:           python scripts/build/build-all.py --configuration Release
Perl:             perl scripts/build/build-all.pl
Database check:   python scripts/database/init-database.py check
Database init:    python scripts/database/init-database.py init
```

The same automation contract is intended for local development and CI, reducing platform-specific drift. Generated outputs belong under `artifacts/` and source files remain in their original language directories.

## ThamudicScan web application

Public deployment:

https://thamudicscan-s3wz30.public.builtwithrocket.new/

The hosted application provides a browser-accessible interface for the Thamudic / North Arabian research tooling maintained in this repository.

## Ancient North Arabian Unicode support

The repository includes a canonical registry at `data/ancient_north_arabian/alphabet.json` covering the complete Unicode Old North Arabian block `U+10A80–U+10A9F`:

- 29 letters
- 3 encoded numbers
- Unicode character
- Unicode code point
- scholarly transliteration
- exact UTF-8 byte sequence
- Dadanitic encoding basis
- variant-script metadata for Safaitic, Hismaic, Taymanitic, Minaic and Thamudic B

Unicode encodes Old North Arabian using Dadanitic forms. Variant historical forms are represented as variant/font metadata rather than fabricated Unicode code points.

## Implementations

- `cpp/thamudic/` — C++20 library and Unicode registry.
- `vcpp/` — Visual Studio native Windows desktop scanner.
- `dotnet/` — CLI, WPF desktop and web implementations.
- `python/thamudic/` — Python Unicode/UTF-8/transliteration API.
- `python/ancient_languages/` — shared historical-language, Unicode and pronunciation services.
- `python/tests/` — Unicode, Hebrew and pronunciation regression tests.
- `data/ancient_languages/` — language-stage, alphabet, Unicode and pronunciation registries.
- `db/sql/` — normalized ER and analytical/MADM schemas.
- `web/javascript/` — browser workbench and voice controls.
- `ThamudicScan/` — documentation and link to the public web deployment.

## Documentation

- `docs/superpowers/specs/2026-09-12-ancient-language-intelligence-expansion.md`
- `docs/superpowers/specs/2026-09-12-hebrew-alphabet-utf8-voice-addendum.md`
- `docs/superpowers/specs/2026-09-12-web-first-build-deploy-database-batch-architecture-design.md`
- `docs/superpowers/plans/2026-09-12-cross-platform-build-deploy-scripts-plan.md`
- `docs/superpowers/plans/2026-09-12-hebrew-alphabet-utf8-voice-plan.md`
- `docs/ANCIENT_LANGUAGE_VOICE.md`

## Sources

Unicode's Character Database and code charts are the normative encoding sources. ISO 15924 supplies standardized script codes. Scholarly resources are retained as provenance-qualified sources for historical stages, transliteration and pronunciation.
