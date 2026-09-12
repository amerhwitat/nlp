# Historical Objects PDF + KPI + Multi-Repository Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the Ancient Languages / ThamudicEpiPlatform stack with provenance-aware PDF import/export for historical objects and script translations, application KPI dashboards across all maintained language implementations, synchronized automation, and documented integration into both `amerhwitat/nlp` and `amerhwitat/general`.

**Architecture:** Keep `ThamudicEpiPlatform` as the research/data source of truth in `nlp`, while exposing a reusable integration mirror under `general/applications/AncientLanguages/ThamudicEpiPlatform`. PDF processing is a separate service layer: import extracts text/metadata and records page/source provenance; export produces deterministic research reports with object, script, transliteration, translation, confidence, citations, and KPI sections. Dashboards consume a common metrics API/schema so Python, C++, C#, Java, Go, Rust, JavaScript, and TypeScript clients report compatible KPIs without duplicating business logic.

**Tech Stack:** FastAPI, SQLite/PostgreSQL-compatible SQL, React + TypeScript + Vite, Python PDF tooling (pypdf/reportlab with optional image/OCR adapters), language-neutral JSON/CSV registries, C++20, C#/.NET, Java, Go, Rust, JavaScript/TypeScript, Bash, PowerShell, Windows CMD, Docker, GitHub Actions.

**Spec:** `ThamudicEpiPlatform/docs/superpowers/specs/2026-09-13-ancient-languages-platform-design.md`

## Global Constraints

- Preserve Unicode code points as the canonical character identity; UTF-8 is the interchange encoding.
- Preserve scholarly transliteration separately from translation; transliteration is not translation. citeturn0search2
- Use BCP 47/CLDR identifiers for modern target languages and locale/script metadata. citeturn0search0turn0search4
- Preserve source, rights, provenance, model, reviewer, confidence, and timestamp metadata for imported/exported research information.
- Never fabricate an ancient-language translation when no supported model/rule set exists; return an explicit unsupported/uncertain result.
- PDF imports must be bounded, validated, and safe against decompression/path traversal/resource exhaustion risks.
- PDF exports must identify generated content, source records, software/version, data/model versions, and citations.
- Third-party source code is not copied wholesale; only compatible ideas, documented APIs, or permissively licensed dependencies are incorporated.
- Existing Thamudic/Ancient North Arabian registries remain canonical and are reused by all language implementations.
- Historical pronunciation is labeled reconstructed/scholarly where no native TTS model exists.
- Automation must work from clean checkout and fail with actionable diagnostics.

---

### Task 1: Define the PDF and KPI contracts

**Files:**
- Create: `ThamudicEpiPlatform/docs/PDF_IMPORT_EXPORT.md`
- Create: `ThamudicEpiPlatform/docs/KPI_SCHEMA.md`
- Create: `ThamudicEpiPlatform/data/schemas/pdf_export.schema.json`
- Create: `ThamudicEpiPlatform/data/schemas/kpi.schema.json`
- Test: `ThamudicEpiPlatform/server/tests/test_contracts.py`

**Interfaces:**
- Produces versioned JSON contracts consumed by API, UI, and cross-language clients.
- KPI contract includes ingestion counts, objects, scripts, languages, readings, translations, confidence, review status, PDF imports/exports, processing latency, errors, and speech jobs.

- [ ] Step 1: Write failing schema validation tests for required fields and version compatibility.
- [ ] Step 2: Add JSON Schemas with stable identifiers and examples.
- [ ] Step 3: Implement contract validation helpers.
- [ ] Step 4: Run contract tests.
- [ ] Step 5: Document backward-compatibility policy.

### Task 2: Add historical-object PDF import

**Files:**
- Create: `ThamudicEpiPlatform/server/pdf_import.py`
- Create: `ThamudicEpiPlatform/server/pdf_models.py`
- Modify: `ThamudicEpiPlatform/server/app.py`
- Create: `ThamudicEpiPlatform/server/tests/test_pdf_import.py`
- Modify: `ThamudicEpiPlatform/server/requirements.txt`

**Interfaces:**
- `import_pdf(path_or_stream, *, source_id, options) -> PdfImportResult`
- API: `POST /api/pdf/import`
- Result records document metadata, page count, extracted text, detected Unicode scripts, candidate objects, citations, hashes, and warnings.

- [ ] Step 1: Add tests for valid PDF, malformed PDF, empty PDF, Unicode text, metadata, and bounded page limits.
- [ ] Step 2: Implement safe PDF validation and text extraction.
- [ ] Step 3: Add page-level provenance and SHA-256 content identity.
- [ ] Step 4: Detect script/language candidates without inventing translations.
- [ ] Step 5: Persist import/session records.
- [ ] Step 6: Run the PDF test suite.

### Task 3: Add historical-object and translation PDF export

**Files:**
- Create: `ThamudicEpiPlatform/server/pdf_export.py`
- Modify: `ThamudicEpiPlatform/server/app.py`
- Create: `ThamudicEpiPlatform/server/tests/test_pdf_export.py`
- Modify: `ThamudicEpiPlatform/server/requirements.txt`

**Interfaces:**
- `export_research_pdf(report, output) -> PdfExportResult`
- API: `POST /api/pdf/export`
- Supports object dossier, script/transliteration report, translation comparison, interlinear reading, provenance appendix, and KPI appendix.

- [ ] Step 1: Write tests for deterministic metadata, Unicode text, page numbering, citations, and confidence display.
- [ ] Step 2: Implement report models and styles.
- [ ] Step 3: Implement reportlab-based PDF generation.
- [ ] Step 4: Add provenance/citation appendix and machine-readable sidecar manifest.
- [ ] Step 5: Add export endpoint and download headers.
- [ ] Step 6: Run export tests and inspect generated PDFs.

### Task 4: Add database migrations for PDF jobs, reports, and KPIs

**Files:**
- Create: `ThamudicEpiPlatform/database/migrations/002_pdf_translation_kpi.sql`
- Modify: `ThamudicEpiPlatform/database/views.sql`
- Modify: `ThamudicEpiPlatform/database/queries.sql`
- Create: `ThamudicEpiPlatform/database/kpi_views.sql`
- Create: `ThamudicEpiPlatform/server/tests/test_kpi_queries.py`

**Interfaces:**
- Tables for PDF imports/exports, report sections, translation results, model runs, and metric snapshots.
- KPI views provide current totals and time-window aggregates.

- [ ] Step 1: Add migration tests.
- [ ] Step 2: Implement normalized tables with foreign keys and indexes.
- [ ] Step 3: Add KPI views and parameterized queries.
- [ ] Step 4: Verify SQLite compatibility and document PostgreSQL compatibility.
- [ ] Step 5: Run migration/query tests.

### Task 5: Add KPI API and dashboard aggregation

**Files:**
- Create: `ThamudicEpiPlatform/server/kpi.py`
- Modify: `ThamudicEpiPlatform/server/app.py`
- Create: `ThamudicEpiPlatform/server/tests/test_kpi_api.py`

**Interfaces:**
- `GET /api/kpis/summary`
- `GET /api/kpis/timeseries`
- `GET /api/kpis/languages`
- `GET /api/kpis/scripts`
- `GET /api/kpis/applications`
- `GET /api/kpis/health`

- [ ] Step 1: Write API tests and fixture datasets.
- [ ] Step 2: Implement aggregation service.
- [ ] Step 3: Add filtering by application, language, script, date range, and status.
- [ ] Step 4: Add cache-safe bounded queries.
- [ ] Step 5: Run API tests.

### Task 6: Build the web KPI dashboard and PDF workflow UI

**Files:**
- Modify: `ThamudicEpiPlatform/web/src/main.tsx`
- Modify: `ThamudicEpiPlatform/web/src/styles.css`
- Create: `ThamudicEpiPlatform/web/src/components/KpiDashboard.tsx`
- Create: `ThamudicEpiPlatform/web/src/components/PdfWorkspace.tsx`
- Create: `ThamudicEpiPlatform/web/src/components/TranslationReport.tsx`
- Modify: `ThamudicEpiPlatform/web/package.json`
- Create: `ThamudicEpiPlatform/web/vite.config.ts`

**Interfaces:**
- Dashboard cards: objects, inscriptions, scripts, languages, reviewed readings, translation confidence, PDF jobs, errors, latency, speech jobs.
- PDF workspace: upload/import, page/object selection, transliteration/translation preview, provenance, export profile.

- [ ] Step 1: Add component tests for empty/loading/error/success states.
- [ ] Step 2: Implement KPI cards and charts without hardcoded metrics.
- [ ] Step 3: Implement PDF import/export workflow.
- [ ] Step 4: Add accessibility, RTL Arabic, Unicode-safe rendering, and responsive layouts.
- [ ] Step 5: Pin compatible dependency versions and add lockfile/configuration.
- [ ] Step 6: Run TypeScript build/tests.

### Task 7: Integrate existing `nlp` language registries and implementations

**Files:**
- Create: `ThamudicEpiPlatform/languages/registry/index.json`
- Create: `ThamudicEpiPlatform/languages/mesopotamia/registry.json`
- Create: `ThamudicEpiPlatform/languages/egypt/registry.json`
- Create: `ThamudicEpiPlatform/languages/arabia/registry.json`
- Create: `ThamudicEpiPlatform/languages/greek/registry.json`
- Create: `ThamudicEpiPlatform/languages/latin/registry.json`
- Create: `ThamudicEpiPlatform/unicode/scripts.json`
- Create: `ThamudicEpiPlatform/transliteration/systems.json`
- Modify: `ThamudicEpiPlatform/docs/SOURCES.md`
- Modify: root `README.md`

**Interfaces:**
- One language-neutral registry consumed by every implementation.
- Existing Ancient North Arabian/Thamudic registry remains referenced rather than duplicated.

- [ ] Step 1: Add registry validation tests.
- [ ] Step 2: Populate Mesopotamian/Egyptian/Arabian/Greek/Latin metadata with Unicode/script identifiers and provenance.
- [ ] Step 3: Add transliteration-system metadata and explicit distinction from translation.
- [ ] Step 4: Link modern target-language registry to CLDR/BCP 47.
- [ ] Step 5: Run registry tests.

### Task 8: Add common KPI/PDF client interfaces to all maintained languages

**Files:**
- Modify: `ThamudicEpiPlatform/python/client.py`
- Modify: `ThamudicEpiPlatform/cpp/client.cpp`
- Modify: `ThamudicEpiPlatform/csharp/Program.cs`
- Modify: `ThamudicEpiPlatform/java/Main.java`
- Modify: `ThamudicEpiPlatform/go/main.go`
- Modify: `ThamudicEpiPlatform/rust/main.rs`
- Create: `ThamudicEpiPlatform/javascript/client.mjs`
- Create: `ThamudicEpiPlatform/typescript/client.ts`
- Create: language-specific build metadata where absent (Cargo.toml, CMakeLists.txt, .csproj, Maven/Gradle metadata, go.mod).

- [ ] Step 1: Add typed KPI summary retrieval.
- [ ] Step 2: Add PDF import/export endpoint wrappers.
- [ ] Step 3: Add language/script/target-language selection.
- [ ] Step 4: Add error handling and provenance fields.
- [ ] Step 5: Build each client independently.

### Task 9: Create synchronized `general` application integration

**Files:**
- Create: `general/applications/AncientLanguages/ThamudicEpiPlatform/README.md`
- Create: `general/applications/AncientLanguages/ThamudicEpiPlatform/INTEGRATION.md`
- Create: `general/applications/AncientLanguages/ThamudicEpiPlatform/docs/KPI_DASHBOARD.md`
- Create: `general/applications/AncientLanguages/ThamudicEpiPlatform/docs/PDF_WORKFLOW.md`
- Create: `general/applications/AncientLanguages/ThamudicEpiPlatform/data/registry-manifest.json`
- Create: `general/applications/AncientLanguages/ThamudicEpiPlatform/scripts/sync-nlp-assets.sh`
- Create: `general/applications/AncientLanguages/ThamudicEpiPlatform/scripts/sync-nlp-assets.ps1`
- Create: `general/applications/AncientLanguages/ThamudicEpiPlatform/scripts/sync-nlp-assets.bat`

- [ ] Step 1: Define the integration boundary and source-of-truth rules.
- [ ] Step 2: Add registry/build manifests that reference the `nlp` implementation.
- [ ] Step 3: Add reproducible synchronization scripts with SHA-256 verification.
- [ ] Step 4: Add general-repo KPI aggregation documentation.
- [ ] Step 5: Validate clean checkout behavior.

### Task 10: Update automation across both repositories

**Files:**
- Modify/create: `ThamudicEpiPlatform/scripts/deploy.sh`
- Modify/create: `ThamudicEpiPlatform/scripts/deploy.ps1`
- Modify/create: `ThamudicEpiPlatform/scripts/deploy.bat`
- Create: `ThamudicEpiPlatform/scripts/build-all.sh`
- Create: `ThamudicEpiPlatform/scripts/build-all.ps1`
- Create: `ThamudicEpiPlatform/scripts/build-all.bat`
- Create: corresponding `general/applications/AncientLanguages/ThamudicEpiPlatform/scripts/build-all.*`
- Modify: `.github/workflows/thamudic-epi-platform.yml`

- [ ] Step 1: Add dependency/bootstrap checks.
- [ ] Step 2: Make PowerShell scripts compatible with Windows PowerShell 5.1 and PowerShell 7 where practical.
- [ ] Step 3: Add Bash/CMD equivalents for build, test, PDF fixtures, web build, client builds, and packaging.
- [ ] Step 4: Add optional Docker build/run commands.
- [ ] Step 5: Add CI matrix for Python/web and language-client compile checks.
- [ ] Step 6: Run automation from clean environments.

### Task 11: Documentation and citations

**Files:**
- Modify: `ThamudicEpiPlatform/README.md`
- Modify: `ThamudicEpiPlatform/docs/SOURCES.md`
- Create: `ThamudicEpiPlatform/docs/PDF_CITATIONS.md`
- Create: `ThamudicEpiPlatform/docs/KPI_CITATIONS.md`
- Modify: `general/applications/AncientLanguages/ThamudicEpiPlatform/README.md`
- Modify: `general/README.md` if the integration index requires it

- [ ] Step 1: Document every major implementation area and its source/provenance.
- [ ] Step 2: Cite Unicode/CLDR/BCP 47 material and open-source inspiration at the exact feature documentation.
- [ ] Step 3: Document PDF libraries and licenses.
- [ ] Step 4: Document generated reports and model limitations.
- [ ] Step 5: Add source-code citation index for every maintained language implementation.

### Task 12: End-to-end verification and release integration

**Files:**
- Create: `ThamudicEpiPlatform/tests/e2e/test_pdf_dashboard_flow.py`
- Create: `ThamudicEpiPlatform/tests/fixtures/sample_historical_object.json`
- Create: `ThamudicEpiPlatform/tests/fixtures/sample_translation.json`
- Create: `general/applications/AncientLanguages/ThamudicEpiPlatform/tests/integration_manifest.json`

- [ ] Step 1: Run schema/database tests.
- [ ] Step 2: Run PDF import/export tests.
- [ ] Step 3: Run API/KPI tests.
- [ ] Step 4: Build web application.
- [ ] Step 5: Compile language clients.
- [ ] Step 6: Run Bash/PowerShell/CMD automation tests.
- [ ] Step 7: Verify generated PDF contains provenance, transliteration, translation, citations, and KPI data.
- [ ] Step 8: Verify general-repo integration manifests match nlp source hashes.
- [ ] Step 9: Run repository-wide documentation/link checks.
- [ ] Step 10: Record verification evidence before claiming completion.

## Acceptance Criteria

1. A valid historical-object PDF can be imported with page-level provenance, Unicode-aware script detection, and bounded resource use.
2. Research objects, readings, transliterations, literal translations, meaning translations, confidence, and citations can be exported into a structured PDF report.
3. PDF export has a machine-readable provenance sidecar and deterministic report metadata.
4. KPI endpoints and dashboard render the same metrics across the application and language clients.
5. All maintained language implementations expose the common API contracts and compile/build independently.
6. `nlp` remains the canonical research implementation and `general` contains a reproducible integration layer rather than divergent copied logic.
7. Bash, PowerShell, and Windows CMD automation can bootstrap dependencies, build, test, package, and optionally deploy the stack.
8. README and documentation files identify source material, licenses, dependencies, and implementation provenance.
9. Unsupported ancient-language translation and unavailable historical TTS never silently produce fabricated authoritative results.
10. Verification output is captured before any completion claim.
