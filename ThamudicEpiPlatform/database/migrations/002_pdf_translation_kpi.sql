PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS pdf_imports (
  id INTEGER PRIMARY KEY,
  source_id INTEGER REFERENCES sources(id),
  filename TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  page_count INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'queued',
  extracted_chars INTEGER NOT NULL DEFAULT 0,
  warnings_json TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  completed_at TEXT
);

CREATE TABLE IF NOT EXISTS pdf_exports (
  id INTEGER PRIMARY KEY,
  report_id TEXT NOT NULL UNIQUE,
  object_id INTEGER REFERENCES objects(id),
  filename TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  report_type TEXT NOT NULL,
  manifest_json TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS translation_results (
  id INTEGER PRIMARY KEY,
  object_id INTEGER REFERENCES objects(id) ON DELETE CASCADE,
  source_language TEXT NOT NULL,
  source_script TEXT,
  source_text TEXT NOT NULL,
  transliteration TEXT,
  target_language TEXT NOT NULL,
  literal_translation TEXT,
  meaning_translation TEXT,
  confidence REAL CHECK(confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
  model_id TEXT,
  model_version TEXT,
  provenance_json TEXT,
  status TEXT NOT NULL DEFAULT 'candidate',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS kpi_snapshots (
  id INTEGER PRIMARY KEY,
  application TEXT NOT NULL,
  generated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  schema_version TEXT NOT NULL,
  metrics_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pdf_imports_source ON pdf_imports(source_id);
CREATE INDEX IF NOT EXISTS idx_pdf_imports_sha ON pdf_imports(sha256);
CREATE INDEX IF NOT EXISTS idx_pdf_exports_object ON pdf_exports(object_id);
CREATE INDEX IF NOT EXISTS idx_translation_results_object ON translation_results(object_id);
CREATE INDEX IF NOT EXISTS idx_translation_results_language ON translation_results(source_language, target_language);
CREATE INDEX IF NOT EXISTS idx_kpi_snapshots_application ON kpi_snapshots(application, generated_at);
