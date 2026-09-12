CREATE TABLE IF NOT EXISTS ocr_jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  filename TEXT,
  source_sha256 TEXT NOT NULL,
  engine TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'completed',
  confidence REAL,
  script_candidates_json TEXT,
  warnings_json TEXT,
  preprocessing_json TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_ocr_jobs_sha256 ON ocr_jobs(source_sha256);
CREATE INDEX IF NOT EXISTS idx_ocr_jobs_engine ON ocr_jobs(engine);
CREATE INDEX IF NOT EXISTS idx_ocr_jobs_status ON ocr_jobs(status);
