PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS ocr_geometry_hypotheses (
  id INTEGER PRIMARY KEY,
  ocr_job_id INTEGER REFERENCES ocr_jobs(id) ON DELETE CASCADE,
  direction TEXT NOT NULL,
  rotation_degrees REAL NOT NULL DEFAULT 0,
  skew_degrees REAL NOT NULL DEFAULT 0,
  perspective_score REAL NOT NULL DEFAULT 0,
  weathering_score REAL NOT NULL DEFAULT 0,
  confidence REAL NOT NULL DEFAULT 0,
  operations_json TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS translation_results (
  id INTEGER PRIMARY KEY,
  object_id INTEGER REFERENCES objects(id) ON DELETE SET NULL,
  source_language TEXT NOT NULL,
  target_language TEXT NOT NULL,
  source_text TEXT NOT NULL,
  mode TEXT NOT NULL CHECK(mode IN ('literal','meaning','interlinear','scholarly')),
  output TEXT,
  confidence REAL CHECK(confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
  alternatives_json TEXT NOT NULL DEFAULT '[]',
  proof_notes_json TEXT NOT NULL DEFAULT '[]',
  provenance_json TEXT NOT NULL DEFAULT '{}',
  review_required INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS speech_proof_jobs (
  id INTEGER PRIMARY KEY,
  translation_id INTEGER REFERENCES translation_results(id) ON DELETE SET NULL,
  source_text TEXT NOT NULL,
  language TEXT NOT NULL,
  voice_profile TEXT,
  phoneme_json TEXT,
  proof_status TEXT NOT NULL DEFAULT 'pending',
  confidence REAL,
  audio_sha256 TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chatbot_sessions (
  id TEXT PRIMARY KEY,
  provider TEXT NOT NULL,
  model TEXT,
  evidence_json TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_translation_languages ON translation_results(source_language, target_language);
CREATE INDEX IF NOT EXISTS idx_speech_translation ON speech_proof_jobs(translation_id);
