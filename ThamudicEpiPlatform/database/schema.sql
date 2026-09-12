PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sources (
  id INTEGER PRIMARY KEY,
  source_key TEXT NOT NULL UNIQUE,
  institution TEXT,
  homepage TEXT,
  record_url TEXT,
  api_url TEXT,
  rights TEXT,
  image_policy TEXT,
  retrieved_at TEXT,
  content_sha256 TEXT,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS periods (
  id INTEGER PRIMARY KEY,
  period_key TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  parent_id INTEGER REFERENCES periods(id),
  start_year INTEGER,
  end_year INTEGER,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS objects (
  id INTEGER PRIMARY KEY,
  external_id TEXT UNIQUE,
  title TEXT NOT NULL,
  period_id INTEGER REFERENCES periods(id),
  object_type TEXT,
  culture TEXT,
  script TEXT,
  date_from INTEGER,
  date_to INTEGER,
  site TEXT,
  material TEXT,
  description TEXT,
  source_id INTEGER REFERENCES sources(id),
  image_url TEXT,
  rights TEXT,
  provenance TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS annotations (
  id INTEGER PRIMARY KEY,
  object_id INTEGER NOT NULL REFERENCES objects(id) ON DELETE CASCADE,
  x1 REAL, y1 REAL, x2 REAL, y2 REAL,
  glyph_candidate TEXT,
  transliteration_candidate TEXT,
  confidence REAL CHECK(confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
  reviewer TEXT,
  status TEXT NOT NULL DEFAULT 'candidate',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS readings (
  id INTEGER PRIMARY KEY,
  object_id INTEGER NOT NULL REFERENCES objects(id) ON DELETE CASCADE,
  reading_type TEXT NOT NULL,
  transliteration TEXT,
  arabic_interpretation TEXT,
  english_interpretation TEXT,
  reviewer TEXT,
  confidence REAL CHECK(confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
  status TEXT NOT NULL DEFAULT 'candidate',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS scan_sessions (
  id TEXT PRIMARY KEY,
  filename TEXT,
  status TEXT NOT NULL,
  progress REAL NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS asset_manifest (
  id INTEGER PRIMARY KEY,
  site_url TEXT NOT NULL,
  asset_url TEXT NOT NULL,
  asset_type TEXT NOT NULL,
  http_status INTEGER,
  content_type TEXT,
  sha256 TEXT,
  license_hint TEXT,
  retrieved_at TEXT,
  copied INTEGER NOT NULL DEFAULT 0,
  notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_objects_source ON objects(source_id);
CREATE INDEX IF NOT EXISTS idx_annotations_object ON annotations(object_id);
CREATE INDEX IF NOT EXISTS idx_readings_object ON readings(object_id);
CREATE INDEX IF NOT EXISTS idx_assets_site ON asset_manifest(site_url);
