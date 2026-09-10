PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS objects (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  period_key TEXT,
  period_name TEXT,
  object_type TEXT,
  culture TEXT,
  script_key TEXT,
  language TEXT,
  date_start TEXT,
  date_end TEXT,
  site TEXT,
  region TEXT,
  country TEXT,
  current_location TEXT,
  material TEXT,
  technique TEXT,
  description TEXT,
  transliteration TEXT,
  translation_ar TEXT,
  translation_en TEXT,
  source_name TEXT,
  source_url TEXT,
  source_record_id TEXT,
  image_url TEXT,
  image_page_url TEXT,
  image_iiif TEXT,
  image_local_path TEXT,
  license TEXT,
  rights_notes TEXT,
  creator TEXT,
  provenance TEXT,
  bibliography TEXT,
  subjects TEXT,
  latitude REAL,
  longitude REAL,
  confidence REAL,
  reviewer TEXT,
  competing_readings TEXT,
  tags TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS annotations (
  id TEXT PRIMARY KEY,
  object_id TEXT NOT NULL,
  label TEXT,
  x REAL,
  y REAL,
  width REAL,
  height REAL,
  unicode_candidate TEXT,
  transliteration_candidate TEXT,
  confidence REAL,
  reviewer TEXT,
  notes TEXT,
  FOREIGN KEY(object_id) REFERENCES objects(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sources (
  id TEXT PRIMARY KEY,
  name TEXT UNIQUE,
  homepage TEXT,
  api_url TEXT,
  rights_policy TEXT,
  image_policy TEXT,
  enabled INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS database_meta (key TEXT PRIMARY KEY, value TEXT);
