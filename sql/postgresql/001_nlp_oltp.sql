CREATE SCHEMA IF NOT EXISTS nlp;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS nlp.script_family (
  script_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  canonical_code VARCHAR(64) NOT NULL UNIQUE,
  name VARCHAR(200) NOT NULL,
  family VARCHAR(200),
  writing_direction VARCHAR(16) NOT NULL DEFAULT 'unknown',
  decipherment_status VARCHAR(32) NOT NULL DEFAULT 'partial',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS nlp.source_document (
  document_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  external_id VARCHAR(256), title TEXT, uri TEXT, media_type VARCHAR(128),
  checksum_sha256 CHAR(64), captured_at TIMESTAMPTZ, created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS nlp.inscription (
  inscription_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id UUID REFERENCES nlp.source_document(document_id),
  script_id UUID NOT NULL REFERENCES nlp.script_family(script_id),
  scholarly_id VARCHAR(256), language_code VARCHAR(64),
  transliteration TEXT, translation TEXT, confidence NUMERIC(6,5),
  uncertainty TEXT, provenance JSONB, created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS nlp.glyph_observation (
  glyph_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  inscription_id UUID NOT NULL REFERENCES nlp.inscription(inscription_id) ON DELETE CASCADE,
  sequence_no INTEGER NOT NULL, unicode_codepoint VARCHAR(32),
  glyph_label VARCHAR(128), bounding_box JSONB, image_uri TEXT,
  recognized_text TEXT, confidence NUMERIC(6,5), damaged BOOLEAN NOT NULL DEFAULT FALSE,
  alternate_readings JSONB, created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(inscription_id, sequence_no)
);
CREATE TABLE IF NOT EXISTS nlp.reading (
  reading_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  inscription_id UUID NOT NULL REFERENCES nlp.inscription(inscription_id) ON DELETE CASCADE,
  reading_text TEXT NOT NULL, transliteration TEXT, normalized_text TEXT,
  confidence NUMERIC(6,5), reviewer VARCHAR(256), evidence JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS nlp.translation_candidate (
  candidate_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  reading_id UUID NOT NULL REFERENCES nlp.reading(reading_id) ON DELETE CASCADE,
  target_language VARCHAR(32) NOT NULL, translation TEXT NOT NULL,
  model_id UUID, confidence NUMERIC(6,5), rank_no INTEGER,
  evidence JSONB, human_verified BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS nlp.model_registry (
  model_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  model_name VARCHAR(256) NOT NULL UNIQUE, architecture VARCHAR(128), task VARCHAR(128),
  provider VARCHAR(256), license VARCHAR(256), metadata JSONB, created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS nlp.model_version (
  model_version_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  model_id UUID NOT NULL REFERENCES nlp.model_registry(model_id),
  version VARCHAR(128) NOT NULL, artifact_uri TEXT, checksum_sha256 CHAR(64),
  parameters BIGINT, framework VARCHAR(128), metrics JSONB, created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(model_id, version)
);
CREATE TABLE IF NOT EXISTS nlp.inference_run (
  run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  model_version_id UUID REFERENCES nlp.model_version(model_version_id),
  inscription_id UUID REFERENCES nlp.inscription(inscription_id),
  operation VARCHAR(64) NOT NULL, input_hash CHAR(64), output JSONB,
  latency_ms NUMERIC(18,3), confidence NUMERIC(6,5), created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_inscription_script ON nlp.inscription(script_id);
CREATE INDEX IF NOT EXISTS ix_glyph_inscription ON nlp.glyph_observation(inscription_id);
CREATE INDEX IF NOT EXISTS ix_inference_model ON nlp.inference_run(model_version_id);
