CREATE TABLE IF NOT EXISTS DIM_ALPHABET (
  alphabet_key INTEGER PRIMARY KEY,
  alphabet_id VARCHAR(128) NOT NULL,
  language_stage_id VARCHAR(128) NOT NULL,
  script_id VARCHAR(32) NOT NULL,
  direction VARCHAR(8) NOT NULL
);

CREATE TABLE IF NOT EXISTS DIM_ENCODING (
  encoding_key INTEGER PRIMARY KEY,
  codepoint VARCHAR(16) NOT NULL,
  utf8_hex VARCHAR(64) NOT NULL,
  unicode_version VARCHAR(32) NOT NULL
);

CREATE TABLE IF NOT EXISTS DIM_PRONUNCIATION (
  pronunciation_key INTEGER PRIMARY KEY,
  language_stage_id VARCHAR(128) NOT NULL,
  pronunciation_type VARCHAR(32) NOT NULL,
  locale VARCHAR(32),
  backend VARCHAR(64) NOT NULL,
  confidence DECIMAL(6,5)
);

CREATE TABLE IF NOT EXISTS FACT_LANGUAGE_TEXT_VOICE (
  fact_key INTEGER PRIMARY KEY,
  alphabet_key INTEGER,
  encoding_key INTEGER,
  pronunciation_key INTEGER,
  character_count INTEGER,
  speech_requested INTEGER NOT NULL DEFAULT 0,
  speech_available INTEGER NOT NULL DEFAULT 0,
  observation_confidence DECIMAL(6,5)
);
