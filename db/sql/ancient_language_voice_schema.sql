-- Portable OLTP core; adapt identity/boolean syntax per target SQL engine.
CREATE TABLE IF NOT EXISTS ALPHABET (
  alphabet_id VARCHAR(128) PRIMARY KEY,
  language_stage_id VARCHAR(128) NOT NULL,
  script_id VARCHAR(32) NOT NULL,
  name VARCHAR(255) NOT NULL,
  direction VARCHAR(8) NOT NULL,
  source_id VARCHAR(128),
  source_version VARCHAR(64),
  validation_status VARCHAR(32) NOT NULL
);

CREATE TABLE IF NOT EXISTS ALPHABET_MEMBER (
  alphabet_member_id VARCHAR(128) PRIMARY KEY,
  alphabet_id VARCHAR(128) NOT NULL,
  unicode_character_id VARCHAR(128),
  glyph_or_sign_id VARCHAR(128),
  transliteration VARCHAR(128),
  historical_variant VARCHAR(255),
  pronunciation_profile_id VARCHAR(128),
  source_id VARCHAR(128),
  confidence DECIMAL(6,5),
  FOREIGN KEY (alphabet_id) REFERENCES ALPHABET(alphabet_id)
);

CREATE TABLE IF NOT EXISTS UTF8_ENCODING (
  unicode_character_id VARCHAR(128) PRIMARY KEY,
  codepoint VARCHAR(16) NOT NULL,
  utf8_hex VARCHAR(64) NOT NULL,
  normalization_form VARCHAR(16),
  unicode_version VARCHAR(32) NOT NULL,
  source_checksum VARCHAR(128),
  validation_status VARCHAR(32) NOT NULL
);

CREATE TABLE IF NOT EXISTS PRONUNCIATION_PROFILE (
  pronunciation_profile_id VARCHAR(128) PRIMARY KEY,
  language_stage_id VARCHAR(128) NOT NULL,
  dialect_id VARCHAR(128),
  transliteration_system_id VARCHAR(128),
  pronunciation_type VARCHAR(32) NOT NULL,
  locale VARCHAR(32),
  ipa VARCHAR(1024),
  grapheme_to_phoneme VARCHAR(4096),
  stress_metadata VARCHAR(2048),
  backend VARCHAR(64) NOT NULL,
  confidence DECIMAL(6,5),
  source_id VARCHAR(128),
  model_version VARCHAR(128),
  validation_status VARCHAR(32) NOT NULL
);

CREATE TABLE IF NOT EXISTS VOICE_PROFILE (
  voice_profile_id VARCHAR(128) PRIMARY KEY,
  pronunciation_profile_id VARCHAR(128) NOT NULL,
  provider VARCHAR(128) NOT NULL,
  voice_name VARCHAR(255),
  locale VARCHAR(32),
  is_modern_approximation INTEGER NOT NULL DEFAULT 0,
  provenance_note VARCHAR(2048),
  FOREIGN KEY (pronunciation_profile_id) REFERENCES PRONUNCIATION_PROFILE(pronunciation_profile_id)
);

CREATE TABLE IF NOT EXISTS AUDIO_ASSET (
  audio_asset_id VARCHAR(128) PRIMARY KEY,
  pronunciation_profile_id VARCHAR(128) NOT NULL,
  uri VARCHAR(2048) NOT NULL,
  mime_type VARCHAR(128),
  checksum VARCHAR(128),
  rights VARCHAR(2048),
  source_id VARCHAR(128),
  FOREIGN KEY (pronunciation_profile_id) REFERENCES PRONUNCIATION_PROFILE(pronunciation_profile_id)
);
