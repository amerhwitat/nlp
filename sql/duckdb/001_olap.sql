CREATE SCHEMA IF NOT EXISTS nlp;
CREATE TABLE IF NOT EXISTS nlp.dim_script (script_key BIGINT PRIMARY KEY, canonical_code VARCHAR, script_name VARCHAR, family_name VARCHAR, decipherment_status VARCHAR);
CREATE TABLE IF NOT EXISTS nlp.dim_model (model_key BIGINT PRIMARY KEY, model_name VARCHAR, architecture VARCHAR, version VARCHAR, framework VARCHAR);
CREATE TABLE IF NOT EXISTS nlp.dim_language (language_key BIGINT PRIMARY KEY, language_code VARCHAR, language_name VARCHAR);
CREATE TABLE IF NOT EXISTS nlp.dim_date (date_key INTEGER PRIMARY KEY, calendar_date DATE, year_number INTEGER, quarter_number INTEGER, month_number INTEGER, day_number INTEGER);
CREATE TABLE IF NOT EXISTS nlp.fact_ocr_translation (fact_key BIGINT PRIMARY KEY, date_key INTEGER, script_key BIGINT, model_key BIGINT, source_language_key BIGINT, target_language_key BIGINT, glyph_count BIGINT, token_count BIGINT, confidence DECIMAL(9,6), latency_ms DECIMAL(20,3), translation_count BIGINT, human_verified_count BIGINT);
