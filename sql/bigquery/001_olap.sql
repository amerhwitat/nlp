CREATE SCHEMA IF NOT EXISTS `nlp.olap`;
CREATE TABLE IF NOT EXISTS `nlp.olap.dim_script` (script_key INT64, canonical_code STRING, script_name STRING, family_name STRING, decipherment_status STRING);
CREATE TABLE IF NOT EXISTS `nlp.olap.dim_model` (model_key INT64, model_name STRING, architecture STRING, version STRING, framework STRING);
CREATE TABLE IF NOT EXISTS `nlp.olap.dim_language` (language_key INT64, language_code STRING, language_name STRING);
CREATE TABLE IF NOT EXISTS `nlp.olap.dim_date` (date_key INT64, calendar_date DATE, year_number INT64, quarter_number INT64, month_number INT64, day_number INT64);
CREATE TABLE IF NOT EXISTS `nlp.olap.fact_ocr_translation` (fact_key INT64, date_key INT64, script_key INT64, model_key INT64, source_language_key INT64, target_language_key INT64, glyph_count INT64, token_count INT64, confidence NUMERIC, latency_ms NUMERIC, translation_count INT64, human_verified_count INT64)
PARTITION BY calendar_date;
