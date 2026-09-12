-- Portable analytical model. Adapt identity/time functions to the target warehouse.
CREATE TABLE dim_script (script_key BIGINT PRIMARY KEY, canonical_code VARCHAR(64) NOT NULL, script_name VARCHAR(200) NOT NULL, family_name VARCHAR(200), decipherment_status VARCHAR(32));
CREATE TABLE dim_model (model_key BIGINT PRIMARY KEY, model_name VARCHAR(256) NOT NULL, architecture VARCHAR(128), version VARCHAR(128), framework VARCHAR(128));
CREATE TABLE dim_language (language_key BIGINT PRIMARY KEY, language_code VARCHAR(64) NOT NULL, language_name VARCHAR(200));
CREATE TABLE dim_date (date_key INTEGER PRIMARY KEY, calendar_date DATE NOT NULL, year_number INTEGER, quarter_number INTEGER, month_number INTEGER, day_number INTEGER);
CREATE TABLE fact_ocr_translation (fact_key BIGINT PRIMARY KEY, date_key INTEGER NOT NULL, script_key BIGINT NOT NULL, model_key BIGINT, source_language_key BIGINT, target_language_key BIGINT, glyph_count BIGINT, token_count BIGINT, confidence DECIMAL(9,6), latency_ms DECIMAL(20,3), translation_count BIGINT, human_verified_count BIGINT, FOREIGN KEY(date_key) REFERENCES dim_date(date_key), FOREIGN KEY(script_key) REFERENCES dim_script(script_key), FOREIGN KEY(model_key) REFERENCES dim_model(model_key), FOREIGN KEY(source_language_key) REFERENCES dim_language(language_key), FOREIGN KEY(target_language_key) REFERENCES dim_language(language_key));
CREATE INDEX ix_fact_date ON fact_ocr_translation(date_key);
CREATE INDEX ix_fact_script ON fact_ocr_translation(script_key);
