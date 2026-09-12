CREATE SCHEMA IF NOT EXISTS `nlp.mdm`;
CREATE TABLE IF NOT EXISTS `nlp.mdm.entity` (entity_id STRING, entity_type STRING, canonical_name STRING, status STRING, source_system STRING, source_key STRING, effective_from TIMESTAMP, effective_to TIMESTAMP, record_hash STRING, attributes JSON);
CREATE TABLE IF NOT EXISTS `nlp.mdm.crosswalk` (crosswalk_id STRING, entity_id STRING, system_code STRING, external_key STRING, match_method STRING, match_confidence NUMERIC, verified BOOL, created_at TIMESTAMP);
CREATE TABLE IF NOT EXISTS `nlp.mdm.provenance` (provenance_id STRING, entity_id STRING, source_uri STRING, source_version STRING, license STRING, attribution STRING, retrieved_at TIMESTAMP);
