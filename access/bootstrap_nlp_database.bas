Attribute VB_Name = "BootstrapNlpDatabase"
Option Compare Database
Option Explicit

' Creates the NLP schema in the current Access database.
' Run from the VBA editor after creating an empty .accdb file.
Public Sub BuildNlpDatabase()
    Dim db As DAO.Database
    Set db = CurrentDb

    ExecuteIgnoreExists db, "CREATE TABLE script_family (script_id TEXT(36), canonical_code TEXT(64), name TEXT(200), family TEXT(200), writing_direction TEXT(16), decipherment_status TEXT(32), created_at DATETIME, CONSTRAINT pk_script_family PRIMARY KEY (script_id), CONSTRAINT uq_script_code UNIQUE (canonical_code))"
    ExecuteIgnoreExists db, "CREATE TABLE source_document (document_id TEXT(36), external_id TEXT(256), title LONGTEXT, uri LONGTEXT, media_type TEXT(128), checksum_sha256 TEXT(64), captured_at DATETIME, created_at DATETIME, CONSTRAINT pk_source_document PRIMARY KEY (document_id))"
    ExecuteIgnoreExists db, "CREATE TABLE inscription (inscription_id TEXT(36), document_id TEXT(36), script_id TEXT(36), scholarly_id TEXT(256), language_code TEXT(64), transliteration LONGTEXT, translation LONGTEXT, confidence DOUBLE, uncertainty LONGTEXT, provenance LONGTEXT, created_at DATETIME, CONSTRAINT pk_inscription PRIMARY KEY (inscription_id))"
    ExecuteIgnoreExists db, "CREATE TABLE glyph_observation (glyph_id TEXT(36), inscription_id TEXT(36), sequence_no INTEGER, unicode_codepoint TEXT(32), glyph_label TEXT(128), bounding_box LONGTEXT, image_uri LONGTEXT, recognized_text LONGTEXT, confidence DOUBLE, damaged BIT, alternate_readings LONGTEXT, created_at DATETIME, CONSTRAINT pk_glyph PRIMARY KEY (glyph_id), CONSTRAINT uq_glyph UNIQUE(inscription_id,sequence_no))"
    ExecuteIgnoreExists db, "CREATE TABLE reading (reading_id TEXT(36), inscription_id TEXT(36), reading_text LONGTEXT, transliteration LONGTEXT, normalized_text LONGTEXT, confidence DOUBLE, reviewer TEXT(256), evidence LONGTEXT, created_at DATETIME, CONSTRAINT pk_reading PRIMARY KEY (reading_id))"
    ExecuteIgnoreExists db, "CREATE TABLE translation_candidate (candidate_id TEXT(36), reading_id TEXT(36), target_language TEXT(32), translation LONGTEXT, model_id TEXT(36), confidence DOUBLE, rank_no INTEGER, evidence LONGTEXT, human_verified BIT, created_at DATETIME, CONSTRAINT pk_translation PRIMARY KEY(candidate_id))"
    ExecuteIgnoreExists db, "CREATE TABLE model_registry (model_id TEXT(36), model_name TEXT(256), architecture TEXT(128), task TEXT(128), provider TEXT(256), license TEXT(256), metadata LONGTEXT, created_at DATETIME, CONSTRAINT pk_model PRIMARY KEY(model_id), CONSTRAINT uq_model_name UNIQUE(model_name))"
    ExecuteIgnoreExists db, "CREATE TABLE model_version (model_version_id TEXT(36), model_id TEXT(36), version TEXT(128), artifact_uri LONGTEXT, checksum_sha256 TEXT(64), parameters LONG, framework TEXT(128), metrics LONGTEXT, created_at DATETIME, CONSTRAINT pk_model_version PRIMARY KEY(model_version_id), CONSTRAINT uq_model_version UNIQUE(model_id,version))"
    ExecuteIgnoreExists db, "CREATE TABLE inference_run (run_id TEXT(36), model_version_id TEXT(36), inscription_id TEXT(36), operation TEXT(64), input_hash TEXT(64), output LONGTEXT, latency_ms DOUBLE, confidence DOUBLE, created_at DATETIME, CONSTRAINT pk_inference PRIMARY KEY(run_id))"

    ExecuteIgnoreExists db, "CREATE INDEX ix_inscription_script ON inscription(script_id)"
    ExecuteIgnoreExists db, "CREATE INDEX ix_glyph_inscription ON glyph_observation(inscription_id)"
    MsgBox "NLP database schema created.", vbInformation
End Sub

Private Sub ExecuteIgnoreExists(ByVal db As DAO.Database, ByVal sqlText As String)
    On Error GoTo IgnoreExists
    db.Execute sqlText, dbFailOnError
IgnoreExists:
    Err.Clear
    On Error GoTo 0
End Sub
