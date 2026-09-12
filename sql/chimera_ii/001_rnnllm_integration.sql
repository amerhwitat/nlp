-- Shared logical schema. Use UUID/GUID/native identity types in dialect adapters.
CREATE TABLE chimera_node (
  node_id VARCHAR(64) PRIMARY KEY,
  node_name VARCHAR(256) NOT NULL UNIQUE,
  node_version VARCHAR(128),
  architecture VARCHAR(64),
  endpoint_uri VARCHAR(1024),
  trust_status VARCHAR(32) NOT NULL DEFAULT 'pending',
  public_identity_fingerprint VARCHAR(256),
  capabilities TEXT,
  last_seen_at TIMESTAMP,
  created_at TIMESTAMP NOT NULL
);
CREATE TABLE chimera_model (
  model_id VARCHAR(64) PRIMARY KEY,
  model_name VARCHAR(256) NOT NULL,
  family VARCHAR(128),
  architecture VARCHAR(128),
  task VARCHAR(128),
  framework VARCHAR(128),
  license VARCHAR(256),
  created_at TIMESTAMP NOT NULL
);
CREATE TABLE chimera_model_version (
  model_version_id VARCHAR(64) PRIMARY KEY,
  model_id VARCHAR(64) NOT NULL,
  version VARCHAR(128) NOT NULL,
  artifact_uri VARCHAR(2048),
  checksum_sha256 CHAR(64),
  parameter_count BIGINT,
  quantization VARCHAR(64),
  metrics TEXT,
  lineage TEXT,
  created_at TIMESTAMP NOT NULL
);
CREATE TABLE chimera_dataset (
  dataset_id VARCHAR(64) PRIMARY KEY,
  dataset_name VARCHAR(256) NOT NULL,
  version VARCHAR(128),
  modality VARCHAR(64),
  license VARCHAR(256),
  checksum_sha256 CHAR(64),
  provenance TEXT,
  created_at TIMESTAMP NOT NULL
);
CREATE TABLE chimera_training_run (
  training_run_id VARCHAR(64) PRIMARY KEY,
  model_version_id VARCHAR(64) NOT NULL,
  dataset_id VARCHAR(64) NOT NULL,
  node_id VARCHAR(64),
  started_at TIMESTAMP NOT NULL,
  finished_at TIMESTAMP,
  status VARCHAR(32) NOT NULL,
  hyperparameters TEXT,
  metrics TEXT,
  artifact_uri VARCHAR(2048)
);
CREATE TABLE chimera_inference_event (
  inference_id VARCHAR(64) PRIMARY KEY,
  model_version_id VARCHAR(64) NOT NULL,
  node_id VARCHAR(64),
  source_type VARCHAR(64),
  source_ref VARCHAR(512),
  input_hash CHAR(64),
  output_hash CHAR(64),
  confidence DECIMAL(9,6),
  latency_ms DECIMAL(20,3),
  token_count BIGINT,
  evidence TEXT,
  created_at TIMESTAMP NOT NULL
);
CREATE TABLE chimera_embedding (
  embedding_id VARCHAR(64) PRIMARY KEY,
  model_version_id VARCHAR(64) NOT NULL,
  entity_type VARCHAR(64) NOT NULL,
  entity_ref VARCHAR(512) NOT NULL,
  vector_format VARCHAR(64) NOT NULL,
  dimensions INTEGER NOT NULL,
  vector_uri VARCHAR(2048),
  vector_checksum CHAR(64),
  created_at TIMESTAMP NOT NULL
);
CREATE TABLE chimera_sync_event (
  sync_event_id VARCHAR(64) PRIMARY KEY,
  source_node_id VARCHAR(64) NOT NULL,
  target_node_id VARCHAR(64) NOT NULL,
  entity_type VARCHAR(64) NOT NULL,
  entity_ref VARCHAR(512) NOT NULL,
  operation VARCHAR(32) NOT NULL,
  payload_hash CHAR(64) NOT NULL,
  trust_decision VARCHAR(32) NOT NULL,
  occurred_at TIMESTAMP NOT NULL
);
CREATE INDEX ix_chimera_model_version_model ON chimera_model_version(model_id);
CREATE INDEX ix_chimera_training_model ON chimera_training_run(model_version_id);
CREATE INDEX ix_chimera_inference_model ON chimera_inference_event(model_version_id);
CREATE INDEX ix_chimera_sync_nodes ON chimera_sync_event(source_node_id,target_node_id);
