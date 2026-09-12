# Chimera II OS Database Integration

## Purpose

The NLP repository can exchange ancient-script, OCR, translation and model metadata with the Chimera II OS RNN/LLM deep-learning subsystem through a stable relational contract.

## Logical domains

`chimera_node` records participating nodes and their advertised capabilities. `trust_status` is descriptive state; actual authentication and authorization must remain in the Chimera II security layer.

`chimera_model` and `chimera_model_version` identify model lineage independently from runtime nodes. A model artifact is addressed by URI and protected by a SHA-256 checksum.

`chimera_dataset` and `chimera_training_run` provide reproducibility metadata for training. Training metrics and hyperparameters are stored as structured text/JSON in dialects that support it.

`chimera_inference_event` records inference provenance, latency, confidence, hashes and evidence. This is deliberately separate from raw application text so sensitive or large payloads can remain in the application/object store.

`chimera_embedding` records vector dimensions and external vector artifacts. The database does not require a particular vector engine; adapters can map this contract to native vector columns/indexes where available.

`chimera_sync_event` records cross-node synchronization decisions. Only records that pass the Chimera trust/authentication policy should be applied to authoritative data.

## OLTP/OLAP separation

Use the integration schema as OLTP metadata. Stream or batch its immutable event records into the NLP warehouse for model, node, script and translation analytics. Avoid running large analytical scans against the operational node database.

## MDM relationship

NLP MDM canonical IDs should be referenced by `entity_type` + `entity_ref` when an NLP entity is exchanged with Chimera. External database IDs belong in MDM crosswalks, not in hard-coded application logic.

## Security

Never put API keys, passwords, private keys, session tokens or secure-boot secrets into these SQL tables. Store only public identity fingerprints, hashes, status and non-secret provenance. The operating system or deployment secret manager remains responsible for credentials.

## Synchronization model

Recommended flow:

`NLP OLTP -> CDC/event export -> MDM resolution -> Chimera trust check -> Chimera node ingest -> model/inference processing -> immutable event -> OLAP load`

Conflict resolution should be deterministic: canonical ID first, source-system priority second, verified scholarly record third, and explicit human review for unresolved conflicts.
