# NLP Database Architecture

## Scope

The NLP repository now has a database-neutral persistence contract for its main applications: ancient-script OCR, inscription records, glyph observations, transliteration, translation candidates, model registry, model versions and inference telemetry.

## OLTP

OLTP is the authoritative operational layer. It is normalized around `script_family`, `source_document`, `inscription`, `glyph_observation`, `reading`, `translation_candidate`, `model_registry`, `model_version` and `inference_run`.

Operational records should be immutable where they represent scholarly evidence or model provenance. Corrections should create new readings/versions rather than destroying the original evidence.

## OLAP

The warehouse layer uses a star schema with script, model, language and date dimensions and OCR/translation facts. This supports confidence distributions, latency, throughput, model comparison, script coverage and human-verification rates without slowing the transactional database.

Warehouse implementations are supplied for Snowflake, BigQuery, DuckDB and SAP HANA, with a portable star-schema definition under `sql/olap/`.

## MDM

Master Data Management supplies canonical IDs and crosswalks for entities that cross applications or databases. A record can retain multiple external keys while resolving to one canonical entity. Provenance, license and attribution are stored with the master record.

SAP HANA has a dedicated column-store MDM implementation under `sql/saphana/002_mdm.sql`. HANA's column store is appropriate for analytical access while its SQL/DDL supports the database objects required by this design. citeturn0search3turn0search11

## Access

Microsoft Access/ACE is supported as a local desktop database for small projects and offline research. Access SQL supports `CREATE TABLE` and indexes, and Microsoft documents the Access-specific SQL data types and constraints. citeturn0search0turn0search6turn0search7

The repository supplies both Access DDL and `access/bootstrap_nlp_database.bas`. The VBA bootstrap creates the tables in the currently open `.accdb`. A binary `.accdb` is intentionally not checked into source control.

## Chimera II RNN/LLM database

`sql/chimera_ii/001_rnnllm_integration.sql` defines a shared integration contract for:

- trusted Chimera II nodes;
- model families and versions;
- datasets and provenance;
- training runs and metrics;
- inference events and latency/confidence;
- embeddings and vector artifacts;
- node-to-node synchronization events.

The schema stores hashes and artifact URIs rather than secrets. Credentials, private keys and access tokens must remain in the operating system's secret-management layer.

## SAP HANA notes

HANA supports column-store tables and explicit load operations. Its column store uses main and delta structures, allowing read-optimized compressed main storage while writes are accumulated in delta storage. citeturn0search2turn0search11

The HANA scripts therefore use `CREATE COLUMN TABLE` for OLTP/MDM objects that benefit from columnar access and provide a separate analytical star schema. HANA's SQL reference also exposes table/column metadata through system views such as `TABLE_COLUMNS`. citeturn0search1

## Big-data MDM strategy

For BigQuery/Snowflake/DuckDB, MDM is implemented as warehouse dimensions plus canonical IDs/crosswalks. The operational source of truth remains OLTP; the warehouse is fed through CDC/batch/stream pipelines. Do not make an analytical warehouse the authoritative transactional identity store.

## Data governance

Every externally sourced inscription, glyph image, corpus, model, dataset and translation should carry provenance and license metadata. Uncertain or damaged readings remain explicitly uncertain. A model-generated translation is a candidate until human or scholarly verification marks it verified.
