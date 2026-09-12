# NLP Database Platform

This directory defines the persistent data layer for the NLP and Ancient Scripts applications and the integration contract with the Chimera II OS RNN/LLM deep-learning module.

## Architecture

The database model is split into three compatible layers:

1. **OLTP** — normalized operational records for projects, documents, inscriptions, glyph observations, readings, translations, models, inference runs and provenance.
2. **OLAP** — dimensional facts and dimensions for OCR/translation/model-performance analytics.
3. **MDM** — master-data entities, canonical identities, crosswalks, aliases and source provenance. This is the interoperability layer for NLP and Chimera II nodes.

The canonical logical model is database-neutral. Dialect scripts adapt types, identity generation, indexes and analytical features to each engine.

## SQL targets

The `sql/` tree provides deployable starting schemas for widely used relational and analytical engines:

- PostgreSQL
- MySQL
- MariaDB
- SQLite
- Microsoft SQL Server
- Oracle Database
- IBM Db2
- SAP HANA / HANA Cloud
- Snowflake
- Google BigQuery
- DuckDB
- Microsoft Access / ACE SQL

These scripts intentionally use a portable core and engine-specific extensions only where useful. They are not claimed to make every SQL dialect byte-for-byte interchangeable.

## Chimera II integration

`sql/chimera_ii/` contains the integration schema for the Chimera II OS neural/RNN/LLM subsystem. It records nodes, trust identities, model versions, datasets, training runs, inference events, embeddings, synchronization events and model/data lineage. It does not store credentials or private keys.

The MDM crosswalk allows the same model, script, inscription, dataset or node to have stable identities across separate databases and Chimera II nodes.

## Access

Microsoft Access is supported through ACE/Jet-compatible DDL and a VBA bootstrap script. An `.accdb` file is a binary container and therefore is not represented as source text in Git. Run `access/bootstrap_nlp_database.bas` from Access or the included Windows automation procedure to create the database and tables.

## Deployment order

1. Deploy the OLTP schema.
2. Load reference/master data.
3. Deploy MDM crosswalks.
4. Deploy OLAP dimensions/facts and ETL views.
5. Configure the Chimera II integration schema.
6. Point applications at the dialect-specific connection adapter.

See `docs/DATABASE_ARCHITECTURE.md` and `docs/CHIMERA_II_DATABASE_INTEGRATION.md` for the full design.
