# NLP / Ancient Scripts Intelligence Research Toolkit

This repository contains Python, C++, .NET, Visual C++, and web implementations for Thamudic and Ancient North Arabian research, expanded into a language-neutral ancient-script OCR, epigraphy, transliteration and low-resource translation platform.

## Ancient Visual Research Suite

`AncientVisualResearchSuite/` is the standalone historical visualization and simulation layer. It combines ancient-language evidence, archaeological findings, artifacts, excavation records, historical geography, environment, astronomy and 3D/4D visualization without replacing the underlying evidence model.

It supports synchronized visualization of historical events and simultaneous sub-events, archaeological sites and excavation findings, historical maps and environmental layers, terrain/water/vegetation/weather/atmosphere, evidence-linked parametric characters, artifact and inscription 3D representations, seasonal and historical night-sky views, stars/constellations/Sun/Moon/planets through astronomy adapters, timeline playback and alternate scenario branches, the 128D state model, neural sequence/comprehension feature extraction, and Three.js/WebGL/Cesium-compatible web visualization plus Android/iOS reference APIs.

The suite explicitly separates observed evidence, supported interpretation, inference, speculation and visualization. A generated character or reconstructed building is therefore traceable to its evidence rather than presented as an undocumented fact.

See `AncientVisualResearchSuite/README.md`, `docs/HISTORICAL_SIMULATION.md` and `docs/WEB_GEOSPATIAL_ASTRONOMY.md`.

## Start here

Use the repository build orchestration layer:

```bat
build-tools\build.bat
```

PowerShell:

```powershell
.\build-tools\build.ps1
```

POSIX:

```bash
./build-tools/build.sh
```

For the new historical visualization suite:

```powershell
AncientVisualResearchSuite\scripts\build\build-all.ps1
```

or

```bash
AncientVisualResearchSuite/scripts/build/build-all.sh
```

## ThamudicScan

Public deployment:

https://thamudicscan-s3wz30.public.builtwithrocket.new/

The deployment is the reference web experience for the Thamudic / North Arabian tooling. The reusable implementation lives under `ancient_scripts/`, `python/ancient_scripts/`, `cpp/ancient_scripts/`, and `typescript/ancient-scripts/`.

## Architecture

`image -> detection -> segmentation -> glyph recognition -> script identification -> Unicode/transliteration -> normalization -> lexical/morphological analysis -> RNN/LLM translation -> retrieval/context -> confidence -> scholarly review -> OLTP -> OLAP/MDM -> historical scene -> map/terrain -> sky/environment -> 3D/4D visualization`

The architecture intentionally preserves uncertainty, damaged signs, alternate readings and provenance.

## Database platform

The repository includes a database portability layer for the main NLP applications and the Chimera II RNN/LLM integration:

- **OLTP:** PostgreSQL, MySQL, MariaDB, SQLite, SQL Server, Oracle, IBM Db2 and SAP HANA.
- **OLAP:** portable star schema plus SAP HANA, Snowflake, BigQuery and DuckDB warehouse definitions.
- **MDM:** canonical entity/crosswalk/provenance model, with SAP HANA and warehouse-compatible patterns.
- **Microsoft Access:** ACE/Jet DDL plus a VBA bootstrapper for creating an `.accdb` database locally.
- **Chimera II:** node, trust status, model/model-version, dataset, training-run, inference-event, embedding and synchronization records.

See `database/README.md`, `docs/DATABASE_ARCHITECTURE.md`, `docs/CHIMERA_II_DATABASE_INTEGRATION.md` and `sql/`.

## Ancient-script coverage

The registry includes Dadanitic, Taymanitic, Dumaitic, Safaitic, Hismaic, Thamudic B/C/D, Himaitic/Thamudic F, Nabataean, Old Arabic, Aramaic, Phoenician, Paleo-Hebrew, Ugaritic, Old Persian, Egyptian hieroglyphs, Sumerian, Akkadian and Hittite. Coverage status is represented explicitly rather than implying that every script is equally deciphered.

## RNN / LLM engine

`python/ancient_scripts/rnnllm.py` provides the common inference contract for low-resource models and ensembles. The design supports RNN/LSTM/GRU, Transformer, state-space/Mamba-style models, retrieval augmentation, lexicons and contextual evidence without pretending that pretrained weights are present in source control.

## Implementations

- `ancient_scripts/` — language-neutral schemas and architecture.
- `python/ancient_scripts/` — Python research/inference engine.
- `cpp/ancient_scripts/` — C++ core interface for high-performance implementations.
- `rust/ancient_scripts/` — memory-safe core primitives.
- `typescript/ancient-scripts/` — browser/service interface types.
- `data/ancient_scripts/` — extensible script-family registry.
- `data/ancient_north_arabian/` — canonical Old North Arabian Unicode registry.
- `cpp/thamudic/` — C++20 Thamudic implementation.
- `vcpp/` — Visual Studio native Windows implementation.
- `dotnet/` — CLI, WPF and web implementations.
- `python/thamudic/` — Python Unicode/UTF-8/transliteration API.
- `ThamudicScan/` — deployment documentation and integration point.
- `AncientVisualResearchSuite/` — historical event, archaeology, map, environment, sky and 3D/4D visualization platform.

## Research integration

The design is informed by public research and software for ancient-script OCR and translation, including OCIANA, MNAMON, CDLI Sumerian-English machine translation work, MARDUK's hybrid Mamba/Transformer/RAG Akkadian approach, and Old Persian OCR research. External code is not copied merely because it is public: license, attribution and redistribution rights are recorded before integration.

## Unicode

The existing `data/ancient_north_arabian/alphabet.json` remains the canonical Unicode registry for Old North Arabian U+10A80–U+10A9F. Variant historical forms must not be represented as fabricated Unicode code points.

## Documentation

- `docs/ANCIENT_SCRIPTS_RESEARCH.md`
- `docs/DATABASE_ARCHITECTURE.md`
- `docs/CHIMERA_II_DATABASE_INTEGRATION.md`
- `database/README.md`
- `ancient_scripts/README.md`
- `ancient_scripts/core/schema.json`
- `AncientVisualResearchSuite/README.md`
- `AncientVisualResearchSuite/docs/HISTORICAL_SIMULATION.md`
- `AncientVisualResearchSuite/docs/WEB_GEOSPATIAL_ASTRONOMY.md`

Modern .NET 6 is `net6.0`; .NET Framework targets use TFMs such as `net48`. There is no Microsoft target named “.NET Framework 6.0”.
