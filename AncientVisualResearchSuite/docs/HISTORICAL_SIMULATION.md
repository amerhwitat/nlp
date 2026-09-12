# Historical Event Simulation

## Scene dimensions

Each scene combines:

1. **Space** — site coordinates, terrain, buildings, artifact locations and movement paths.
2. **Time** — absolute date/time, duration, chronology uncertainty and synchronized sub-events.
3. **Perspective** — observer/camera position, field of view and visibility constraints.
4. **Energy/light** — sunlight, moonlight, artificial fire/light and material response metadata.
5. **Events** — causally or temporally linked actions.
6. **Objects/materials** — artifacts, architecture, clothing, tools, terrain and environmental state.
7. **Information** — source IDs, confidence, uncertainty and competing reconstructions.
8. **Cognition** — neural feature vectors, labels, hypotheses and observer annotations.

The implementation maps these domains into the project's 128D state representation while treating the dimensions as computational state variables rather than additional physical spatial axes.

## Reconstruction classes

- **Observed:** directly visible/measured evidence.
- **Supported:** interpretation supported by multiple credible sources.
- **Inferred:** a model required to make a visualization continuous or complete.
- **Speculative:** a hypothesis or scenario without sufficient evidence.
- **Visualization:** rendering choices that do not assert historical fact.

Every character, building, event, map layer and astronomical layer can carry evidence IDs.

## Event playback

The timeline supports simultaneous events. Each event has a start, duration, actors, location and evidence links. The viewer can scrub time, pause, change camera perspective, inspect evidence, and branch into alternate scenarios.

## Characters

Characters are parametric agents. The reference generator creates reproducible non-identifying placeholders from role/activity/evidence inputs. It does not invent a historically named person unless that identity is explicitly supplied by the source dataset.

## Environment

Environment layers may include terrain, elevation, water, vegetation, weather, cloud cover, visibility and atmospheric effects. External GIS/DEM/photogrammetry data can be attached through adapters.

## Astronomy

The lightweight sky layer is suitable for interactive visualization and data plumbing. Research-grade archaeoastronomy should connect an authoritative ephemeris engine and record engine version, time scale, coordinates, refraction assumptions, catalogue and uncertainty.

Stellarium is a useful reference because it provides historical-sky features, time control, realistic atmosphere and astronomical object catalogues. CesiumJS is a useful reference for time-dynamic geospatial visualization, terrain, photogrammetry and 3D Tiles.

## Methodological standard

Historical visualization must clearly distinguish evidence from reconstruction. The architecture follows the London Charter-oriented principle that inferred visualization should remain traceable and explainable.
