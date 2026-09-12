# Research Sources and Feature Rationale

The architecture was expanded after reviewing current public capabilities in historical mapping, geospatial streaming and astronomy visualization.

## Historical visualization

- Running Reality / OpenHistoricalMap documentation demonstrates time-aware historical maps, structured geographic history and traceable citations.
- The London Charter-oriented guidance emphasizes distinguishing data visualization from inferred reconstruction and documenting methodological choices.

## Geospatial / 3D

- Cesium documentation covers terrain tiling, photogrammetry, point clouds, 3D Tiles and time-dependent visualization.
- OGC 3D Tiles is used as an open streaming target for large heterogeneous geospatial datasets.
- Heritage projects using Cesium demonstrate georeferenced monument models, WGS84 scenes and cultural-heritage metadata.

## Astronomy

- Stellarium provides realistic skies, large star catalogues, cultural constellations, time control, ephemerides, atmosphere and historical astronomical visualization.
- The suite therefore models astronomy as a synchronized event layer rather than a decorative background.

## Feature decisions

These sources motivated the addition of:

1. time-dynamic event playback;
2. georeferenced historical scenes;
3. terrain/photogrammetry/point-cloud adapters;
4. evidence-linked dynamic characters;
5. environmental state layers;
6. seasonal/night-sky reconstruction;
7. astronomy provenance and ephemeris metadata;
8. alternate reconstruction branches;
9. 128D state synchronization between geometry, time, perspective, events, objects, information and cognition.

Only architectural concepts and interoperable standards are adopted. Proprietary source code and restricted datasets are not copied into this repository.
