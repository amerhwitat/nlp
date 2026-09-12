# Web Geospatial and Astronomy Integrations

## Geospatial

The web layer is designed for Three.js/WebGL local 3D scenes plus an optional CesiumJS adapter for globe-scale historical geography. OGC 3D Tiles is the preferred streaming representation for very large terrain, photogrammetry and point-cloud datasets.

The adapter can represent:

- WGS84 location
- elevation
- historical time intervals
- dynamic entities
- terrain and imagery layers
- photogrammetry
- point clouds
- clipping/classification regions
- time-dependent paths

The implementation is intentionally provider-neutral; hosted terrain or imagery services remain optional.

## Historical maps

Map layers can contain a validity interval so the same map canvas can display different political boundaries, settlements, roads, rivers and environmental states at different dates.

## Sky maps

A sky scene has:

- UTC/local time and calendar system
- observer latitude/longitude/elevation
- season
- horizon/landscape profile
- stars and constellations
- Sun/Moon/planet positions
- twilight and atmospheric state
- cultural constellation metadata
- ephemeris source and version

The viewer exposes a time slider so a historical event can show the night sky that would have been visible at that place and date. For high-precision work, an external astronomical engine is used rather than the simplified reference calculator.

## Sources researched

Cesium documents describe terrain tiling, photogrammetry, point clouds, 3D Tiles and time-dynamic visualization. Stellarium documents provide historical sky simulation, ephemerides, cultural constellations and astronomical scripting. OpenHistoricalMap/Running Reality demonstrates historical map visualization with time-aware geographic data.

## User experience

A single event workspace provides:

`Timeline | Map | 3D Scene | Sky | Characters | Evidence | Environment | Alternatives`

Selecting an event synchronizes all panels to the same time and observer perspective.
