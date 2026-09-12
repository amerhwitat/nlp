# Ancient Visual Research Suite

Unified historical-event, archaeology, ancient-language, artifact, 3D reconstruction, environment, astronomy and neural-comprehension platform.

## Purpose

This standalone subdirectory combines the Thamudic/North Arabian scanner and Ancient Languages/Artifacts Database with the Dimensional Studio architecture. It models historical findings as evidence-backed, time-aware scenes that can be explored in 2D maps, 3D/4D environments and seasonal/night-sky views.

## Core workflow

`evidence -> entities -> chronology -> geography -> environment -> sky -> characters -> event graph -> simulation -> visualization`

Every generated or inferred element carries provenance, confidence and an explicit evidence class. A reconstruction is never silently presented as established historical fact.

## Major capabilities

- historical events, excavations and findings
- chronology and event graphs
- historical and environmental maps
- archaeological sites, terrain and 3D artifacts
- ancient-script glyph/OCR integration
- parametric character and population agents
- movement, interaction and event simulation
- seasonal and historical night-sky reconstruction
- solar/lunar/planetary/stellar visibility metadata
- weather, terrain, water and atmosphere scenario layers
- 128D state representation: geometry, time, perspective, energy/light, events, objects/materials, information and cognition
- neural/RNN feature extraction and sequence prediction boundaries
- cross-repository ONNX/RNN engine registry with provenance
- CPU fallback plus OpenGL compute, DirectX 12/SM6, CUDA and OpenCV acceleration hooks
- Unreal Engine 5 plugin bridge and Unity/Unity3D C# bridge
- uncertainty-aware alternate reconstructions
- WebGL/Three.js/Cesium-compatible web visualization
- C++, C, Python, Java, TypeScript/JavaScript, Kotlin, Swift, Dart and C# API/reference layers
- Windows/Linux/macOS/Android/iOS build and deployment scripts

## GPU and engine architecture

AVRS uses a backend-neutral compute contract so historical scene tensors, glyph features, event graphs, character transforms, terrain samples and sky data can run on CPU or an accelerator. OpenGL 4.6 compute shaders, DirectX 12/Shader Model 6, CUDA and optional OpenCV CUDA acceleration are included as source-level adapters. Unreal Engine 5 and Unity/Unity3D integrations remain engine-native plugins rather than copied engine source.

## Neural acceleration

The neural layer can select ONNX Runtime execution providers at runtime. Preferred providers include TensorRT, CUDA, DirectML, OpenVINO, CoreML, NNAPI, WebGPU, XNNPACK and CPU fallback. The registry describes capabilities; it does not claim every provider exists on every host.

The Library-derived architecture retains axis-wise 128D embeddings, event/entity GNN message passing, temporal tensor-RNN memory, semantic concept embeddings and observer/perspective weighting. fileciteturn153file1L85-L112

## Scientific and historical guardrail

The suite separates observed evidence, source-supported interpretation, modeled reconstruction and speculative visualization. This follows the research archive's evidence/interpretation/hypothesis methodology and the London Charter approach to historical visualization.

## External standards and engines

Adapters are designed around open interchange such as JSON/GeoJSON, glTF, USD, OpenStreetMap/OpenHistoricalMap-style historical geography, OGC 3D Tiles and astronomical ephemerides. Proprietary application internals are not copied.

## Build

See `docs/BUILD_AND_DEPLOY.md`, `docs/GPU_ENGINE_INTEGRATION.md` and `scripts/build/`.

Native GPU, Unreal, Unity, Android and iOS builds remain toolchain-dependent. The repository contains source/configuration and reproducible build entry points; it does not falsely package hardware- or signing-dependent binaries without the corresponding toolchain.

## License

Original source in this directory is GPL-3.0-or-later, matching the repository's existing licensing direction.
