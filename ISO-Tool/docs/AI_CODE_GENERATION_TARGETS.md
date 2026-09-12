# AI code-generation targets

ISO-Tool documents two downstream systems as buildable AI-generated targets.

## Thamudic & Ancient Script Translation Engine

Reference architecture: React/Next.js + Tailwind/Canvas, FastAPI, OpenCV, PyTorch/ViT or ResNet-18, FST transliteration, Semitic morphology, dictionary/Levenshtein lookup and manual RTL/LTR/boustrophedon glyph ordering.

The ISO-Tool contract treats this as an application workload. Image preprocessing, glyph classification, Unicode/ANA mapping, linguistic decoding and web UI assets are scanned as separate build/artifact groups.

## Chimera II runtime

Reference architecture: C++/Rust runtime, lock-free rings, fixed-block memory pools, RMS/EDF scheduler, shared-memory zero-copy IPC, wide-word 8192-bit register emulation and WebAssembly visualization.

The ISO-Tool contract treats kernel/runtime sources, native libraries, boot images and WebAssembly assets as distinct artifacts. Native code is never inferred from AI prose; generated code must be scanned, dependency-checked, compiled and verified before packaging.

## AI generation policy

AI-generated source is input to the same recursive source scanner and dependency detector as human-authored source. No generated source is trusted merely because an AI model produced it. Provenance, source hashes, compiler/toolchain results and verification status are retained in build manifests.
