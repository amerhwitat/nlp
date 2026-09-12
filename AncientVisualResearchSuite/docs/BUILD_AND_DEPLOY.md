# Build and Deploy

## Desktop/core

- C++20/CMake: `scripts/build/build-all.ps1`, `.bat`, `.sh`
- Python: reference engine and tests can run with Python 3.10+
- Web: Node.js + npm + TypeScript + Vite + Three.js

## Android

The `mobile/android` Gradle project targets a release APK/AAB. Run the mobile build script or Gradle directly with an installed Android SDK/JDK.

## iOS

The `mobile/ios` directory contains the Swift reference layer. An IPA is produced only on macOS with Xcode, an Apple signing identity and the appropriate provisioning configuration. The repository deliberately does not contain signing credentials or claim a signed IPA exists until that build is actually executed.

## Flutter

`mobile/flutter` contains the shared Dart scene API. A complete Flutter application can consume the same event schema and produce Android/iOS packages.

## Deployment modes

1. static WebGL/Three.js site;
2. optional CesiumJS geospatial frontend;
3. native desktop viewer;
4. Android APK/AAB;
5. iOS app/archive/IPA;
6. research workstation with Python/C++ engines.

## Reproducibility

Build artifacts are generated locally/CI and are not committed by default. Data sources, model versions, astronomy engines, terrain datasets and reconstruction parameters must be recorded in the evidence manifest.
