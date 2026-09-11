# ISO-Tool Python implementation

The Python implementation is the reference orchestration layer for recursive source analysis, dependency detection, compiler/build adapters, Spit Fire boot construction, artifact staging and ISO mastering.

## Entry points

Single repository:

```text
python -m iso_tool.build_entrypoint <source> --output <directory> --compiler auto
```

Combined Chimera II workspace:

```text
python build_workspace.py --output <directory> --compiler auto
```

The default multi-repository profile builds `ChimeraIIOS`, `BizX` and `BizXtreme` independently and combines their source trees and compatible artifacts into one staging tree.

## Output contract

```text
<output>/
├── sources/
├── repositories/<id>/
├── build/
├── staging/
│   ├── src/<id>/
│   ├── bin/
│   ├── lib/
│   ├── boot-images/
│   └── metadata/workspace-manifest.json
├── iso/
├── img/
├── boot-images/
├── executables/
├── libraries/
├── knowledge/
└── manifests/
```

## Safety and reproducibility

Build commands are selected from registered build adapters. Missing tools are reported as unavailable, build failures remain visible, and independent jobs continue. Imported images are inspected as data rather than executed. Repository provenance and machine-readable build manifests are retained.

Run the tests with:

```text
python -m unittest discover ISO-Tool\\python\\tests -v
```
