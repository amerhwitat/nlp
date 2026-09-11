# Related repository integration

ISO-Tool is the build and image-production boundary for the current Chimera II workspace. The standard related-repository set is:

| Repository | Role | Default source |
|---|---|---|
| `amerhwitat/ChimeraIIOS` | OS, kernel, boot, services and system components | `https://github.com/amerhwitat/ChimeraIIOS` |
| `amerhwitat/BizX` | application/commerce/wallet platform | `https://github.com/amerhwitat/BizX` |
| `amerhwitat/BizXtreme` | game, crypto, WebGL/Three.js and application integration | `https://github.com/amerhwitat/BizXtreme` |

## Multi-repository build

`python/build_workspace.py` is the single command-line entry point for the combined workspace. It:

1. acquires each repository through the existing safe source-acquisition layer;
2. recursively scans every repository and discovers nested build systems;
3. builds compatible projects with the selected GNU/MSVC policy;
4. keeps independent repository failures isolated and visible;
5. preserves the complete source trees under `staging/src/<repository>`;
6. collects executables into `staging/bin`, libraries into `staging/lib`, and boot/image artifacts into `staging/boot-images`;
7. builds and stages the Spit Fire BIOS first stage;
8. writes a machine-readable workspace manifest; and
9. creates `Chimera-II-Workspace.iso` and a corresponding `.img` copy when an ISO backend is available.

Example:

```text
cd ISO-Tool/python
python build_workspace.py --output <selected-output> --compiler auto
```

The profile is controlled by `ISO-Tool/engine/repository-profiles.json`. Individual sources can be overridden with `--repos ID=URL`.

## Artifact policy

Compiled outputs are copied rather than executed. Existing APK, WebGL and other packaged artifacts remain artifacts. A file is only reported as successfully built when its registered build adapter completes successfully. Missing toolchains are recorded as unavailable.

## Provenance

The output contains source copies, repository identifiers, recursive tree indexes, build reports, application discovery data, AI/refinement data, and the combined workspace manifest. Deterministic build-system evidence remains authoritative over advisory AI planning.

## Security boundary

Remote source acquisition and package installation remain explicit operations. Imported boot sectors are treated as opaque data and are never executed by ISO-Tool. Arbitrary downloaded installers are not executed automatically. Archive extraction rejects unsafe traversal/link patterns.
