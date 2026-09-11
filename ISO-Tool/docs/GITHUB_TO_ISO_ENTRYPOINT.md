# GitHub → Compile/Link → ISO entry point

ISO-Tool starts from explicit source selection. The GUI accepts a GitHub URL (`https://github.com/owner/repository` or `owner/repository`), archive, or local repository path.

## Single-repository flow

1. Select the GitHub repository.
2. ISO-Tool acquires it into the user-selected `sources/` directory.
3. The build entry point recursively discovers projects and invokes registered build adapters.
4. Executables, DLL/shared libraries/static libraries, boot images and build manifests are staged under the final output directory.
5. The ISO pipeline creates the staging tree and invokes `xorriso`, `xorrisofs`, or `oscdimg` when available.

The GNU and MSVC builds use separate build trees so their object files and CRT/linker outputs are never mixed.

## Related-repository workspace flow

For the complete Chimera II workspace, use:

```text
cd ISO-Tool/python
python build_workspace.py --output <selected-output> --compiler auto
```

The standard profile builds:

```text
amerhwitat/ChimeraIIOS
amerhwitat/BizX
amerhwitat/BizXtreme
```

The profile lives at `ISO-Tool/engine/repository-profiles.json`. Use `--repos ID=URL` to override or add sources. Each repository is recursively analyzed and built independently; failures are preserved in the combined manifest while unrelated repositories continue.

## Combined staging contract

```text
staging/
├── src/<repository>/        # complete native source trees
├── bin/<artifact>            # executables and compatible binary tools
├── lib/<artifact>            # DLL/shared/static libraries
├── boot-images/<repository>/ # binary/image/EFI artifacts
├── boot/bios/first_stage.bin # Spit Fire BIOS stage
└── metadata/workspace-manifest.json
```

Existing APK/WebGL outputs remain packaged artifacts. ISO-Tool does not fabricate source implementations from binaries.

## Command-line single-repository entry point

```text
python -m iso_tool.build_entrypoint <source> --output <directory> --compiler gnu
python -m iso_tool.build_entrypoint <source> --output <directory> --compiler msvc
```

The GUI's **Select GitHub repo…** button is the normal interactive entry point.

## Chimera II OS

The Chimera II OS repository also contains its own repository-local build entry points. ISO-Tool treats those as source evidence and invokes only recognized build systems through its own adapters.

GitHub repository contents can be read/updated through the GitHub Contents API; updates to existing files require the current blob SHA, so repository mutations remain serialized.
