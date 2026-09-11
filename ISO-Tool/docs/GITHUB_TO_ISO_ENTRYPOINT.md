# GitHub → Compile/Link → ISO entry point

ISO-Tool now starts from an explicit source selection. The GUI accepts either a GitHub URL (`https://github.com/owner/repository` or `owner/repository`) or a local repository path.

## Flow

1. Select the GitHub repository.
2. ISO-Tool clones a shallow checkout into the user-selected output `sources/` directory.
3. The single Python build entry point configures CMake, compiles and links with GNU C++ and Microsoft Visual C++ independently when available.
4. Executables, DLL/shared libraries/static libraries, boot images and build manifests are staged under the final output directory.
5. `Compile + Link + Build ISO` creates an `iso-staging/` tree from those artifacts and invokes the existing ISO backend (`xorriso`, `xorrisofs`, or `oscdimg`).

The GNU and MSVC builds use separate build trees so their object files and CRT/linker outputs are never mixed.

## Command-line entry point

```text
python -m iso_tool.build_entrypoint <source> --output <directory> --compiler gnu
python -m iso_tool.build_entrypoint <source> --output <directory> --compiler msvc
```

The GUI's **Select GitHub repo…** button is the normal interactive entry point.

## Chimera II OS

The Chimera II OS repository also contains `tools/build/main.py`, a repository-local build entry point that runs CMake configure/build and writes `build/chimera-build-artifacts.json`.

GitHub repository contents can be read/updated through the GitHub Contents API; updates to existing files require the current blob SHA, so ISO-Tool keeps repository mutations serialized. See the GitHub REST API documentation for the repository contents contract.
