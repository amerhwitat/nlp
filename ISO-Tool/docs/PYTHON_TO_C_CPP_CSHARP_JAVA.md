# Python → C/C++/C#/.NET/Java parity

ISO-Tool treats the Python implementation as the behavioral reference and now inventories every Python module recursively before compilation. The `source_translator.py` AST pass emits deterministic C, C++, C# and Java parity units plus a SHA-256 manifest.

## Pipeline

`toolchain detection → dependency inventory → Python AST inspection → parity generation → recursive source scan → build planning → compile/link → artifact staging → Spit Fire → ISO/IMG verification`

Run from `ISO-Tool/python`:

```text
python -m iso_tool.source_translator <python-root> --output <output>/generated/python-parity
```

The normal `build_entrypoint.py` invokes this automatically after dependency detection.

## Target baselines

- C: C11-compatible metadata units
- C++: C++17-compatible metadata units; native implementation may use C++20 where the selected toolchain supports it
- C#: APIs restricted to a common .NET 6 / .NET Framework 4.8-compatible surface where practical
- Java: Java 8+ baseline

## Semantic safety

Dynamic Python behavior is not silently guessed. Imports, classes, functions, source identity and hashes are recorded. Native implementations must replace parity units with behaviorally equivalent implementations and pass contract tests before being declared equivalent.

This distinction prevents generated code from being represented as a complete semantic conversion when a Python feature has no direct language equivalent.

## Generated output

```text
<output>/generated/python-parity/
  c/*.c
  cpp/*.cpp
  csharp/*.cs
  java/*.java
  parity-manifest.json
```

The manifest records every discovered `.py` file, imports, classes, functions, source hash and target language set.

## Required native parity areas

Dependency detection, toolchain discovery, recursive scanning, build adapters/planning, GitHub acquisition, artifact collection, boot assembly/import/validation, ISO/IMG mastering, output management, AI/document/image engines, resilience and GUI/web entry points must each have explicit native contracts.
