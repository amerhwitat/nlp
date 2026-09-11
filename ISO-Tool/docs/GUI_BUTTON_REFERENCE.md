# ISO-Tool unified GUI button reference

The canonical feature vocabulary is `ISO-Tool/gui/feature_manifest.json`. Every native GUI exposes the same **66 feature buttons** in these groups:

1. **Source & Repository** — GitHub/local/archive acquisition, deep recursive scan, tree, dependency and application discovery.
2. **Toolchains** — Windows detection, MSVC, Clang, GCC/G++, NASM and explicit bootstrap/fallback operations.
3. **Build** — planning, compile/link, Debug/Release, GNU/MSVC, all-project builds, tests and journal.
4. **ISO / Boot** — Spit Fire, BIOS, UEFI, hybrid profiles, ISO/IMG, artifact merging, manifests and verification.
5. **Packages / Applications** — package-manager discovery, authorized dependency installation, application and artifact inventories.
6. **Diagnostics** — configuration, dependency/runtime/build errors, logs, export and final verification.
7. **Knowledge & AI** — web search, bounded crawling, documentation indexing, knowledge-base generation, RNN/Transformer/LLM adapters, AI build analysis, provenance, model metrics and offline mode.

| Language | GUI | Entry point |
|---|---|---|
| Python | Tkinter | `python/launch_gui.py` |
| Java | Swing | `java/src/main/java/iso/tool/Main.java` |
| C# | WPF | `dotnet/ISO-Tool/MainWindow.xaml` |
| C++ | Win32 | `vcpp/ISO-Tool-UnifiedGui.vcxproj` |

All four use the same labels/order while retaining native UI technology.

## AI safety contract
Web and LLM output is evidence/recommendation. It cannot silently execute downloaded scripts, install packages, invoke compilers, or create an ISO. Those actions remain inside the existing authorized build pipeline.

## Provenance
Knowledge records retain source URL/path, retrieval time, SHA-256 and source type. This makes recommendations auditable and allows offline rebuilds of the knowledge index.
