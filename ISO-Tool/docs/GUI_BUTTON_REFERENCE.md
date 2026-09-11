# ISO-Tool unified GUI button reference

The canonical feature vocabulary is `ISO-Tool/gui/feature_manifest.json`. Every native GUI exposes the same 48 feature buttons in these groups:

1. **Source & Repository** — GitHub/local/archive acquisition, deep recursive scan, tree, dependency and application discovery.
2. **Toolchains** — Windows detection, MSVC, Clang, GCC/G++, NASM and explicit bootstrap/fallback operations.
3. **Build** — planning, compile/link, Debug/Release, GNU/MSVC, all-project builds, tests and journal.
4. **ISO / Boot** — Spit Fire, BIOS, UEFI, hybrid profiles, ISO/IMG, artifact merging, manifests and verification.
5. **Packages / Applications** — package-manager discovery, authorized dependency installation, application and artifact inventories.
6. **Diagnostics** — configuration, dependency/runtime/build errors, logs, export and final verification.

## Implementations

| Language | GUI | Entry point |
|---|---|---|
| Python | Tkinter | `python/launch_gui.py` |
| Java | Swing | `java/src/main/java/iso/tool/Main.java` (no arguments) |
| C# | WPF | `dotnet/ISO-Tool/MainWindow.xaml` |
| C++ | Win32 | `vcpp/ISO-Tool-UnifiedGui.vcxproj` |

Native layouts are deliberately similar rather than sharing one GUI toolkit. This keeps Windows-native behavior in C++/.NET while retaining a dependency-light Python GUI and standard-library Java GUI.

## Error handling

Feature callbacks update status and logs instead of allowing ordinary input/runtime exceptions to crash the GUI. External tool absence is reported as a dependency condition. Downloaded source/scripts are not executed merely because they were discovered.

The existing full pipeline implementations remain the authoritative build engine; the unified GUIs are front ends to those capabilities and expose the same terminology.
