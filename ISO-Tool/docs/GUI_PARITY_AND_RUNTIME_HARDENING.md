# ISO-Tool GUI parity and runtime hardening

## Unified GUI contract

All native ISO-Tool front ends now use `gui/feature_manifest.json` as the canonical feature vocabulary and ordering. The groups are Source & Repository, Toolchains, Build, ISO / Boot, Packages / Applications, and Diagnostics.

Implementations:

- Python/Tkinter: `python/iso_tool/unified_gui.py`, launched by `python/launch_gui.py`.
- Java/Swing: `java/.../UnifiedGui.java`; `Main` launches the GUI by default when no source argument is supplied.
- C#/.NET/WPF: `dotnet/ISO-Tool/MainWindow.xaml` plus `GuiFeatureHandlers.cs`.
- C++/Win32: `vcpp/UnifiedGui.cpp`, built by `ISO-Tool-UnifiedGui.vcxproj`; the existing full pipeline GUI remains available in `ISO-Tool.cpp`.

The visual contract is intentionally native per language, but uses the same title, section order, button names, action vocabulary, source field, status area, and log area. This avoids toolkit-specific behavior while keeping user interaction consistent.

## Runtime-error handling

GUI callbacks must not terminate the process on ordinary user/input failures. Errors are surfaced in the status/log area. Long-running Python operations are dispatched to worker threads so the Tk event loop remains responsive. Java and C++ feature actions likewise report their operation state in the log.

The GUI is a front end to the existing scan/build/boot pipeline; a button that requires a missing external tool must report that dependency rather than silently downloading or executing untrusted content.

## Toolchain compatibility

NASM integration follows the current NASM release/tooling model, while GCC integration follows the separate source/build directory model and explicit assembler/linker selection documented by GCC. urlNASM documentationhttps://www.nasm.us/docs.html and urlGCC configure/build documentationhttps://gcc.gnu.org/install/configure.html.

## Verification

CI should compile/test each maintained implementation. A successful repository update does **not** imply that NASM/GCC were built on a user's Windows workstation; those operations remain environment-dependent and are performed by the toolchain bootstrap actions when explicitly invoked.
