# Chimera Amiga Web

Browser-based Amiga emulation integration for the `nlp` repository.

## What is included

The application provides a modern web shell around an Amiga emulator core:

- Amiga model selector: A500, A500+, A600, A1200, A2000, A3000 and A4000/030.
- PAL/NTSC selector.
- Start, pause, reset and fullscreen controls.
- Kickstart ROM, DF0/DF1 floppy and HDF/VHD file pickers.
- Chip/Fast RAM controls.
- Keyboard/mouse and Gamepad API integration hooks.
- Live diagnostics and emulator status.
- Adapter API for a locally vendored SAE or vAmigaWeb browser engine.

The UI is intentionally separated from the emulator core so that upstream
emulation code can be updated without replacing the Chimera interface.

## Upstream research and code sources

Two browser-oriented open-source projects were selected after researching the
Library and public web sources:

1. **Scripted Amiga Emulator (SAE)** — JavaScript/HTML5 emulator based heavily
   on UAE/WinUAE. Its project documents A1000/A500/A500+/A600/A1200/A3000/A4000
   models, 68000/68010/68020/68030 CPUs, OCS/ECS/AGA, PAL/NTSC, WebGL/Canvas2D,
   WebAudio, keyboard/mouse/gamepad and disk-image support. It is GPL-licensed.
   Source: https://github.com/naTmeg/ScriptedAmigaEmulator

2. **vAmigaWeb** — C++ vAmiga core exposed to the browser through WebAssembly,
   with JavaScript/HTML UI integration. It is GPL-3.0 licensed.
   Source: https://github.com/vAmigaWeb/vAmigaWeb

The Library also contains prior Chimera II Web OS research describing an
Amiga/Workbench browser profile and recommending browser sandboxing and explicit
user-consent file access.

## Vendoring the emulator core

The repository intentionally does not silently copy Kickstart ROMs or
copyrighted commercial Amiga software. The emulator core is open source, while
ROMs and disk images can have separate licensing requirements.

Run:

```bash
./fetch_upstream.sh
```

or on Windows PowerShell:

```powershell
./fetch_upstream.ps1
```

The scripts clone the upstream open-source source trees into `vendor/` for a
local build. Review the upstream license files before redistributing modified
or combined binaries.

### SAE vendor layout

```text
Amiga/
├── index.html
├── app.js
├── style.css
├── emulator_adapter.js
├── README.md
├── LICENSES.md
├── sources.json
├── fetch_upstream.sh
├── fetch_upstream.ps1
└── vendor/
    ├── sae/
    └── vamigaweb/
```

## Browser security

The browser sandbox prevents the emulator from directly modifying the host
filesystem. User-selected ROM/disk files are passed through browser APIs only.
Do not place copyrighted Kickstart ROMs, commercial games, or other restricted
images in this repository unless you have distribution rights.

## Running

Serve the repository with a local HTTP server rather than opening `index.html`
with `file://`:

```bash
python -m http.server 8080 --directory Amiga
```

Then open `http://localhost:8080/`.

## Library research basis

The existing Chimera II Web OS research describes a browser-based desktop with
an Amiga profile and emphasizes browser sandboxing. The AmigaGuruBook in the
Library also contains technical material concerning Kickstart, Amiga memory,
and low-level hardware/software interfaces.
