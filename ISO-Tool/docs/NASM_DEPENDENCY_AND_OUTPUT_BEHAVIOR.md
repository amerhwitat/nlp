# NASM Dependency and Output Behavior

## NASM dependency lifecycle

ISO-Tool treats NASM as a build dependency when assembly sources or boot-image assembly require it.

1. Check whether NASM is already available on `PATH`.
2. If it is missing, download the trusted NASM distribution into the user's profile Downloads cache:
   `%USERPROFILE%\\Downloads\\Chimera-II-ISO-Tool\\dependencies\\nasm\\`.
3. Show the exact downloaded dependency folder in the UI/log immediately after download.
4. Verify the downloaded package before installation (archive integrity and expected NASM executable).
5. Install NASM automatically when the user has authorized automatic dependency installation.
6. Re-detect NASM and report the installed version/path.
7. Never execute arbitrary downloaded scripts; dependency installation is restricted to the declared dependency package and installer.

The dependency cache is retained so subsequent builds can reuse the downloaded NASM package rather than downloading it again.

## Generated output lifecycle

Before ISO mastering, ISO-Tool asks the user to select the final output directory. The selected directory is the root for the generated deliverables.

The application creates and reports:

- `iso/` — final ISO image(s)
- `boot-images/` — Spit Fire BIOS/UEFI boot images and other generated boot binaries
- `binaries/executables/` — generated EXE and other executable images
- `binaries/libraries/` — DLL, LIB, SO and other generated libraries
- `logs/` — build, dependency and mastering logs
- `manifests/` — artifact/dependency manifests

After ISO generation succeeds, ISO-Tool opens/shows the selected destination folder and highlights the generated ISO file. The UI and live log report both the complete destination directory and the final ISO path.

The same behavior applies to the Python/Tkinter, C# WPF and native Visual C++ front ends: dependency downloads use the profile Downloads cache, while final generated media uses the user-selected output directory.
