# Toolchain acquisition policy

ISO-Tool distinguishes **embedded source**, **redistributable cached tools**, and **vendor-installed tools**.

- Open-source tool source may be vendored only when its license and repository policy permit it.
- NASM may be downloaded to the profile Downloads dependency cache; its upstream project is BSD-2-Clause licensed.
- LLVM/binutils/YASM can be consumed from compliant installations or package-manager caches.
- MASM is detected from Visual Studio/Build Tools and is not redistributed by ISO-Tool.
- Proprietary vendor SDKs are detected but not copied into the repository.
- Every selected backend is recorded in the build manifest with executable path and version.

The registry is deliberately broader than a hard-coded list of one assembler: ISO-Tool can discover target-specific GNU/LLVM cross assemblers and disassemblers when installed.
