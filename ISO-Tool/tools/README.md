# ISO-Tool assembler/disassembler toolchain

ISO-Tool provides one toolchain registry for assembly, disassembly, object inspection and binary transformation. Backends are discovered locally first and may be cached under the user's Downloads dependency cache.

## Backends

| Backend | Assembly | Disassembly | License/distribution policy |
|---|---:|---:|---|
| NASM | x86/x86-64 | via companion inspection tools | BSD-2-Clause; external package/cache supported |
| YASM | x86/x86-64 | companion inspection tools | BSD-derived; external package/cache supported |
| GNU binutils (`as`, `objdump`) | yes | yes | GPL; use installed/package-managed copy or compliant source distribution |
| LLVM MC (`llvm-mc`) | yes | yes | Apache-2.0 WITH LLVM-exception |
| LLVM disassembler libraries | no | yes | Apache-2.0 WITH LLVM-exception |
| Microsoft MASM (`ml.exe`/`ml64.exe`) | yes | no | Microsoft tool; detect installed Visual Studio/Build Tools, do not redistribute |
| GNU/LLVM cross targets | yes | yes | Target availability depends on installed toolchain |
| Chimera II C8192/R8192 | yes | yes | ISO-Tool/Chimera II native definitions |

"All assemblers" means all supported/discoverable assemblers rather than copying software whose license or vendor terms prohibit redistribution. The registry records vendor ownership and acquisition policy.

## Registry

`registry.json` describes executable names, capabilities, target architectures, syntax variants and acquisition policy. `toolchain.py` performs deterministic discovery and returns a selected backend.

The default target set includes x86/x86-64, ARM/AArch64, RISC-V, MIPS, PowerPC, SPARC, AVR, WebAssembly/LLVM targets, and Chimera II CISC/RISC targets where the selected backend supports them.

## C++ linkage

The native Windows front end explicitly uses Windows headers and pragma-linked system libraries. In particular the entry point includes `<windows.h>` and `<shlobj.h>` and links `comctl32.lib`. Other Windows libraries are also declared explicitly when referenced.

## Security

ISO-Tool never executes downloaded source or installer scripts. Package-manager installation remains an explicit user-authorized operation. Binary tools are validated by executable presence/version checks before use.
