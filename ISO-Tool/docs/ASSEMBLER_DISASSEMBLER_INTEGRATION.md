# Assembler / disassembler integration

ISO-Tool now treats assembler and disassembler selection as a first-class build stage between dependency scan and compile/link.

Flow:

`dependency scan → backend discovery → target/syntax selection → assemble/compile → disassemble/inspect → artifact staging → ISO mastering`

Backends include NASM, YASM, GNU binutils, LLVM MC/LLVM disassembler, Visual Studio MASM detection, and Chimera II C8192/R8192 adapters. The backend registry records target architecture, syntax, commands and acquisition policy.

LLVM MC supports assembly and disassembly and explicit architecture/triple selection; LLVM's MC disassembler API provides target-specific instruction decoding. NASM is the x86 assembler backend. Sources are referenced rather than blindly vendored when license/vendor policy requires installed tools.

Windows native code uses explicit Win32 headers and pragma-linked system libraries. The required `comctl32.lib` linkage is included in the shared native linkage contract.
