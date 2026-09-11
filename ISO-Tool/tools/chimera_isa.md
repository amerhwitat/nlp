# Chimera II assembler/disassembler contract

The ISO-Tool registry reserves native backends for the Chimera II C8192 variable-width CISC ISA and R8192 fixed-width RISC ISA.

## C8192

- Variable instruction packets from the Chimera II ISA contract.
- Register width family up to 8192 bits.
- Encoding/decoding must preserve packet bytes exactly.
- Native disassembly must report opcode, operands, packet width and unknown/reserved encodings.

## R8192

- Fixed 64-bit instruction packets.
- Deterministic decode by opcode/field layout.
- Native disassembler reports pipeline/branch metadata where available.

Until a native Chimera assembler binary is present, the registry reports these targets as unavailable rather than pretending another ISA assembler is compatible.
