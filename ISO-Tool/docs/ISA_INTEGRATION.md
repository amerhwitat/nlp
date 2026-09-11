# Chimera II universal ISA integration

ISO-Tool treats the Chimera II ISA registry and canonical micro-op contract as build inputs rather than opaque source files.

## Inputs

- `../..` Chimera II repository profile: `isa/registry.py`
- Canonical target metadata: `include/chimera/universal_isa.hpp`
- Semantic lowering contract: `include/chimera/universal_microop.hpp`

## Build behavior

1. Acquire and hash the configured Chimera II source tree.
2. Validate the ISA registry before compilation.
3. Record target names, execution mode, encoding model, privilege model and feature capabilities.
4. Preserve the canonical 16-byte instruction/emulator ABI.
5. Stage ISA metadata and the micro-op contract with the generated build manifest.
6. Compile architecture-specific implementations only when their registered toolchain/backend is available.
7. Never mark a foreign instruction as natively implemented solely because its architecture appears in the compatibility registry.

## Universal lowering

Foreign instruction streams are expected to follow:

`decode -> verified descriptor -> canonical MicroOp -> RegisterN/memory/control state -> execution service`

This permits x86-64, AArch64, RISC-V, POWER, MIPS, SPARC, IBM Z/s390x and legacy targets to share semantic execution infrastructure while retaining architecture-specific decoding and privilege behavior.

## Reproducibility

The workspace manifest records source SHA-256 values, selected compiler/toolchain, ISA registry state, build artifacts and boot-image inputs. Generated ISO contents therefore remain traceable to the source and architecture metadata used for the build.
