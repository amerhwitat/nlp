# Chimera II universal ISA integration

ISO-Tool treats the Chimera II ISA registry, canonical micro-op contract, universal CPU toolchain catalog and virtual memory-bus profiles as build inputs rather than opaque source files.

## Inputs

- `../..` Chimera II repository profile: `isa/registry.py`
- Canonical target metadata: `include/chimera/universal_isa.hpp`
- Semantic lowering contract: `include/chimera/universal_microop.hpp`
- CPU toolchain capability catalog: `toolchains/registry.json`
- Virtual memory-bus profiles: `memory/bus-profiles.json`
- Memory-bus API/probe: `include/chimera/memory_bus.hpp`, `include/chimera/memory_bus_probe.hpp`

## Build behavior

1. Acquire and hash the configured Chimera II source tree.
2. Validate ISA and toolchain registries before compilation.
3. Validate memory-bus profile metadata before staging.
4. Record target names, execution mode, encoding model, privilege model and feature capabilities.
5. Preserve the canonical 16-byte instruction/emulator ABI.
6. Stage ISA, toolchain and memory-bus metadata with the generated build manifest.
7. Compile architecture-specific implementations only when their registered toolchain/backend is available.
8. Never mark a foreign instruction as natively implemented solely because its architecture appears in the compatibility registry.
9. Never package proprietary compiler or assembler binaries unless they are supplied with compatible redistribution rights.

## Universal lowering

Foreign instruction streams are expected to follow:

`decode -> verified descriptor -> canonical MicroOp -> RegisterN/memory/control state -> execution service`

This permits x86-64, AArch64, RISC-V, POWER, MIPS, SPARC, IBM Z/s390x and legacy targets to share semantic execution infrastructure while retaining architecture-specific decoding and privilege behavior.

## Reproducibility

The workspace manifest records source SHA-256 values, selected compiler/toolchain, ISA registry state, toolchain capability state, memory-bus profile state, build artifacts and boot-image inputs. Generated ISO contents therefore remain traceable to the source and architecture metadata used for the build.
