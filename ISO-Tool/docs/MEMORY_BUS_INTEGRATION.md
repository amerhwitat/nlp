# ISO-Tool memory-bus integration

ISO-Tool stages Chimera II's architecture-neutral virtual memory-bus contract and the architecture profiles used by Koronos inspection.

## Staged inputs
- `memory/bus-profiles.json`
- `include/chimera/memory_bus.hpp`
- `include/chimera/memory_bus_probe.hpp`
- `src/memory/memory_bus.cpp`
- `src/memory/memory_bus_probe.cpp`

## Validation
The build validates profile identity, address/data-width metadata and deterministic JSON before creating the ISO manifest. The runtime probe consumes firmware, device-tree, ACPI, hypervisor or virtual-platform descriptors; it does not read arbitrary physical memory during host-side ISO construction.

## ISO metadata
The generated image retains architecture and toolchain provenance together with memory-bus capability metadata. The selected ISA backend, assembler/compiler capability and virtual memory-bus model are recorded without silently embedding proprietary host tools.

## Compatibility
A physical memory bus may be narrower than a Chimera RegisterN value. Wide RegisterN operations therefore remain semantic CPU state and can be decomposed into multiple bus transactions or vector lanes.
