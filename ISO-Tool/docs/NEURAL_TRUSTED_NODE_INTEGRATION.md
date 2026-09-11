# Chimera II Neural / Trusted-Node ISO Integration

The ISO Tool must stage the Chimera cognitive registry, neural database schema and trusted-node protocol alongside the normal ISA and memory-bus metadata.

## Required staged files

- `neural/registry.json`
- `database/manifests/neural-network-database.yaml`
- `include/chimera/neural_node.hpp`
- `src/neural/trusted_node.cpp`
- `docs/NEURAL_TRUSTED_NODE_FABRIC.md`

## Integrity policy

Each file is hashed into the deterministic artifact manifest. The ISO builder records source commit, toolchain, SHA-256 digest and build timestamp policy. Neural model checkpoints are not activated merely because they appear on media; activation requires signature/provenance verification.

## Node discovery policy

The ISO contains configuration for published endpoint discovery only. Supported discovery sources are signed bootstrap directories, DNS SRV/TXT, local mDNS/Bonjour and explicitly configured peers. The ISO tool must not add arbitrary network scanners or autonomous port-probing behavior.

## Compatibility research

The metadata layer can reference Windows ML/ONNX execution providers, Linux AF_XDP/eBPF/io_uring patterns and macOS virtualization/security patterns as external compatibility targets. It must preserve license and provenance metadata and must not redistribute proprietary operating-system binaries or source.
