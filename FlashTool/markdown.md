# FlashTool Architecture

## Execution model

```text
GUI / Web UI / CLI
        |
        v
Analysis + Planning API
        |
        v
Canonical C++ Safety Core
   |             |
   v             v
Image/OTA      Device/Transport
Analyzers      ADB/Fastboot/Fastbootd/USB
   |             |
   +------v------+
          |
   Compatibility Preflight
          |
      Dry-Run Plan
          |
   Explicit Confirmation
          |
      Authorized Execute
          |
     Verify + Journal
```

## Device lifecycle

1. Discover ADB/Fastboot/USB interfaces.
2. Collect non-destructive identity and capability information.
3. Inspect physical and logical partition topology and A/B slots.
4. Detect boot/init_boot/vendor_boot, sparse, VBMeta, `super` and CrAU artifacts.
5. Parse and validate the requested image/package.
6. Compare image size, partition capacity, slot and device capability.
7. Produce a human-readable and JSON dry-run plan.
8. Require explicit confirmation for writes.
9. Execute only through an authorized transport backend.
10. Verify written data, hashes and boot state where supported.
11. Record a machine-readable operation journal.

## New analysis layers

- **AVB:** VBMeta/footer detection, future descriptor/rollback-index parsing.
- **OTA:** OTA ZIP and CrAU header/metadata discovery.
- **Dynamic partitions:** logical partition and `super` metadata model.
- **Sparse images:** bounded sparse-header inspection before capacity checks.
- **Compatibility:** device/slot/partition/image preflight.
- **Provenance:** hashes, tool versions and deterministic analysis records.

## Compatibility

Device-specific flashing protocols remain backend-specific. The core API must never assume that a partition exists, that an unlock is permitted, or that a vendor image is compatible merely from a filename.

## Chimera II

RegisterN/C8192/R8192 interfaces are optional acceleration adapters for hashing, image processing and parallel validation. Physical Android devices are not assumed to contain Chimera hardware.

## Security boundary

FlashTool is an authorized servicing tool. No exploit, FRP bypass, credential extraction, secure-boot bypass, rollback bypass or unauthorized bootloader-circumvention mechanism is part of the architecture.
