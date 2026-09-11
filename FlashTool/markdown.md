# FlashTool Architecture

## Execution model

```text
GUI -> Flash Orchestrator -> Device Manager -> Transport Backend
                         -> Partition/Image Manager
                         -> Security/AVB Validator
                         -> Verification Engine
                         -> Chimera Runtime adapters
```

## Device lifecycle

1. Discover ADB/Fastboot/USB interfaces.
2. Collect non-destructive identity and capability information.
3. Inspect partition topology and A/B slots.
4. Parse and validate the requested image/package.
5. Produce a human-readable dry-run plan.
6. Require explicit confirmation for writes or OEM-authorized unlocking.
7. Execute the minimum required operations.
8. Verify written data and boot state where supported.
9. Record a machine-readable operation log.

## Compatibility

Device-specific flashing protocols remain backend-specific. The core API must never assume that a partition exists, that an unlock is permitted, or that a vendor image is compatible merely from a filename.

## Chimera II

RegisterN/C8192/R8192 interfaces are optional acceleration adapters for hashing, image processing and parallel validation. Physical Android devices are not assumed to contain Chimera hardware.

## Security boundary

FlashTool is an authorized servicing tool. No exploit, FRP bypass, credential extraction, secure-boot bypass or unauthorized bootloader-circumvention mechanism is part of the architecture.
