# Flashing Safety

FlashTool defaults to inspection and dry-run behavior. A write is allowed only after device identification, partition/image compatibility validation and explicit user confirmation.

## Never bypass

- OEM bootloader authorization
- Android Verified Boot security
- Factory Reset Protection
- device credentials or authentication
- secure boot or vendor cryptographic controls

## Recommended workflow

1. Back up user data.
2. Confirm the exact device model/build and region.
3. Capture partition metadata.
4. Validate image hashes and format.
5. Review the dry-run plan.
6. Flash only supported partitions.
7. Verify and reboot.
8. Keep operation logs for recovery/debugging.

Vendor-specific emergency download modes and proprietary protocols must be implemented only from publicly documented, authorized interfaces.
