# Android Platform-Tools Integration

FlashTool treats `adb` and `fastboot` as external, versioned host transports. It does not bundle Google binaries.

## Supported roles

- ADB: device discovery, non-destructive diagnostics and authorized device-side inspection.
- Fastboot: bootloader transport and authorized flashing where the device permits it.
- Fastbootd: userspace fastboot for dynamic-partition workflows.

The transport layer should report the detected tool version, executable path and capability set in the operation journal. Feature detection is preferred over version-string assumptions.

## Current ecosystem note

Platform-Tools release notes document changes in ADB USB and mDNS backends, so transport discovery should remain modular and testable.
