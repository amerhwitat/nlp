# ISO-Tool Configuration Reference

## Media

- **CD** — optical media profile intended for CD-sized images.
- **DVD** — optical media profile intended for DVD-sized images.
- The media choice affects the default output name and capacity validation; it does not magically compress an oversized source tree.

## Filesystems

- **ISO9660** — maximum compatibility baseline.
- **ISO9660 + Joliet + Rock Ridge** — compatibility plus long/Unicode names and Unix metadata where supported by the backend.
- **UDF** — optical-media filesystem option suited to larger/modern media workflows.

Microsoft Oscdimg documents ISO 9660, Joliet and UDF as supported image filesystem choices. https://learn.microsoft.com/windows-hardware/manufacture/desktop/oscdimg-command-line-options

## Boot

- None
- BIOS
- UEFI
- BIOS + UEFI

BIOS boot uses an El Torito boot entry and the configured BIOS boot artifact. UEFI uses an EFI System Partition/EFI boot image as supported by the selected backend.

## Artifact policy

The build pipeline can collect:

- `.exe`
- `.dll`
- `.lib`
- `.a`
- `.so`
- `.bin`
- `.img`
- `.efi`

Executables and boot/binary images are staged under `/bin`; libraries are staged under `/lib`.

## Source policy

The selected repository is copied into `/src` with its directory hierarchy preserved, excluding `.git` metadata. This allows the ISO to contain the exact source used to produce the generated artifacts.

## Applications

`/applications/linux` and `/applications/windows` are reserved for locally authorized free/open-source applications and package metadata. The tool does not imply redistribution rights merely because an application can be discovered online.

## Dependency policy

At startup the native front end checks for build and image tools. The user can choose:

- Scan only
- Install missing dependencies

Automatic installation is limited to trusted package-manager mechanisms. Internet discovery should produce a reviewable dependency record rather than executing arbitrary remote scripts.

## Backend selection

Preferred order on Windows:

1. xorriso/xorrisofs when installed and compatible with the selected profile.
2. Microsoft Oscdimg when installed through the Windows ADK.
3. Report an explicit failure if no backend can produce the requested image.

## Reproducibility

The staging directory contains `metadata/iso-tool-manifest.txt`. Future revisions can extend this manifest with source SHA-256, tool versions, backend command lines, file counts, boot-artifact hashes and ISO SHA-256.
