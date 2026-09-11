# ISO-Tool Configuration Reference

## User-selected output directory

The final ISO is **never silently forced into the repository**. Before an ISO or boot-image build starts, the native GUI asks the user where generated media should be saved and provides a Browse control. The Python reference API accepts the same explicit output directory.

The default suggestion is:

```text
%USERPROFILE%\\Downloads\\Chimera-II-ISO-Tool
```

The user may select any writable directory. The selected directory receives deterministic subdirectories:

- `iso/` — final `.iso` files.
- `boot-images/` — generated and exported boot `.img`, `.bin`, and `.efi` images.
- `binaries/executables/` — generated `.exe` and executable binary images.
- `binaries/libraries/` — generated `.dll`, `.lib`, `.a`, and `.so` files.
- `logs/` — build, dependency, and mastering logs.
- `manifests/` — artifact and reproducibility manifests.

## Dependency download/cache directory

Dependency discovery may search trusted package-manager sources, but downloaded installers/packages are cached under the current user's profile Downloads directory:

```text
%USERPROFILE%\\Downloads\\Chimera-II-ISO-Tool\\dependencies
```

This cache is separate from the final ISO output selection. It prevents build dependencies from being mixed into the repository or silently written to an arbitrary project directory.

The native Windows front end must show the dependency list and destination before an installation action. Installation remains an explicit user-authorized action; arbitrary remote scripts are never executed.

## Media

- **CD** — optical media profile intended for CD-sized images.
- **DVD** — optical media profile intended for DVD-sized images.
- The media choice affects the default output name and capacity validation; it does not magically compress an oversized source tree.

## Filesystems

- **ISO9660** — maximum compatibility baseline.
- **ISO9660 + Joliet + Rock Ridge** — compatibility plus long/Unicode names and Unix metadata where supported by the backend.
- **UDF** — optical-media filesystem option suited to larger/modern media workflows.

Microsoft Oscdimg documents ISO 9660, Joliet and UDF as supported image filesystem choices.

## Boot

- None
- BIOS
- UEFI
- BIOS + UEFI

BIOS boot uses an El Torito boot entry and the configured BIOS boot artifact. UEFI uses an EFI System Partition/EFI boot image as supported by the selected backend.

## Boot evidence and generated binary images

When boot-image generation is requested, ISO-Tool keeps the generated boot artifacts outside the source tree and also stages the selected boot artifacts into the ISO. It records hashes and source provenance in the build manifest. A fallback Spit Fire container is clearly marked as non-bootable when no assembled first-stage binary exists.

If a boot emulator is available, validation can produce boot-test evidence/logs alongside the selected output. ISO-Tool does not claim that an image booted merely because an image file was generated.

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

Automatic installation is limited to trusted package-manager mechanisms. Internet discovery should produce a reviewable dependency record before download/install. Dependency installers are stored in the Downloads cache described above.

## Backend selection

Preferred order on Windows:

1. xorriso/xorrisofs when installed and compatible with the selected profile.
2. Microsoft Oscdimg when installed through the Windows ADK.
3. Report an explicit failure if no backend can produce the requested image.

## Reproducibility

The staging directory contains `metadata/iso-tool-manifest.txt`. Future revisions can extend this manifest with source SHA-256, tool versions, backend command lines, file counts, boot-artifact hashes and ISO SHA-256.
