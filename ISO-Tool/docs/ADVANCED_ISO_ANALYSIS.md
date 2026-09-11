# Advanced ISO Analysis

ISO-Tool now has a bounded offline analyzer in `python/iso_tool/advanced_inspect.py`.

## Detection model

| Structure | Detection |
|---|---|
| ISO 9660 | Volume descriptors beginning at sector 16 with `CD001` |
| Joliet | Supplementary Volume Descriptor escape sequence |
| Rock Ridge | Root-directory System Use area hint |
| UDF | `BEA01`, `NSR02`, or `NSR03` markers |
| MBR | `0x55AA` system-area signature |
| GPT | `EFI PART` header marker |
| El Torito | Boot record descriptor and validation/catalog entries |

The analyzer does not mount the image and does not execute boot code.

## El Torito / UEFI

UEFI defines EFI System Partition booting through El Torito no-emulation entries and uses platform ID `0xEF`. ISO-Tool therefore reports platform IDs rather than assuming that every boot image is a BIOS image. citeturn0search36

Microsoft's Oscdimg documentation similarly models BIOS and UEFI multi-boot entries separately and supports ISO 9660, Joliet and UDF. citeturn0search0

## Large images

The profile system can request deterministic boot-file ordering for large images. This is represented as build intent so the selected backend can implement its native ordering mechanism. Oscdimg documents explicit boot-order files for images larger than 4.5 GB. citeturn0search0

## Hardening

Parsing is bounded by a finite descriptor scan and fixed sector reads. Truncated catalogs and malformed validation entries produce warnings instead of attempts to execute or repair the source image. This follows the general direction of recent libarchive ISO parser hardening, including fixes for path normalization, Joliet overflows, null dereferences and malformed continuation areas. citeturn0search5
