# ISO and optical-image format matrix

## Filesystem/index layers

ISO-Tool models ISO 9660 Levels 1–3, Joliet, Rock Ridge, UDF and combinations such as ISO9660+Joliet+Rock Ridge and ISO9660/UDF bridge images.

## Boot layers

El Torito boot catalogs can describe BIOS-style boot images and EFI boot entries. UEFI optical boot requires an appropriate ISO/UDF layout and EFI boot image. GPT/MBR are modeled separately for hybrid disk/USB-style images.

## Backends

The backend abstraction can use xorriso/xorrisofs, Windows Oscdimg, or another explicitly supported local image utility. Backend capability is detected before execution; unsupported combinations are rejected instead of silently producing a misleading image.

## Other media

CD/DVD/BD are media classes, not interchangeable filesystem formats. DVD/BD data may use UDF; video-disc authoring and audio-CD authoring are separate workflows and are not represented as ordinary ISO creation. The architecture leaves extension points for specialized media authoring backends.
