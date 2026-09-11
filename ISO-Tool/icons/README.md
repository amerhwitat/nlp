# ISO-Tool Application Icons

The ISO-Tool branding follows the visual language of the Chimera II OS artwork already present in the user's Library: circular Chimera/griffin emblems, infinity/boot-disc geometry, dark navy foundations, and gold, cyan and violet accents.

Library references used for the visual direction include **Chimera II Multidimensional Computing Architecture**, **Chimera II OS Aurora Showcase**, and **Chimera II: Black Hole Photon Geodesics**. The Library material describes the circular Chimera II seal, infinity-like motifs, and the Aurora dark/glowing UI language.

`ISO-Tool-logo.svg` is the deterministic vector master committed with this repository. It is deliberately text-free at icon scale and can be rasterized into platform-specific sizes without depending on an external image-generation service.

## Platform use

- Web/documentation: use `ISO-Tool-logo.svg` directly.
- Code::Blocks/MinGW: the SVG is the source artwork; a local Windows `.ico` conversion may be supplied when packaging the executable.
- Visual C++: the existing native application remains buildable without requiring an icon resource. Packaging can add an `.ico` generated from the SVG master.

The repository does not claim that a Windows `.ico` has been generated until an actual raster/icon conversion has been performed.
