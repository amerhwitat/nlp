# Web Implementations

The `web/` layer provides a standalone browser NLP workbench plus a PHP service boundary and TypeScript text-metrics contract. Browser processing is local-first; server APIs should expose explicit, documented operations and must not silently upload user text.

Future WebAssembly and worker implementations should preserve the Python reference behavior through shared JSON fixtures.