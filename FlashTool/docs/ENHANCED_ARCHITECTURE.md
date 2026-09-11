# Enhanced FlashTool Architecture

## Purpose

FlashTool is organized around a canonical C++ safety core with language bindings and offline analyzers. The architecture separates **inspection**, **planning**, **authorization**, **execution**, and **verification** so a parser cannot accidentally become a flashing primitive.

## Components

```text
                 +-----------------------+
                 | GUI / Web UI / CLI    |
                 +-----------+-----------+
                             |
                     Analysis / Plan API
                             |
                 +-----------v-----------+
                 | Canonical C++ Core    |
                 | Policy + Preflight    |
                 +-----+-----------+-----+
                       |           |
                +------v--+   +---v-------+
                | Parsers |   | Transport |
                | AVB     |   | ADB       |
                | Sparse  |   | Fastboot  |
                | OTA     |   | Fastbootd |
                | Super   |   | USB       |
                +---------+   +-----------+
                       |
                 +-----v-----------------+
                 | Verify + Journal      |
                 +-----------------------+
```

## Safety state machine

1. `DISCOVER` — identify transport and device.
2. `SNAPSHOT` — capture device and partition metadata.
3. `ANALYZE` — inspect artifact without modifying it.
4. `PREFLIGHT` — compare image, partition and device metadata.
5. `DRY_RUN` — produce the exact intended operation set.
6. `CONFIRM` — require an explicit user decision for writes.
7. `EXECUTE` — invoke only an authorized backend operation.
8. `VERIFY` — check hashes, boot state and expected metadata where supported.
9. `JOURNAL` — persist machine-readable results.

## Extensibility

Transport adapters are isolated from image parsers. A future backend can support another documented transport without teaching the parser how to write devices. Likewise, new image formats are analyzers first and do not automatically acquire write privileges.

## Chimera integration

Chimera RegisterN/C8192/R8192 support is an optional acceleration boundary for parallel hashing, image scanning and validation. The fallback implementation remains portable and does not require Chimera hardware.
