# Linux Commands Source Integration

Chimera II source references include GNU Coreutils, util-linux, iproute2/net-tools, sudo, POSIX shells, and Toybox. Upstream licenses/SPDX metadata must be retained; project code uses adapters.

Performance policy: CPU-heavy independent NLP work may use multiprocessing or bounded native worker pools; I/O may use asynchronous/threaded execution. Avoid shared mutable state and provide deterministic serial execution.

Canonical standalone and Aurora Web UI integration is maintained in `amerhwitat/ChimeraIIOS`.
