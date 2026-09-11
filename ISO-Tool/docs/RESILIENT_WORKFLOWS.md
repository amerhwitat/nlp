# Resilient ISO-Tool workflows

ISO-Tool supports fail-forward execution and offline recovery.

## Fail-forward

Independent compilation, assembly, documentation, boot-artifact, and packaging jobs are isolated. A recoverable runtime/tool failure is logged with its type and message, the job is recorded as failed/skipped, progress advances, and unrelated jobs continue. A safety or integrity failure may still stop the workflow.

## Workflow entry points

- `analyze-source` — inspect a local checkout or acquired repository.
- `build-compiled-images` — execute the authorized compile/assemble/link plan and collect artifacts.
- `import-boot-image` — inspect a local ISO/IMG/BIN and copy a bounded boot-sector region into staging.
- `build-iso` — stage artifacts and invoke the selected ISO/IMG backend.
- `validate-image` — validate output metadata and hashes.

Imported boot bytes are treated as inert data. ISO-Tool does not execute imported boot code during import.

## Local repositories

The source field accepts a local repository directory as well as a GitHub source. Local operation does not require Internet access. The repository is inventoried directly and can proceed through the configured build plan.

## Internet recovery

Remote acquisition can monitor connectivity and retry after connectivity returns. The monitor periodically checks a small connectivity endpoint (by default GitHub TCP/443). Network errors may be retried indefinitely when configured; non-network exceptions are not blindly retried. Cancellation must always be available to the GUI.

The application logs online/offline transitions and retry waits in the live operation-details pane.

## GUI observability

All three front ends expose a live operation area containing timestamped/staged events, current status, cumulative progress, skipped-job messages, and recovery information. The UI must remain responsive while work executes in a worker thread/task.

## Boot-image import safety

The import feature supports `.iso`, `.img`, and `.bin` sources. The default import is the first 512 bytes and is bounded; callers must explicitly request another bounded region. The source image is never modified.

## Important distinction

"Continue after error" means recoverable job isolation, not suppression of every failure. Missing required boot artifacts, invalid image metadata, corrupted outputs, unsafe disk-write requests, or policy violations remain visible and can block final publication of an image.
