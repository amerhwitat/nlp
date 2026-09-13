# Containerized version

Build and run:

```bash
docker compose build
docker compose up
```

The compose stack exposes the service ports used by the repository and persists application data/logs in named volumes. `THAMUDIC_OCR_TIMEOUT` defaults to 45 seconds for bounded OCR work.

Interactive diagnostics:

```bash
docker compose run --rm nlp-app bash
```

The container is non-root. Tkinter/native desktop operation remains available outside the container; container execution is intended for CLI, service, batch, and headless workflows.
