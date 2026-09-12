# Thamudic Scanner Architecture

```text
React + Vite browser UI
        |
        | JSON / multipart HTTP + SSE
        v
FastAPI service
   |       |       |
   |       |       +--> SQLite sessions/results/events
   |       +----------> upload validation / export
   v
scanner_adapter
   |
   v
python/thamudic (canonical Unicode + transliteration registry)
```

The adapter is deliberately narrow: HTTP and persistence code do not own historical-script mapping tables. Recognition confidence is presented as scanner output, not as historical certainty.

Session events receive a monotonically increasing sequence number in SQLite. The SSE endpoint emits stored events in sequence order, which also gives a reopened session a durable progress history.

Development defaults bind to localhost. CORS is explicit/configurable, uploads are size- and extension-bounded, and uploaded content is never executed.
