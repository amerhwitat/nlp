# Performance & Concurrency Policy

Use bounded concurrency for independent, non-sensitive NLP/data-processing workloads.

- Prefer multiprocessing for Python CPU-bound pure-Python work; threads are primarily for I/O or libraries that release the GIL.
- Use bounded worker pools and avoid nested pools/oversubscription.
- Keep deterministic single-worker execution available for tests and reproducibility.
- Use immutable inputs, per-worker state, and ordered result collection when output order matters.
- Keep GUI/framework state on its owning thread.
- Do not parallelize security-sensitive key-search, credential-search, or cryptographic guessing workflows merely for speed.
- Benchmark representative workloads before changing defaults.
