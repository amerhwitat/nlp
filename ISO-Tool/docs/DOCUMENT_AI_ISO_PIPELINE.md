# Document-aware AI ISO pipeline

ISO-Tool now treats repository documentation and build metadata as first-class build inputs.

## Pipeline

1. Select GitHub repository or local repository.
2. Scan source/document/build files and create `knowledge/repository-knowledge.json`.
3. Infer components and create deterministic `knowledge/build-plan.json`.
4. Optionally ask a local RNN/LLM backend for recommendations; it cannot override mandatory dependency ordering or execute unregistered commands.
5. Configure, compile and link with GNU C++ and MSVC in isolated build trees.
6. Stage executables, libraries and boot artifacts.
7. Merge the repository hierarchy into a canonical staging tree.
8. Master the final ISO with the registered image backend.

## AI policy

The AI subsystem is provider-neutral and supports a local llama.cpp-compatible executable. Model weights are not silently downloaded. The deterministic planner remains authoritative when no model is available or when AI output is malformed.

The current local LLM integration is compatible with llama.cpp's CLI model workflow; llama.cpp supports local GGUF models and CPU/GPU backends. See the upstream project: https://github.com/ggml-org/llama.cpp

## Internet solution research

ISO-Tool records recommended image/build backends and their evidence in `tools/internet_solution_policy.json`. Internet research is advisory: sources are not executed as code and arbitrary scripts are never trusted automatically.
