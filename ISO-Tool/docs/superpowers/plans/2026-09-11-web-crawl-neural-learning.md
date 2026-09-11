# ISO-Tool Web Crawl + Neural Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add safe integrated web search/crawling and repository-learning with RNN/LLM-ready inference to every ISO-Tool language implementation.

**Architecture:** Python is the reference implementation with standard-library HTTP/HTML/robots handling and optional PyTorch RNN/Transformer adapters. Java, C#/.NET and C++ provide dependency-light parity engines. All implementations share JSON feature names, provenance and safety rules.

**Tech Stack:** Python 3/Tkinter; optional PyTorch; Java 17/Swing; C#/.NET 6/WPF; C++20/Win32; JSON; HTTP(S); RFC 9309.

**Spec:** `ISO-Tool/docs/GUI_PARITY_AND_RUNTIME_HARDENING.md` plus this plan.

## Global Constraints
- Respect robots.txt and treat web content as untrusted evidence.
- Cache robots.txt for no more than 24 hours unless unreachable.
- Never execute arbitrary downloaded scripts.
- Preserve URL, retrieval time, content hash and evidence type.
- Existing ISO build pipeline remains authoritative.
- GUI feature names and command IDs remain synchronized.
- Neural/LLM inference recommends actions but cannot silently authorize execution.

### Task 1: Python crawler, search and learning engine
- [ ] Create `web_engine.py`, `learning_engine.py`, `neural_engine.py` and offline tests.
- [ ] Implement URL normalization, HTTP fetching, robots policy, depth/page/host limits, duplicate hashing and link extraction.
- [ ] Implement configurable web search and provenance records.
- [ ] Ingest repository documentation/build files into JSONL knowledge records.
- [ ] Implement a dependency-light recurrent sequence model plus optional PyTorch RNN/Transformer adapters.
- [ ] Generate evidence-backed installation/build recommendations.

### Task 2: GUI contract
- [ ] Expand `feature_manifest.json` with Knowledge & AI features.
- [ ] Wire Python, Java, C# and C++ GUIs to their engines/adapters.
- [ ] Keep labels/order identical across languages.

### Task 3: Java engine
- [ ] Create `WebLearningEngine.java` and offline JUnit tests.
- [ ] Implement HTTP crawling, bounded link extraction, robots policy, hashing, local documentation ingestion, sequence scoring and JSON provenance.

### Task 4: C# engine
- [ ] Create `WebLearningEngine.cs` and offline tests.
- [ ] Implement HttpClient crawling, robots policy, documentation ingestion, sequence learning, provenance and LLM/search adapter interfaces.

### Task 5: C++ engine
- [ ] Create `WebLearningEngine.hpp/.cpp`.
- [ ] Add the engine to the unified GUI project.
- [ ] Implement URL/document knowledge structures, local crawl primitives, hashing and sequence scoring.

### Task 6: Documentation and CI
- [ ] Add `WEB_CRAWLER_AND_NEURAL_LEARNING.md`.
- [ ] Update GUI button reference, README and CI.
- [ ] Add feature-parity and Python-learning smoke tests.

### Task 7: Mirror
- [ ] Apply the same source, GUI contract, tests and docs to `amerhwitat/ChimeraIIOS/ISO-Tool`.
- [ ] Verify intended synchronized files and commit independently.
