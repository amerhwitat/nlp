# ISO-Tool Web Crawler + Neural Learning

ISO-Tool now includes a cross-language Knowledge & AI layer that learns from repository documentation and selected web evidence to improve installation, compilation, dependency and ISO-build planning.

## Pipeline

`repository tree -> documentation/build files -> web search -> bounded crawler -> provenance records -> knowledge index -> RNN/Transformer/LLM adapter -> evidence-backed build plan -> existing authorized build pipeline`

### Web engine
- HTTP(S) search and bounded same-host crawling.
- URL deduplication and content SHA-256 provenance.
- robots.txt enforcement with conservative failure handling and a 24-hour cache.
- Page/depth limits and request timeouts.
- Downloaded web content is data, not executable code.

The robots behavior follows RFC 9309. In particular, robots.txt is a crawler policy mechanism, not authorization, and unreachable rules are handled conservatively. See the RFC for the normative protocol. urlRFC 9309 Robots Exclusion Protocolhttps://www.rfc-editor.org/rfc/rfc9309.html

### Repository learning
The engine indexes README files, manuals, build manifests, source comments and common build-system files. Every record retains its source, retrieval timestamp and SHA-256 hash.

### Neural engine
A dependency-free recurrent model provides deterministic sequence scoring for build traces. The Python implementation can optionally use PyTorch for a real `torch.nn.RNN` backend and Transformer experimentation. PyTorch documents `torch.nn.RNN` as a multi-layer Elman RNN and provides Transformer building blocks for more advanced models. urlPyTorch RNN documentationhttps://docs.pytorch.org/docs/main/generated/torch.nn.RNN.html

The LLM layer is an adapter: ISO-Tool does not silently download, train or execute a third-party model. Model recommendations remain subject to the existing package-install and build authorization controls.

## GUI
The common GUI contract now contains **66 features**, including:

- Web Search
- Crawl Website
- Crawl Documentation
- Crawl Repository References
- Build Knowledge Base
- Index Documentation
- Train RNN
- Train Transformer/LLM
- Run AI Build Analysis
- Generate Installation Plan
- Generate AI Build Plan
- Diagnose Build Error
- Explain ISO Build
- View Sources
- View Learning Dataset
- View Model Metrics
- Offline AI Mode

Python, Java 17, C#/.NET 6 and C++20 expose the same feature vocabulary.

## Safety
AI output is classified as recommendation/evidence. Package installation, scripts, compiler invocations and ISO generation remain controlled by the existing ISO-Tool build pipeline. Never treat arbitrary Internet instructions as trusted commands.

## Offline mode
Offline mode indexes local repository material and existing build journals without network access. It is the preferred mode for reproducible CI and air-gapped environments.
