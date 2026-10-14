# Java AI discipline pipeline

Java 17+ provider-neutral registry and deterministic text-normalization stage corresponding to the NLP repository's AI discipline boundary.

## Build/test

```sh
mvn -f java/ai-disciplines/pom.xml test
mvn -f java/ai-disciplines/pom.xml package
```

The pipeline supports MACHINE_LEARNING, DEEP_LEARNING, REINFORCEMENT_LEARNING, SYMBOLIC_AI, COMPUTER_VISION, and NLP provider registrations. A missing provider returns UNAVAILABLE rather than silently claiming inference. The included NLP provider only applies Unicode NFC normalization and reports text metadata; it does not claim OCR, translation, or neural-model parity.
