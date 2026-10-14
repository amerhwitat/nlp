package org.chimera.nlp;

import java.util.List;
import java.util.Map;
import java.util.Objects;

public record StageResult(String stage, Status status, Map<String, String> outputs, List<String> evidence, String message) {
    public enum Status { SUCCESS, UNAVAILABLE, FAILED }
    public StageResult {
        if (stage == null || stage.isBlank()) throw new IllegalArgumentException("stage is required");
        Objects.requireNonNull(status, "status");
        outputs = Map.copyOf(outputs == null ? Map.of() : outputs);
        evidence = List.copyOf(evidence == null ? List.of() : evidence);
        message = message == null ? "" : message;
    }
    public static StageResult unavailable(String stage, String reason) {
        return new StageResult(stage, Status.UNAVAILABLE, Map.of(), List.of(), reason);
    }
}