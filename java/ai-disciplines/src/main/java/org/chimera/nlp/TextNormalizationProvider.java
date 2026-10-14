package org.chimera.nlp;

import java.text.Normalizer;
import java.util.List;
import java.util.Map;

/** Deterministic text preparation; not OCR, translation, or neural inference. */
public final class TextNormalizationProvider implements DisciplineProvider {
    @Override public Discipline discipline() { return Discipline.NLP; }
    @Override public StageResult execute(Map<String, String> input) {
        String text = input == null ? null : input.get("text");
        if (text == null) return new StageResult("text-normalization", StageResult.Status.FAILED,
                Map.of(), List.of(), "input field 'text' is required");
        String normalized = Normalizer.normalize(text, Normalizer.Form.NFC);
        long codePoints = normalized.codePoints().count();
        long words = normalized.isBlank() ? 0 : normalized.strip().split("\\s+").length;
        return new StageResult("text-normalization", StageResult.Status.SUCCESS,
                Map.of("normalizedText", normalized, "codePoints", Long.toString(codePoints),
                        "wordCount", Long.toString(words)),
                List.of("java.text.Normalizer:NFC"), "Normalized Unicode text; no translation was attempted");
    }
}