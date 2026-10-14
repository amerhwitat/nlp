package org.chimera.nlp;

import org.junit.jupiter.api.Test;
import java.text.Normalizer;
import java.util.List;
import java.util.Map;
import static org.junit.jupiter.api.Assertions.*;

class DisciplinePipelineTest {
    @Test void normalizesUnicodeAndReportsMetadata() {
        var pipeline = new DisciplinePipeline(List.of(new TextNormalizationProvider()));
        var result = pipeline.run(List.of(Discipline.NLP), Map.of("text", "e\u0301  Thamudic")).get(0);
        assertEquals(StageResult.Status.SUCCESS, result.status());
        assertEquals(Normalizer.normalize("e\u0301  Thamudic", Normalizer.Form.NFC), result.outputs().get("normalizedText"));
        assertEquals("2", result.outputs().get("wordCount"));
        assertTrue(result.evidence().contains("java.text.Normalizer:NFC"));
    }
    @Test void unavailableProvidersAreExplicit() {
        var pipeline = new DisciplinePipeline(List.of(new TextNormalizationProvider()));
        var results = pipeline.run(List.of(Discipline.COMPUTER_VISION, Discipline.NLP), Map.of("text", "abc"));
        assertEquals(StageResult.Status.UNAVAILABLE, results.get(0).status());
        assertTrue(results.get(0).message().contains("no silent fallback"));
        assertEquals(StageResult.Status.SUCCESS, results.get(1).status());
    }
    @Test void missingTextIsAStageFailureNotAnEmptySuccess() {
        var pipeline = new DisciplinePipeline(List.of(new TextNormalizationProvider()));
        assertEquals(StageResult.Status.FAILED, pipeline.run(List.of(Discipline.NLP), Map.of()).get(0).status());
    }
    @Test void rejectsDuplicateProviders() {
        assertThrows(IllegalArgumentException.class, () -> new DisciplinePipeline(
                List.of(new TextNormalizationProvider(), new TextNormalizationProvider())));
    }
}