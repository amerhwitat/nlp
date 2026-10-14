package org.chimera.nlp;

import java.util.Map;

public interface DisciplineProvider {
    Discipline discipline();
    StageResult execute(Map<String, String> input);
}