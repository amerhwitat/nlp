package org.chimera.nlp;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Ordered provider pipeline that explicitly reports missing disciplines. */
public final class DisciplinePipeline {
    private final Map<Discipline, DisciplineProvider> providers = new EnumMap<>(Discipline.class);
    public DisciplinePipeline(List<DisciplineProvider> providers) {
        for (DisciplineProvider provider : Objects.requireNonNull(providers, "providers")) {
            Objects.requireNonNull(provider, "provider");
            if (this.providers.putIfAbsent(provider.discipline(), provider) != null)
                throw new IllegalArgumentException("duplicate provider for " + provider.discipline());
        }
    }
    public List<StageResult> run(List<Discipline> requested, Map<String, String> input) {
        List<StageResult> results = new ArrayList<>();
        for (Discipline discipline : Objects.requireNonNull(requested, "requested")) {
            DisciplineProvider provider = providers.get(Objects.requireNonNull(discipline, "discipline"));
            results.add(provider == null
                    ? StageResult.unavailable(discipline.name(), "No provider registered; no silent fallback was used")
                    : provider.execute(input == null ? Map.of() : Map.copyOf(input)));
        }
        return List.copyOf(results);
    }
}