-- Parameterized-query templates. Bind values in the API/CLI layer.
SELECT * FROM object_summary
WHERE (:q IS NULL OR title LIKE '%' || :q || '%' OR site LIKE '%' || :q || '%')
ORDER BY id DESC LIMIT :limit OFFSET :offset;

SELECT * FROM reading_dashboard
WHERE (:status IS NULL OR status = :status)
  AND (:min_conf IS NULL OR confidence >= :min_conf)
ORDER BY confidence DESC NULLS LAST;

SELECT * FROM provenance_dashboard
WHERE (:source_key IS NULL OR source_key = :source_key)
ORDER BY object_id DESC;

SELECT glyph_candidate, COUNT(*) AS occurrences,
       AVG(confidence) AS mean_confidence
FROM annotations
WHERE object_id = :object_id
GROUP BY glyph_candidate
ORDER BY occurrences DESC;
