CREATE VIEW IF NOT EXISTS object_summary AS
SELECT o.id, o.external_id, o.title, o.script, o.site, o.image_url,
       p.name AS period_name, s.institution, s.rights,
       COUNT(DISTINCT a.id) AS annotation_count,
       COUNT(DISTINCT r.id) AS reading_count,
       AVG(a.confidence) AS annotation_confidence,
       AVG(r.confidence) AS reading_confidence
FROM objects o
LEFT JOIN periods p ON p.id = o.period_id
LEFT JOIN sources s ON s.id = o.source_id
LEFT JOIN annotations a ON a.object_id = o.id
LEFT JOIN readings r ON r.object_id = o.id
GROUP BY o.id;

CREATE VIEW IF NOT EXISTS reading_dashboard AS
SELECT r.id, r.object_id, o.title, r.reading_type,
       r.transliteration, r.arabic_interpretation,
       r.english_interpretation, r.reviewer, r.confidence, r.status
FROM readings r JOIN objects o ON o.id = r.object_id;

CREATE VIEW IF NOT EXISTS provenance_dashboard AS
SELECT o.id AS object_id, o.title, s.source_key, s.institution,
       s.record_url, s.image_policy, s.rights, s.retrieved_at,
       o.provenance
FROM objects o LEFT JOIN sources s ON s.id = o.source_id;

CREATE VIEW IF NOT EXISTS confidence_dashboard AS
SELECT o.id AS object_id, o.title,
       COALESCE(AVG(a.confidence), 0) AS glyph_confidence,
       COALESCE(AVG(r.confidence), 0) AS reading_confidence,
       COUNT(a.id) AS glyph_count,
       COUNT(r.id) AS reading_count
FROM objects o
LEFT JOIN annotations a ON a.object_id = o.id
LEFT JOIN readings r ON r.object_id = o.id
GROUP BY o.id;
