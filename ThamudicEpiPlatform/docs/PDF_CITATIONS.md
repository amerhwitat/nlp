# PDF import/export citations

## Implemented dependencies

- [pypdf](https://github.com/py-pdf/pypdf) — bounded PDF parsing/text extraction.
- [ReportLab](https://www.reportlab.com/) — generated research PDF output.
- [JSON Schema](https://json-schema.org/) — PDF manifest contract.

## Design rationale

PDF input is treated as a source container, not as authoritative translation data. Extracted text is retained with page numbers and source SHA-256. OCR or neural interpretations must be stored as separate derived records with model/version/confidence metadata.

PDF output is a reproducible research artifact: report ID, generation timestamp, source identity, rights, provenance, citations and output hash are included in the machine-readable manifest.

## Safety

PDF imports are bounded by file size and page count. Malformed pages produce warnings instead of silently becoming trusted research data. Deployments should also apply upload authentication, request quotas, temporary-file isolation and content-type validation at the reverse proxy.
