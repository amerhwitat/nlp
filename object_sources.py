"""Cultural-heritage source adapters and rights-aware provenance normalization."""
from __future__ import annotations

def build_iiif_url(base,width=1200,region="full",rotation="0",quality="default",fmt="jpg"):
    base=base.rstrip("/")
    if base.endswith("/info.json"): base=base[:-10]
    return f"{base}/{region}/{width},/{rotation}/{quality}.{fmt}"

def normalize_source_record(raw,source="unknown"):
    return {
      "source":source,
      "source_record_id":str(raw.get("id") or raw.get("objectID") or raw.get("@id") or ""),
      "title":raw.get("title") or raw.get("name") or raw.get("label") or "Untitled object",
      "description":raw.get("description") or raw.get("summary") or "",
      "image_url":raw.get("image_url") or raw.get("image") or raw.get("primaryImage") or "",
      "image_iiif":raw.get("iiif") or raw.get("iiif_manifest") or "",
      "license":raw.get("license") or raw.get("rights") or "",
      "source_url":raw.get("source_url") or raw.get("url") or raw.get("record_url") or "",
      "object_type":raw.get("object_type") or raw.get("type") or "",
      "period_key":raw.get("period_key") or "",
      "culture":raw.get("culture") or "",
      "material":raw.get("material") or "",
      "site":raw.get("site") or raw.get("place") or "",
      "tags":raw.get("tags") or [],
    }

def source_catalog():
    return [
      {"name":"OCIANA","homepage":"https://ociana.osu.edu/","rights_policy":"Preserve record provenance and check current terms before bulk reuse.","image_policy":"Prefer record/image links and preserve credits."},
      {"name":"DASI","homepage":"https://dasi.cnr.it/","rights_policy":"Check item/dataset terms before reuse.","image_policy":"Prefer source links or IIIF when offered."},
      {"name":"The Metropolitan Museum of Art","homepage":"https://www.metmuseum.org/","rights_policy":"Open Access dataset is CC0 where applicable.","image_policy":"Use public-domain image flags from the API."},
      {"name":"Smithsonian Open Access","homepage":"https://www.si.edu/openaccess","rights_policy":"Metadata is CC0; media availability depends on item rights.","image_policy":"Reuse only media explicitly supplied as open access."},
      {"name":"Europeana","homepage":"https://www.europeana.eu/","rights_policy":"Respect item-level rights statements and API terms.","image_policy":"Store media URL and rights statement; IIIF where available."},
      {"name":"IIIF","homepage":"https://iiif.io/","rights_policy":"IIIF is an interoperability standard, not a blanket image license.","image_policy":"Always preserve originating institution rights."},
    ]
