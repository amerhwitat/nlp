"""Export catalog records into a stable Softr Database import shape.

The schema deliberately carries provenance and rights information alongside
image URLs so a public catalog does not lose attribution or licensing context.
"""
import csv
import json

COLUMNS = [
    'Record ID','Title','Period','Object Type','Culture','Script','Language',
    'Date Start','Date End','Site','Region','Country','Material','Technique',
    'Description','Transliteration','Arabic Translation','English Translation',
    'Source','Source Record ID','Source URL','Image URL','Image Page URL','IIIF URL',
    'Image Local Path','Rights / License','Rights Notes','Creator','Provenance',
    'Bibliography','Subjects','Latitude','Longitude','Confidence','Reviewer',
    'Competing Readings','Tags','Last Verified'
]

MAP = {
    'Record ID':'id','Title':'title','Period':'period_name','Object Type':'object_type',
    'Culture':'culture','Script':'script_key','Language':'language',
    'Date Start':'date_start','Date End':'date_end','Site':'site','Region':'region',
    'Country':'country','Material':'material','Technique':'technique',
    'Description':'description','Transliteration':'transliteration',
    'Arabic Translation':'translation_ar','English Translation':'translation_en',
    'Source':'source_name','Source Record ID':'source_record_id','Source URL':'source_url',
    'Image URL':'image_url','Image Page URL':'image_page_url','IIIF URL':'image_iiif',
    'Image Local Path':'image_local_path','Rights / License':'license',
    'Rights Notes':'rights_notes','Creator':'creator','Provenance':'provenance',
    'Bibliography':'bibliography','Subjects':'subjects','Latitude':'latitude',
    'Longitude':'longitude','Confidence':'confidence','Reviewer':'reviewer',
    'Competing Readings':'competing_readings','Tags':'tags','Last Verified':'last_verified'
}


def object_to_softr_row(obj):
    return {column: obj.get(MAP[column], '') for column in COLUMNS}


def export_softr_csv(rows, path):
    with open(path, 'w', newline='', encoding='utf-8-sig') as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(object_to_softr_row(row) for row in rows)


def export_softr_json(rows, path):
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump([object_to_softr_row(row) for row in rows], fh, ensure_ascii=False, indent=2)
