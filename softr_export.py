"""Export catalog records into a stable Softr Database import shape."""
import csv, json
COLUMNS=['Record ID','Title','Period','Object Type','Culture','Script','Language','Date Start','Date End','Site','Region','Country','Material','Technique','Description','Transliteration','Arabic Translation','English Translation','Source','Source Record ID','Source URL','Image URL','IIIF URL','Image Local Path','Rights / License','Rights Notes','Creator','Provenance','Bibliography','Subjects','Latitude','Longitude','Confidence','Reviewer','Competing Readings','Tags']
MAP={"Record ID":"id","Title":"title","Period":"period_name","Object Type":"object_type","Culture":"culture","Script":"script_key","Language":"language","Date Start":"date_start","Date End":"date_end","Site":"site","Region":"region","Country":"country","Material":"material","Technique":"technique","Description":"description","Transliteration":"transliteration","Arabic Translation":"translation_ar","English Translation":"translation_en","Source":"source_name","Source Record ID":"source_record_id","Source URL":"source_url","Image URL":"image_url","IIIF URL":"image_iiif","Image Local Path":"image_local_path","Rights / License":"license","Rights Notes":"rights_notes","Creator":"creator","Provenance":"provenance","Bibliography":"bibliography","Subjects":"subjects","Latitude":"latitude","Longitude":"longitude","Confidence":"confidence","Reviewer":"reviewer","Competing Readings":"competing_readings","Tags":"tags"}

def object_to_softr_row(obj): return {column:obj.get(MAP[column],"") for column in COLUMNS}
def export_softr_csv(rows,path):
    with open(path,'w',newline='',encoding='utf-8-sig') as fh:
        w=csv.DictWriter(fh,fieldnames=COLUMNS); w.writeheader(); w.writerows(object_to_softr_row(r) for r in rows)
def export_softr_json(rows,path):
    with open(path,'w',encoding='utf-8') as fh: json.dump([object_to_softr_row(r) for r in rows],fh,ensure_ascii=False,indent=2)
