"""Build-plan engine: deterministic precedence first, optional AI refinement second."""
from __future__ import annotations
import json
from pathlib import Path
from .document_intelligence import scan_repository, infer_components

BASE_ORDER=['toolchain','generated_sources','libraries','drivers','kernel','bootloader','system_services','applications','filesystem_staging','boot_images','iso_mastering']

def make_plan(root:Path, knowledge_dir:Path)->dict:
    index=scan_repository(root); components=infer_components(index)
    steps=[]
    for name in BASE_ORDER:
        enabled=True
        if name=='libraries' and not components['libraries']: enabled=False
        if name=='drivers' and not components['drivers']: enabled=False
        if name=='kernel' and not components['kernel']: enabled=False
        if name=='bootloader' and not components['boot']: enabled=False
        if name=='applications' and not components['applications']: enabled=False
        if name=='filesystem_staging' and not components['filesystem']: enabled=True
        if enabled: steps.append({'id':name,'precedence':len(steps)+1,'status':'planned','reason':'repository/document evidence and deterministic build policy'})
    plan={'schema':1,'repository':str(root.resolve()),'components':components,'steps':steps,'authority':'deterministic-policy','ai_refinement':'optional-and-constrained'}
    knowledge_dir.mkdir(parents=True,exist_ok=True)
    (knowledge_dir/'repository-knowledge.json').write_text(json.dumps(index,indent=2),encoding='utf-8')
    (knowledge_dir/'build-plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    return plan
