"""Constrained RNN/LLM planning interface.

The engine is intentionally provider-neutral. It can use a local llama.cpp-compatible
CLI/server for LLM reasoning, while deterministic ISO-Tool policy remains authoritative.
A lightweight sequence/RNN hook may be supplied by a future trained model; no model
weights are silently downloaded.
"""
from __future__ import annotations
import json, os, shutil, subprocess
from pathlib import Path

def discover_backends()->dict:
    return {'llama_cpp': bool(shutil.which('llama-cli') or shutil.which('llama')), 'rnn_model': bool(os.environ.get('ISO_TOOL_RNN_MODEL'))}

def propose_refinement(plan:dict, prompt_context:str='', model:Path|None=None)->dict:
    result={'schema':1,'backend':'none','approved_steps':plan.get('steps',[]),'recommendations':[],'confidence':0.0}
    cli=shutil.which('llama-cli') or shutil.which('llama')
    if not cli or not model or not model.exists():
        result['recommendations'].append('No local LLM model configured; deterministic plan remains authoritative.')
        return result
    prompt=('You are a build-plan assistant. Do not invent commands. Review this JSON plan and '
            'return JSON recommendations only. Never reorder mandatory dependencies.\n'+json.dumps(plan)+'\n'+prompt_context)
    try:
        p=subprocess.run([cli,'-m',str(model),'--temp','0','-n','512','-p',prompt],capture_output=True,text=True,timeout=600)
        if p.returncode==0:
            result['backend']='llama.cpp'; result['raw']=p.stdout[-20000:]; result['confidence']=0.5
    except (OSError,subprocess.SubprocessError):
        result['recommendations'].append('Local LLM invocation failed; deterministic plan retained.')
    return result
