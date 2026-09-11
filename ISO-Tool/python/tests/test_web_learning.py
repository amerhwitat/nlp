import json
from pathlib import Path
from iso_tool.neural_engine import TinyRNN
from iso_tool.learning_engine import ingest_repository, generate_build_plan

def test_rnn_trains_and_scores():
    m=TinyRNN(); m.train([[0,1,2],[1,2,3]],[1,1]); assert 0 <= m.predict_next([1,2,3]) <= 1

def test_repository_ingestion(tmp_path):
    (tmp_path/'README.md').write_text('cmake build ISO',encoding='utf-8')
    r=ingest_repository(tmp_path,tmp_path/'knowledge.jsonl'); assert r['records']==1
    assert 'cmake' in json.loads((tmp_path/'knowledge.jsonl').read_text().splitlines()[0])['text']

def test_plan_requires_authorization(tmp_path):
    (tmp_path/'x.jsonl').write_text('{"text":"cmake"}\n',encoding='utf-8')
    assert generate_build_plan(tmp_path)['authorization_required'] is True
