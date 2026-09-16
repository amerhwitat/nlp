from ai_disciplines.discipline_registry import DISCIPLINES
from ai_disciplines.pipeline import AIPipeline, Stage

def test_six_disciplines():
    assert {d.name for d in DISCIPLINES} == {"ML", "DL", "RL", "Symbolic AI", "Computer Vision", "NLP"}

def test_pipeline_audit():
    p = AIPipeline().add(Stage("CV", "ocr", lambda x: x + " text"))
    assert p.run("image") == "image text"
    assert p.audit[0]["discipline"] == "CV"
