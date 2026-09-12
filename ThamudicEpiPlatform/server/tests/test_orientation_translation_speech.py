from server.ocr_geometry import ReadingDirection, analyze_geometry, build_engine_options
from server.translation_engine import prepare_translation, proof_translation
from server.speech_proof import prove_for_speech

def test_geometry_supports_vertical_spiral_and_weathering():
    h = analyze_geometry(3000, 900, contrast=0.05, blur=20, orientation_hint='ttb')
    assert h.direction is ReadingDirection.TTB
    assert 'vertical-line-segmentation' in h.operations
    assert 'denoise' in h.operations
    assert build_engine_options(h)['vertical_text'] is True

def test_translation_proof_preserves_mode_and_requires_review_when_empty():
    r = prepare_translation('古典', 'classical-chinese', 'en', 'meaning')
    r = proof_translation(r)
    assert r['mode'] == 'meaning'
    assert r['review_required'] is True

def test_speech_proof_never_invents_phonemes():
    r = prove_for_speech('𐪑', 'ancient-north-arabian', reconstructed=True)
    assert r['phonemes'] == []
    assert r['reconstructed'] is True
    assert r['review_required'] is True
