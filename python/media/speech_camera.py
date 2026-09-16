"""NLP media bridge: speech -> text -> NLP, plus camera -> vision -> OCR/NLP."""
from __future__ import annotations
import re
COMMANDS={'start recording':'record_start','stop recording':'record_stop','open camera':'camera_start','close camera':'camera_stop','take photo':'snapshot','scan camera':'vision_scan'}
def voice_to_command(text:str):
    n=re.sub(r'\s+',' ',text.lower().strip()); key=next((k for k in COMMANDS if k in n),None)
    return {'text':text,'normalized':n,'action':COMMANDS.get(key),'requires_confirmation':bool(key)}
def speech_to_text(audio_path:str,provider='whisper'):
    if provider=='whisper':
        import whisper
        return whisper.load_model('base').transcribe(audio_path)['text'].strip()
    import vosk, json, wave
    model=vosk.Model(); rec=vosk.KaldiRecognizer(model,16000)
    with wave.open(audio_path,'rb') as f:
        while data:=f.readframes(4000): rec.AcceptWaveform(data)
    return json.loads(rec.FinalResult()).get('text','')
def camera_to_cv(image_path:str):
    import cv2
    image=cv2.imread(image_path); gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY); edges=cv2.Canny(gray,80,160)
    return {'shape':list(image.shape),'edge_pixels':int((edges>0).sum()),'next':['OCR','face','object','pose','gesture','segmentation','tracking']}
