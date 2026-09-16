# Speech + Camera AI pipeline

```text
Microphone -> recorder -> Whisper/Vosk -> text -> NLP/command parser
Camera -> frame capture -> OpenCV preprocessing -> MediaPipe/model tasks -> OCR/object/face/pose/gesture/segmentation/tracking -> NLP
Audio files -> decoder -> ASR -> transcript -> translation/transliteration/corpus analysis
Video files -> decoder -> sampled frames -> CV -> timestamped observations
```

## Providers

Whisper provides multilingual speech recognition, speech translation and language identification. Vosk provides an offline/streaming ASR API with bindings across multiple languages and platforms.

OpenCV is the base camera/image processing adapter. MediaPipe Tasks supplies optional face, hand/gesture, pose, object and segmentation tasks, including live-stream operation.

Microphone/camera access is opt-in. Model downloads are not implicit; deployments provide and verify model files.
