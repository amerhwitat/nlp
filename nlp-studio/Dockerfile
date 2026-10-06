# NLP / Ancient Script Studio — backend image.
# Build context is the repository root so the image can reuse python/thamudic
# and the data/ registry without modifying the existing code.
#
#   docker build -f nlp-studio/Dockerfile -t nlp-studio .

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Tesseract provides the lightweight OCR fallback when EasyOCR/PyTorch is absent.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for better layer caching.
COPY nlp-studio/requirements.txt ./nlp-studio/requirements.txt
RUN pip install -r nlp-studio/requirements.txt

# Copy the app and the reused, read-only sources.
COPY nlp-studio ./nlp-studio
COPY python ./python
COPY data ./data

WORKDIR /app/nlp-studio
EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
