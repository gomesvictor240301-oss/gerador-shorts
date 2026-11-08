# Imagem leve: sem PyTorch/Whisper local (a transcrição é feita pela Groq)
FROM python:3.11-slim

# FFmpeg (essencial!) e uma fonte para as legendas
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg fonts-dejavu-core && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Render envia a porta na variável $PORT
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}
