# syntax=docker/dockerfile:1
# Faithful port of pjmorales1123/amd-track2-agent-v2 (0.92 / top-3).
# Built for linux/amd64 in CI (see .github/workflows/build.yml).
FROM --platform=linux/amd64 python:3.11-slim

# Track 2 harness does not inject API credentials — bake Fireworks key in.
ARG FIREWORKS_API_KEY
ENV FIREWORKS_API_KEY=${FIREWORKS_API_KEY}

# FFmpeg + runtime libs; temporary build tools for faster-whisper/CTranslate2.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y build-essential cmake \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Pre-download whisper models so runtime is not spent fetching them.
ENV WHISPER_CACHE_DIR=/app/models
RUN python - <<'PY'
from faster_whisper import WhisperModel
for size in ("base", "tiny"):
    print(f"Pre-downloading whisper model: {size}")
    WhisperModel(size, device="cpu", compute_type="int8", download_root="/app/models")
PY

COPY agent.py .
COPY config.py .
COPY schemas.py .
COPY pipeline/ ./pipeline/
COPY examples/ ./examples/

RUN mkdir -p /input /output

CMD ["python", "agent.py"]
