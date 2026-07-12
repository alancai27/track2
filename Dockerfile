# Built for linux/amd64 in CI (see .github/workflows/build.yml)
FROM python:3.11-slim

# Injected at build time from GitHub secret — present at runtime for the
# grader (which does not pass FIREWORKS_API_KEY). Never commit the key itself.
ARG FIREWORKS_API_KEY
ENV FIREWORKS_API_KEY=${FIREWORKS_API_KEY}

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ /app/

# Grader mounts /input and /output
ENTRYPOINT ["python", "/app/entrypoint.py"]
