# amd_track2 — Video Captioning Agent (AMD Hackathon ACT II, Track 2)

Pipeline per clip: download → ffmpeg extracts 4 downsized (512px) frames →
**Groq Llama 4 Scout** writes one rich factual description → **Llama 3.3 70B**
styles it into 4 captions (formal / sarcastic / humorous_tech /
humorous_non_tech) in a single strict-JSON call.

## Layout
- `app/entrypoint.py` — orchestrator: seeds valid fallback output at t≈0,
  incremental atomic rewrites, 3-worker pool, wall-clock budget (9 min soft),
  always exits 0.
- `app/video.py` — download (falls back to ffmpeg streaming the URL directly),
  ffprobe duration, evenly-spaced frame grabs, base64 JPEGs.
- `app/perception.py` — vision model ladder: `VISION_MODEL` env →
  `meta-llama/llama-4-scout-17b-16e-instruct`. First working model is cached.
- `app/styling.py` — text ladder: `STYLE_MODEL` env → `llama-3.3-70b-versatile`
  → `llama-3.1-8b-instant`. One JSON call for all 4 styles; per-style rescue
  calls if JSON is mangled; grounded template fallbacks if everything dies.
- `app/llm.py` — shared OpenAI-compat Groq client; on 400, retries once without
  optional params (Track 1 lesson).

## Local dev
```
cp .env.example .env   # add your GROQ_API_KEY
pip install -r requirements.txt
./run_examples.sh              # runs the 3 dev clips with plain python
./run_examples.sh docker       # runs the built image instead
```

## Pre-submit checklist
- [ ] CI green; manifest step confirms **linux/amd64**
- [ ] GHCR package set to **Public** (verify pull in incognito)
- [ ] `./run_examples.sh docker` → valid JSON, all 4 styles non-empty per clip
- [ ] Captions eyeballed: accurate + 4 clearly distinct voices
- [ ] `.env` not committed
- [ ] lablab form: image `ghcr.io/alancai27/amd_track2:latest` + placeholder
      slides/video
