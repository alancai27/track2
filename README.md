# amd_track2 — Video Captioning Agent (AMD Hackathon ACT II, Track 2)

Pipeline per clip: download → ffmpeg extracts 3–6 frames (256px) →
**Fireworks minimax-m3** draft+verify factual description → **kimi-k2p6**
styles it into 4 captions (formal / sarcastic / humorous_tech /
humorous_non_tech) sequentially with variety + guardrails.

## Layout
- `app/entrypoint.py` — orchestrator: seeds valid fallback output at t≈0,
  incremental atomic rewrites, 1-worker pool + staggered submits,
  wall-clock budget (~9.3 min soft), perception REAL/FALLBACK + token-usage
  summary, always exits 0.
- `app/video.py` — download, ffprobe duration, dynamic 3–6 evenly-spaced
  frames (Fireworks VLM allows up to 30), base64 JPEGs.
- `app/perception.py` — `VISION_MODEL` → `accounts/fireworks/models/minimax-m3`
  with draft + verify passes (`reasoning_effort=none`).
- `app/styling.py` — `STYLE_MODEL` → `accounts/fireworks/models/kimi-k2p6`;
  sequential per-style calls, prior-caption variety, temps, keyword guardrails.
- `app/llm.py` — OpenAI-compat Fireworks client; `FIREWORKS_API_KEY` from env
  (baked into image at CI build time); 429 retries + 400 bare retry.

## Local dev
```
cp .env.example .env   # add your FIREWORKS_API_KEY
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
- [ ] Repo secret `FIREWORKS_API_KEY` set for image bake + smoke test
- [ ] lablab form: image `ghcr.io/alancai27/amd_track2:latest` + placeholder
      slides/video
