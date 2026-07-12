# amd_track2 — Video Captioning Agent (AMD Hackathon ACT II, Track 2)

Pipeline per clip: download → scene-change keyframes (≤8, 256px) + optional
local **faster-whisper tiny** ASR → **Fireworks minimax-m3** draft+verify →
**kimi-k2p6** sequential captions with variety + guardrails.

## Layout
- `app/entrypoint.py` — orchestrator: seeds valid fallback output at t≈0,
  incremental atomic rewrites, 1-worker pool + staggered submits,
  wall-clock budget (~9.3 min soft), perception REAL/FALLBACK + token-usage
  summary, always exits 0.
- `app/video.py` — download; ffmpeg scene-detect keyframes
  (`gt(scene,0.3)`), fall back to evenly-spaced if &lt;3; cap 8.
- `app/transcribe.py` — optional faster-whisper tiny/int8/CPU ASR; hard
  timeout + fail-open (`AUTO_TRANSCRIBE=true`).
- `app/perception.py` — `minimax-m3` draft+verify; transcript fed into
  description when present (`reasoning_effort=none`).
- `app/styling.py` — `kimi-k2p6` sequential per-style calls, prior-caption
  variety, temps, keyword guardrails.
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
