# AMD Track 2 Video Captioning Agent — Project History

## Competition result

- Final score: **0.92**
- Placement: **Top 3** (tied with 1st and 2nd)
- Docker image: `pjmorales04/amd-track2-agent-v2:latest`
- GitHub repo: `https://github.com/pjmorales1123/amd-track2-agent-v2`

## Final architecture

```
/input/tasks.json
  -> download clip
  -> dynamic keyframe extraction (3–6 frames)
  -> optional local Whisper transcription (disabled by default)
  -> MiniMax M3 structured brief
  -> MiniMax M3 verification/correction of the brief
  -> Kimi K2P6 sequential style captions
  -> /output/results.json
```

## Final models

- Vision / brief / verification: Fireworks `accounts/fireworks/models/minimax-m3`
- Captions: Fireworks `accounts/fireworks/models/kimi-k2p6`
- Reasoning effort: `none`
- Transcription (optional): local `faster-whisper` `tiny`

## Key files

- `agent.py` — main orchestration, concurrency, timeouts, placeholder fallback
- `config.py` — environment configuration
- `schemas.py` — Pydantic schemas
- `pipeline/extract.py` — download, scene-aware keyframes, audio extraction
- `pipeline/analyze.py` — MiniMax M3 structured brief + verification
- `pipeline/caption.py` — Kimi K2P6 sequential style captions with guardrails
- `pipeline/transcribe.py` — local faster-whisper
- `Dockerfile` — bakes `FIREWORKS_API_KEY` into image (required by Track 2 harness)
- `.github/workflows/docker-build.yml` — builds and pushes linux/amd64 image
- `.github/workflows/test-image.yml` — simulates harness without env vars

## Why credentials are baked in

The Track 2 harness does **not** inject API keys. Participant guide: "No API key or model restriction is injected. You may call any model, API, or framework: use your own credentials inside the container." Therefore `FIREWORKS_API_KEY` is passed as a Docker build arg and set as `ENV` in the image.

## Key design decisions that worked

1. Two-step visual grounding (brief → verify) reduced hallucinations.
2. Dynamic 3–6 frames kept token costs low while covering clips.
3. Sequential style generation + keyword guardrails improved style separation.
4. Per-clip timeouts and placeholder fallbacks ensured every task got an entry.
5. Kept captions grounded in verified description rather than raw frames.

## Known limitations

- Sea-turtle hallucination on one test clip with transparent overlay/dissolve.
- First/last frame sampling can include fades/title cards.
- Single flattened paragraph may lose structured fact precision.
- Style prompts rely on definitions + keyword guardrails, not few-shot examples.
- Captions allowed 25–60 words, which can drift into filler.

## Testing performed

- Local tests on 3 hymn/prayer videos + 1 Camera Roll clip.
- Official example v1 test via HTTPS.
- Kimi K2P6 vs K2.7 comparison (kept K2P6 for accuracy).
- GitHub Actions harness simulation with examples/tasks.json.

## What was submitted

- Docker image: `pjmorales04/amd-track2-agent-v2:latest`
- Cover image: `submission_cover.png`
- Slides PDF: `submission_slides.pdf`
- Demo video: `submission_video.mp4`
- Submission title: "Style-Aware Vision Caption Agent"

## Security note

The Fireworks token is embedded in the public Docker image. Rotate/regenerate the token after the competition.
