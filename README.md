# amd_track2 — peer 0.92 pipeline port

This branch is a **faithful copy** of
[pjmorales1123/amd-track2-agent-v2](https://github.com/pjmorales1123/amd-track2-agent-v2)
(score **0.92**, top 3), wired to our GHCR image
`ghcr.io/alancai27/amd_track2:latest`.

## Pipeline

```
tasks.json
  -> download clip
  -> dynamic keyframe extraction (3–6 frames; first+last + scene cuts)
  -> optional local Whisper (AUTO_TRANSCRIBE=false by default)
  -> MiniMax M3 structured brief
  -> MiniMax M3 verification/correction
  -> Kimi K2P6 sequential style captions + keyword guardrails
  -> results.json
```

## Models

- Vision / brief / verify: `accounts/fireworks/models/minimax-m3`
- Captions: `accounts/fireworks/models/kimi-k2p6` (`reasoning_effort=none`)

## Layout

- `agent.py` — orchestration, concurrency, per-clip timeout, placeholders
- `config.py` — env-based config
- `schemas.py` — Pydantic schemas
- `pipeline/extract.py` — download + keyframes + audio
- `pipeline/analyze.py` — MiniMax brief + verify
- `pipeline/caption.py` — Kimi sequential styles
- `pipeline/transcribe.py` — optional faster-whisper

## Local run

```bash
cp .env.example .env   # FIREWORKS_API_KEY=...
pip install -r requirements.txt
./run_examples.sh
```

## Docker / submit

CI on `main` (or workflow_dispatch) builds `linux/amd64` and pushes
`ghcr.io/alancai27/amd_track2:latest` with the baked Fireworks key.

See `PROJECT_HISTORY.md`, `IMPROVEMENT_PLAN.md`, and `SUBMISSION.md`
(from the original repo) for context.
