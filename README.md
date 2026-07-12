# amd_track2 — Video Captioning Agent

MiniMax M3 grounds each clip (JSON brief → verify), then Kimi K2P6 writes
four sequential styles with keyword retries. Keyframes: 3–6 stills with
open/close + scene cuts @ 1024px long edge.

## Layout

- `app/entrypoint.py` — concurrent clips, early results seed, always exit 0
- `app/frames.py` — download + scene-aware keyframes
- `app/vision.py` — two-pass visual grounding
- `app/captions.py` — sequential styled captions
- `app/client.py` / `app/settings.py` — Fireworks client + env knobs

## Local

```bash
cp .env.example .env
pip install -r requirements.txt
./run_examples.sh
```

## Local score estimate

```bash
python eval/local_judge.py run --limit 3
```

## Submit

CI builds `linux/amd64` → `ghcr.io/alancai27/amd_track2:latest`.
