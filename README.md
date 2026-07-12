# amd_track2 — Video Captioning Agent

Fireworks MiniMax M3 grounds each clip (structured scene card → vision audit),
then Kimi K2P6 writes the four required styles sequentially with a light
keyword retry. Keyframes: 3–6 stills including open/close plus scene cuts,
long-edge 1024px.

## Layout

- `app/entrypoint.py` — seed results early, concurrent clips, always exit 0
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

## Submit

CI builds `linux/amd64` → `ghcr.io/alancai27/amd_track2:latest` with the
baked `FIREWORKS_API_KEY` secret.
