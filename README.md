# amd_track2 — Video Captioning Agent (AMD Hackathon ACT II, Track 2)

Pipeline per clip: download → temporal-midpoint frames (4/6/8 @ ~768px) →
**Fireworks minimax-m3** draft+verify → structured fact bullets →
**kimi-k2p6** sequential short captions with few-shot exemplars.

## Layout
- `app/entrypoint.py` — orchestrator: seeded fallbacks, atomic rewrites,
  1-worker + stagger, budget, exit 0.
- `app/video.py` — midpoint timestamps `(i+0.5)/n`; optional scene frame if
  meaningfully different; long-edge ~768px.
- `app/perception.py` — minimax-m3 draft + verify (`reasoning_effort=none`).
- `app/styling.py` — kimi-k2p6; fact bullets; few-shot good/bad examples;
  short word budgets; sequential + guardrails.
- `app/llm.py` — Fireworks OpenAI-compat client; key from env / image bake.

## Local dev
```
cp .env.example .env   # add your FIREWORKS_API_KEY
pip install -r requirements.txt
./run_examples.sh
```

## A/B harness (do this before shipping prompt/pipeline changes)

Fixed 10-clip public set + blind multimodal pairwise judge (MiniMax M3 with
frames). Randomizes caption order so the judge cannot favor a label.

```bash
# Snapshot current code as baseline
python eval/ab_harness.py run --tag baseline

# Change code/prompts, then snapshot candidate
python eval/ab_harness.py run --tag candidate

# Blind compare — only ship if recommendation says SHIP
python eval/ab_harness.py compare \
  eval/runs/baseline eval/runs/candidate \
  --out eval/runs/ab_report.json
```

Smoke (2 clips): `python eval/ab_harness.py run --tag smoke --limit 2`

To A/B uncommitted changes against the last commit:
```bash
git stash push -u -m ab-baseline -- app eval
python eval/ab_harness.py run --tag baseline
git stash pop
python eval/ab_harness.py run --tag candidate
python eval/ab_harness.py compare eval/runs/baseline eval/runs/candidate
```

Absolute text-only judge (uses perception descriptions, no frames):
`python eval/judge.py io/output/debug.json`
