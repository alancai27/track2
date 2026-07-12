#!/usr/bin/env python3
"""Local Track 2 caption judge — score accuracy + style_match 0-1.

Uses each clip's perception description as ground-truth content (the judge
cannot see video). Scores every caption with Groq llama-3.3-70b-versatile
at temperature 0.

Usage:
  python eval/judge.py io/output/debug.json
  python eval/judge.py --debug io/output/debug.json --out io/output/judge.json

Input is preferably the debug sidecar from entrypoint (has description +
captions). Plain results.json alone cannot be judged without descriptions.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# Reuse the app's Groq client / 429 handling.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from llm import chat  # noqa: E402
from styling import STYLE_SPECS, STYLES  # noqa: E402

JUDGE_MODEL = os.environ.get(
    "JUDGE_MODEL", "accounts/fireworks/models/kimi-k2p6"
)
_JSON_RE = re.compile(r"\{.*\}", re.S)

STYLE_BLURBS = {
    "formal": (
        "polished, objective, professional, factual — no humor, no opinion, "
        "no exclamation marks"
    ),
    "sarcastic": (
        "dry irony / deadpan mockery / ironic praise / fake enthusiasm / "
        "mock profundity — lightly mocking while still accurate; NOT generic "
        "'oh great / just what the world needed' crutches"
    ),
    "humorous_tech": (
        "genuinely funny caption using a fitting tech/programming metaphor "
        "(git, APIs, RAM, caching, debugging, etc.) — clever, not forced"
    ),
    "humorous_non_tech": (
        "genuinely funny everyday caption — playful, relatable, warm; "
        "absolutely no tech or programming jargon"
    ),
}


def _load_clips(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("clips") or data.get("tasks") or data.get("results") or []
    clips = []
    for item in data:
        if not isinstance(item, dict) or not item.get("task_id"):
            continue
        caps = item.get("captions") or {}
        desc = (item.get("description") or "").strip()
        clips.append({
            "task_id": item["task_id"],
            "description": desc,
            "perception_ok": bool(item.get("perception_ok", bool(desc))),
            "captions": {s: str(caps.get(s, "")).strip() for s in STYLES},
        })
    return clips


def _parse_scores(text: str) -> dict | None:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    try:
        acc = float(obj.get("accuracy"))
        sty = float(obj.get("style_match"))
    except (TypeError, ValueError):
        return None
    acc = max(0.0, min(1.0, acc))
    sty = max(0.0, min(1.0, sty))
    reason = str(obj.get("reason", "")).strip()
    return {"accuracy": acc, "style_match": sty, "reason": reason}


def judge_caption(description: str, style: str, caption: str) -> dict:
    """Return {accuracy, style_match, reason} with 429-safe retries via chat()."""
    if not description.strip():
        return {
            "accuracy": 0.0,
            "style_match": 0.0,
            "reason": "no perception description available",
        }
    if not caption.strip():
        return {
            "accuracy": 0.0,
            "style_match": 0.0,
            "reason": "empty caption",
        }

    style_blurb = STYLE_BLURBS.get(style, STYLE_SPECS.get(style, style))
    prompt = (
        f"You are a strict grading judge for video captions.\n\n"
        f"Video description (ground truth of what is in the clip):\n"
        f"{description}\n\n"
        f"Requested style: {style}\n"
        f"Style definition: {style_blurb}\n\n"
        f"Caption to grade:\n{caption}\n\n"
        f"Score 0.0-1.0 on TWO axes:\n"
        f"- accuracy: how accurately the caption reflects the video "
        f"description (facts, subjects, actions). 1=fully grounded, "
        f"0=wrong or generic filler unrelated to this clip.\n"
        f"- style_match: how well the caption matches the requested "
        f"{style} tone/voice. 1=perfect match, 0=wrong voice.\n\n"
        f"Return ONLY JSON, no markdown:\n"
        f'{{"accuracy": 0.0, "style_match": 0.0, "reason": "one short sentence"}}'
    )
    messages = [{"role": "user", "content": prompt}]
    text = chat(
        JUDGE_MODEL,
        messages,
        max_tokens=200,
        temperature=0.0,
        rate_limit_retries=8,
    )
    parsed = _parse_scores(text)
    if parsed:
        return parsed
    return {
        "accuracy": 0.0,
        "style_match": 0.0,
        "reason": f"unparseable judge output: {text[:160]}",
    }


def _avg(nums: list[float]) -> float:
    return sum(nums) / len(nums) if nums else 0.0


def run_judge(clips: list[dict]) -> dict:
    rows = []
    by_style: dict[str, dict[str, list[float]]] = {
        s: {"accuracy": [], "style_match": [], "combined": []} for s in STYLES
    }
    all_acc: list[float] = []
    all_sty: list[float] = []
    all_comb: list[float] = []

    for clip in clips:
        tid = clip["task_id"]
        desc = clip["description"]
        print(f"\n=== {tid}  perception_ok={clip['perception_ok']} ===",
              flush=True)
        if desc:
            print(f"  desc: {desc[:140]}...", flush=True)
        else:
            print("  desc: (MISSING)", flush=True)

        clip_scores = {}
        for style in STYLES:
            cap = clip["captions"].get(style, "")
            scored = judge_caption(desc, style, cap)
            clip_scores[style] = {**scored, "caption": cap}
            acc, sty = scored["accuracy"], scored["style_match"]
            comb = (acc + sty) / 2.0
            by_style[style]["accuracy"].append(acc)
            by_style[style]["style_match"].append(sty)
            by_style[style]["combined"].append(comb)
            all_acc.append(acc)
            all_sty.append(sty)
            all_comb.append(comb)
            print(
                f"  {style:18s}  acc={acc:.2f}  style={sty:.2f}  "
                f"avg={comb:.2f}  | {cap[:70]}",
                flush=True,
            )
            print(f"    reason: {scored['reason']}", flush=True)
            time.sleep(0.4)  # light TPM cushion between judge calls

        rows.append({
            "task_id": tid,
            "description": desc,
            "perception_ok": clip["perception_ok"],
            "scores": clip_scores,
            "clip_accuracy": _avg([clip_scores[s]["accuracy"] for s in STYLES]),
            "clip_style_match": _avg(
                [clip_scores[s]["style_match"] for s in STYLES]
            ),
        })

    summary = {
        "n_clips": len(clips),
        "n_captions": len(all_comb),
        "overall_accuracy": round(_avg(all_acc), 4),
        "overall_style_match": round(_avg(all_sty), 4),
        "overall_combined": round(_avg(all_comb), 4),
        "by_style": {
            s: {
                "accuracy": round(_avg(by_style[s]["accuracy"]), 4),
                "style_match": round(_avg(by_style[s]["style_match"]), 4),
                "combined": round(_avg(by_style[s]["combined"]), 4),
            }
            for s in STYLES
        },
    }
    return {"summary": summary, "clips": rows}


def _print_summary(summary: dict) -> None:
    print("\n" + "=" * 60, flush=True)
    print("JUDGE SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(
        f"clips={summary['n_clips']}  captions={summary['n_captions']}",
        flush=True,
    )
    print(
        f"OVERALL  accuracy={summary['overall_accuracy']:.3f}  "
        f"style_match={summary['overall_style_match']:.3f}  "
        f"combined={summary['overall_combined']:.3f}",
        flush=True,
    )
    print("\nBy style:", flush=True)
    print(f"  {'style':18s}  {'accuracy':>8}  {'style':>8}  {'combined':>8}",
          flush=True)
    for s, vals in summary["by_style"].items():
        print(
            f"  {s:18s}  {vals['accuracy']:8.3f}  "
            f"{vals['style_match']:8.3f}  {vals['combined']:8.3f}",
            flush=True,
        )
    print(
        f"\nBy axis:  accuracy={summary['overall_accuracy']:.3f}  "
        f"style_match={summary['overall_style_match']:.3f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Track 2 caption judge")
    parser.add_argument(
        "input",
        nargs="?",
        default="io/output/debug.json",
        help="Path to debug.json (captions + descriptions)",
    )
    parser.add_argument(
        "--out",
        default="io/output/judge.json",
        help="Where to write full judge report JSON",
    )
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"ERROR: {path} not found. Run the pipeline first so "
              f"entrypoint writes debug.json with perception descriptions.",
              file=sys.stderr)
        sys.exit(1)

    clips = _load_clips(path)
    if not clips:
        print(f"ERROR: no clips in {path}", file=sys.stderr)
        sys.exit(1)
    missing = sum(1 for c in clips if not c["description"])
    print(f"[judge] model={JUDGE_MODEL}  clips={len(clips)}  "
          f"missing_desc={missing}", flush=True)

    report = run_judge(clips)
    _print_summary(report["summary"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[judge] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
