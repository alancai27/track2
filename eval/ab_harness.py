#!/usr/bin/env python3
"""A/B harness for Track 2 caption changes.

Run the current pipeline on a fixed public-clip set, then blindly compare two
runs with a multimodal judge (frames + captions). Order of A/B is randomized
per (clip, style) so the judge cannot favor a label.

Typical workflow:
  # 1. Snapshot current (or known-good) pipeline
  python eval/ab_harness.py run --tag baseline

  # 2. Change code / prompts, then snapshot candidate
  python eval/ab_harness.py run --tag candidate

  # 3. Blind pairwise compare (only ship if candidate wins)
  python eval/ab_harness.py compare \\
      eval/runs/baseline eval/runs/candidate \\
      --out eval/runs/ab_report.json

Quick smoke (2 clips):
  python eval/ab_harness.py run --tag smoke --limit 2
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))

from llm import chat, key_source  # noqa: E402
from perception import describe  # noqa: E402
from styling import STYLES, style_captions  # noqa: E402
from video import extract_frames_b64  # noqa: E402

DEFAULT_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "clips.json"
DEFAULT_RUNS = Path(__file__).resolve().parent / "runs"
JUDGE_MODEL = os.environ.get(
    "AB_JUDGE_MODEL",
    os.environ.get("JUDGE_MODEL", "accounts/fireworks/models/minimax-m3"),
)
_JSON_RE = re.compile(r"\{.*\}", re.S)

STYLE_BLURBS = {
    "formal": (
        "polished, objective, professional, factual — no humor, no opinion"
    ),
    "sarcastic": (
        "dry irony / deadpan mockery / ironic praise — lightly mocking while "
        "still accurate; not generic 'oh great' crutches"
    ),
    "humorous_tech": (
        "genuinely funny caption using a fitting tech/programming metaphor — "
        "clever, not forced"
    ),
    "humorous_non_tech": (
        "genuinely funny everyday caption — playful, relatable; no tech jargon"
    ),
}


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = val


def _load_clips(path: Path, limit: int) -> list[dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise SystemExit(f"fixtures must be a JSON list: {path}")
    clips = []
    for item in data:
        if not isinstance(item, dict):
            continue
        tid = item.get("task_id")
        url = item.get("video_url")
        if tid and url:
            clips.append({
                "task_id": str(tid),
                "video_url": str(url),
                "note": str(item.get("note", "")),
            })
    if limit > 0:
        clips = clips[:limit]
    if not clips:
        raise SystemExit(f"no clips in {path}")
    return clips


def _save_frames(frames_b64: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, b64 in enumerate(frames_b64):
        (out_dir / f"frame_{i:03d}.jpg").write_bytes(base64.b64decode(b64))


def _load_clip_frames(clip_rec: dict, run_dir: Path) -> list[str]:
    """Load JPEGs saved under runs/<tag>/frames/<task_id>/."""
    frame_dir = run_dir / "frames" / clip_rec["task_id"]
    if not frame_dir.is_dir():
        return []
    return [
        base64.b64encode(p.read_bytes()).decode()
        for p in sorted(frame_dir.glob("frame_*.jpg"))
    ]


def cmd_run(args: argparse.Namespace) -> None:
    clips = _load_clips(Path(args.fixtures), args.limit)
    tag = args.tag.strip() or "run"
    run_dir = Path(args.out_dir) / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "frames").mkdir(exist_ok=True)

    print(f"[ab] run tag={tag} clips={len(clips)} key={key_source()} "
          f"out={run_dir}", flush=True)

    records = []
    t0 = time.time()
    for i, clip in enumerate(clips):
        tid = clip["task_id"]
        url = clip["video_url"]
        print(f"\n=== [{i + 1}/{len(clips)}] {tid} ===", flush=True)
        clip_t0 = time.time()
        frames: list[str] = []
        description = ""
        captions: dict[str, str] = {s: "" for s in STYLES}
        err = ""
        try:
            frames = extract_frames_b64(url)
            if frames:
                frame_dir = run_dir / "frames" / tid
                _save_frames(frames, frame_dir)
                description = describe(frames)
                captions = style_captions(
                    description or "A short video clip of a scene."
                )
        except Exception as e:  # noqa: BLE001
            err = str(e)
            print(f"[ab] {tid} FAILED: {e}", flush=True)

        elapsed = time.time() - clip_t0
        rec = {
            "task_id": tid,
            "video_url": url,
            "note": clip.get("note", ""),
            "n_frames": len(frames),
            "description": description,
            "captions": {s: str(captions.get(s, "")).strip() for s in STYLES},
            "elapsed_s": round(elapsed, 2),
            "error": err,
        }
        records.append(rec)
        (run_dir / f"{tid}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2)
        )
        print(
            f"[ab] {tid} done in {elapsed:.1f}s frames={len(frames)} "
            f"desc_len={len(description)}",
            flush=True,
        )

    manifest = {
        "tag": tag,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_clips": len(records),
        "fixtures": str(Path(args.fixtures)),
        "total_elapsed_s": round(time.time() - t0, 2),
        "clips": records,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    print(f"\n[ab] wrote {run_dir / 'manifest.json'} "
          f"({manifest['total_elapsed_s']:.1f}s)", flush=True)


def _load_run(path: Path) -> dict:
    path = path.resolve()
    man = path / "manifest.json"
    if man.exists():
        return json.loads(man.read_text())
    # Accept a bare debug.json / results-like list
    if path.is_file():
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return {"tag": path.stem, "clips": data, "n_clips": len(data)}
    raise SystemExit(f"no manifest.json in {path}")


def _parse_pairwise(text: str) -> dict | None:
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
    out = {}
    for key in ("caption_1", "caption_2"):
        block = obj.get(key)
        if not isinstance(block, dict):
            return None
        try:
            acc = float(block.get("accuracy"))
            sty = float(block.get("style_match"))
        except (TypeError, ValueError):
            return None
        out[key] = {
            "accuracy": max(0.0, min(1.0, acc)),
            "style_match": max(0.0, min(1.0, sty)),
        }
    out["reason"] = str(obj.get("reason", "")).strip()
    return out


def _frame_messages(prompt: str, frames_b64: list[str]) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": prompt}]
    for b64 in frames_b64[:8]:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    return [{"role": "user", "content": content}]


def judge_pairwise(
    frames_b64: list[str],
    style: str,
    caption_1: str,
    caption_2: str,
) -> dict:
    """Blind scores for two captions; returns caption_1/2 accuracy+style."""
    blurb = STYLE_BLURBS.get(style, style)
    prompt = (
        "You are a strict multimodal judge for video captions.\n"
        "You are shown keyframes from a short video clip (chronological).\n\n"
        f"Requested style: {style}\n"
        f"Style definition: {blurb}\n\n"
        f"Caption 1:\n{caption_1}\n\n"
        f"Caption 2:\n{caption_2}\n\n"
        "Score EACH caption 0.0-1.0 on TWO axes:\n"
        "- accuracy: grounded in what is visible in the frames "
        "(subjects/actions/setting). Penalize hallucination and vague filler.\n"
        "- style_match: how well the caption matches the requested style "
        "register (not keyword presence alone).\n\n"
        "Judge the two captions independently; do not force a winner.\n"
        "Return ONLY JSON:\n"
        '{"caption_1": {"accuracy": 0.0, "style_match": 0.0}, '
        '"caption_2": {"accuracy": 0.0, "style_match": 0.0}, '
        '"reason": "one short sentence"}'
    )
    if not frames_b64:
        # Text-only fallback if frames missing
        text = chat(
            JUDGE_MODEL,
            [{"role": "user", "content": prompt}],
            max_tokens=220,
            temperature=0.0,
            rate_limit_retries=8,
        )
    else:
        text = chat(
            JUDGE_MODEL,
            _frame_messages(prompt, frames_b64),
            max_tokens=220,
            temperature=0.0,
            rate_limit_retries=8,
        )
    parsed = _parse_pairwise(text)
    if parsed:
        return parsed
    return {
        "caption_1": {"accuracy": 0.0, "style_match": 0.0},
        "caption_2": {"accuracy": 0.0, "style_match": 0.0},
        "reason": f"unparseable judge output: {text[:160]}",
    }


def _combined(scores: dict) -> float:
    return (scores["accuracy"] + scores["style_match"]) / 2.0


def _winner(a: float, b: float, eps: float = 0.02) -> str:
    if abs(a - b) <= eps:
        return "tie"
    return "baseline" if a > b else "candidate"


def cmd_compare(args: argparse.Namespace) -> None:
    base = _load_run(Path(args.baseline))
    cand = _load_run(Path(args.candidate))
    base_dir = Path(args.baseline).resolve()
    cand_dir = Path(args.candidate).resolve()
    if base_dir.is_file():
        base_dir = base_dir.parent
    if cand_dir.is_file():
        cand_dir = cand_dir.parent

    base_by_id = {c["task_id"]: c for c in base.get("clips", [])}
    cand_by_id = {c["task_id"]: c for c in cand.get("clips", [])}
    common = sorted(set(base_by_id) & set(cand_by_id))
    if args.limit > 0:
        common = common[: args.limit]
    if not common:
        raise SystemExit("no overlapping task_ids between runs")

    rng = random.Random(args.seed)
    print(
        f"[ab] compare baseline={base.get('tag', args.baseline)} "
        f"candidate={cand.get('tag', args.candidate)} "
        f"clips={len(common)} styles={len(STYLES)} "
        f"judge={JUDGE_MODEL} seed={args.seed}",
        flush=True,
    )

    rows = []
    tallies = {
        "accuracy": {"baseline": 0, "candidate": 0, "tie": 0},
        "style_match": {"baseline": 0, "candidate": 0, "tie": 0},
        "combined": {"baseline": 0, "candidate": 0, "tie": 0},
    }
    sum_base = {"accuracy": 0.0, "style_match": 0.0, "combined": 0.0}
    sum_cand = {"accuracy": 0.0, "style_match": 0.0, "combined": 0.0}
    n_scored = 0

    for tid in common:
        b_clip = base_by_id[tid]
        c_clip = cand_by_id[tid]
        # Prefer candidate frames; else baseline; else re-extract.
        frames = _load_clip_frames(c_clip, cand_dir)
        if not frames:
            frames = _load_clip_frames(b_clip, base_dir)
        if not frames and b_clip.get("video_url"):
            print(f"[ab] re-extracting frames for {tid}...", flush=True)
            try:
                frames = extract_frames_b64(b_clip["video_url"])
            except Exception as e:  # noqa: BLE001
                print(f"[ab] frame extract failed for {tid}: {e}", flush=True)

        print(f"\n=== {tid} frames={len(frames)} ===", flush=True)
        for style in STYLES:
            b_cap = str((b_clip.get("captions") or {}).get(style, "")).strip()
            c_cap = str((c_clip.get("captions") or {}).get(style, "")).strip()
            # Randomize presentation order
            swap = bool(rng.getrandbits(1))
            if swap:
                cap1, cap2 = c_cap, b_cap
                map_1, map_2 = "candidate", "baseline"
            else:
                cap1, cap2 = b_cap, c_cap
                map_1, map_2 = "baseline", "candidate"

            judged = judge_pairwise(frames, style, cap1, cap2)
            s1 = judged["caption_1"]
            s2 = judged["caption_2"]
            scores = {
                map_1: s1,
                map_2: s2,
            }
            b_s = scores["baseline"]
            c_s = scores["candidate"]
            b_comb = _combined(b_s)
            c_comb = _combined(c_s)

            w_acc = _winner(b_s["accuracy"], c_s["accuracy"])
            w_sty = _winner(b_s["style_match"], c_s["style_match"])
            w_comb = _winner(b_comb, c_comb)
            tallies["accuracy"][w_acc] += 1
            tallies["style_match"][w_sty] += 1
            tallies["combined"][w_comb] += 1
            sum_base["accuracy"] += b_s["accuracy"]
            sum_base["style_match"] += b_s["style_match"]
            sum_base["combined"] += b_comb
            sum_cand["accuracy"] += c_s["accuracy"]
            sum_cand["style_match"] += c_s["style_match"]
            sum_cand["combined"] += c_comb
            n_scored += 1

            row = {
                "task_id": tid,
                "style": style,
                "swapped": swap,
                "baseline_caption": b_cap,
                "candidate_caption": c_cap,
                "baseline": {**b_s, "combined": round(b_comb, 4)},
                "candidate": {**c_s, "combined": round(c_comb, 4)},
                "winner_accuracy": w_acc,
                "winner_style": w_sty,
                "winner_combined": w_comb,
                "reason": judged.get("reason", ""),
            }
            rows.append(row)
            print(
                f"  {style:18s}  base={b_comb:.2f}  cand={c_comb:.2f}  "
                f"win={w_comb:9s}  swap={swap}",
                flush=True,
            )
            print(f"    reason: {judged.get('reason', '')}", flush=True)
            time.sleep(0.35)

    def _avg(total: float) -> float:
        return round(total / n_scored, 4) if n_scored else 0.0

    summary = {
        "n_pairs": n_scored,
        "n_clips": len(common),
        "judge_model": JUDGE_MODEL,
        "seed": args.seed,
        "mean_baseline": {
            "accuracy": _avg(sum_base["accuracy"]),
            "style_match": _avg(sum_base["style_match"]),
            "combined": _avg(sum_base["combined"]),
        },
        "mean_candidate": {
            "accuracy": _avg(sum_cand["accuracy"]),
            "style_match": _avg(sum_cand["style_match"]),
            "combined": _avg(sum_cand["combined"]),
        },
        "wins": tallies,
        "delta_combined": round(
            _avg(sum_cand["combined"]) - _avg(sum_base["combined"]), 4
        ),
        "recommendation": _recommend(tallies, sum_base, sum_cand, n_scored),
    }

    report = {
        "baseline_tag": base.get("tag", str(args.baseline)),
        "candidate_tag": cand.get("tag", str(args.candidate)),
        "summary": summary,
        "pairs": rows,
    }
    _print_compare_summary(summary)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[ab] wrote {out}", flush=True)


def _recommend(
    tallies: dict,
    sum_base: dict,
    sum_cand: dict,
    n: int,
) -> str:
    if n == 0:
        return "no scores — cannot recommend"
    comb = tallies["combined"]
    cand_wins = comb["candidate"]
    base_wins = comb["baseline"]
    ties = comb["tie"]
    mean_delta = (sum_cand["combined"] - sum_base["combined"]) / n
    # Ship only if candidate wins more pairs and mean delta is positive.
    if cand_wins > base_wins and mean_delta > 0.01:
        return (
            f"SHIP candidate — wins {cand_wins}/{n} combined "
            f"(baseline {base_wins}, ties {ties}), "
            f"mean Δcombined={mean_delta:+.3f}"
        )
    if base_wins > cand_wins or mean_delta < -0.01:
        return (
            f"KEEP baseline — candidate loses "
            f"(cand {cand_wins} / base {base_wins} / ties {ties}), "
            f"mean Δcombined={mean_delta:+.3f}"
        )
    return (
        f"INCONCLUSIVE — cand {cand_wins} / base {base_wins} / ties {ties}, "
        f"mean Δcombined={mean_delta:+.3f}; need more clips or larger gap"
    )


def _print_compare_summary(summary: dict) -> None:
    print("\n" + "=" * 64, flush=True)
    print("A/B COMPARE SUMMARY", flush=True)
    print("=" * 64, flush=True)
    mb = summary["mean_baseline"]
    mc = summary["mean_candidate"]
    print(
        f"pairs={summary['n_pairs']}  clips={summary['n_clips']}",
        flush=True,
    )
    print(
        f"baseline  acc={mb['accuracy']:.3f}  style={mb['style_match']:.3f}  "
        f"comb={mb['combined']:.3f}",
        flush=True,
    )
    print(
        f"candidate acc={mc['accuracy']:.3f}  style={mc['style_match']:.3f}  "
        f"comb={mc['combined']:.3f}",
        flush=True,
    )
    print(f"Δcombined={summary['delta_combined']:+.3f}", flush=True)
    print("\nPair wins (baseline / candidate / tie):", flush=True)
    for axis, w in summary["wins"].items():
        print(
            f"  {axis:12s}  {w['baseline']:3d} / {w['candidate']:3d} / "
            f"{w['tie']:3d}",
            flush=True,
        )
    print(f"\n→ {summary['recommendation']}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="A/B harness for Track 2 caption pipeline",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Generate captions for fixture clips")
    run.add_argument("--tag", required=True, help="Run label (e.g. baseline)")
    run.add_argument(
        "--fixtures",
        default=str(DEFAULT_FIXTURES),
        help="JSON list of {task_id, video_url}",
    )
    run.add_argument(
        "--out-dir",
        default=str(DEFAULT_RUNS),
        help="Parent directory for runs/<tag>/",
    )
    run.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only first N clips (0 = all)",
    )

    cmp_ = sub.add_parser("compare", help="Blind pairwise compare two runs")
    cmp_.add_argument("baseline", help="Path to runs/<baseline_tag>/")
    cmp_.add_argument("candidate", help="Path to runs/<candidate_tag>/")
    cmp_.add_argument(
        "--out",
        default=str(DEFAULT_RUNS / "ab_report.json"),
        help="Where to write the A/B report",
    )
    cmp_.add_argument("--seed", type=int, default=42, help="RNG seed for A/B swap")
    cmp_.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only first N overlapping clips",
    )
    return p


def main() -> None:
    _load_dotenv()
    args = build_parser().parse_args()
    if args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "compare":
        cmd_compare(args)
    else:
        raise SystemExit(f"unknown cmd {args.cmd}")


if __name__ == "__main__":
    main()
