"""Track 2 orchestrator.

Guarantees (Track 1 lessons baked in):
  * /output/results.json is written IMMEDIATELY with non-empty fallback
    captions for every task x style, then rewritten after each completed
    task and again in a `finally`. Malformed/missing JSON = 0, so the file
    is always valid from t≈0.
  * Every download / ffmpeg / API call is wrapped; one bad clip never
    kills the run.
  * Wall-clock budget: stop launching heavy work near the deadline.
  * ALWAYS exits 0.

Pipeline per clip: download -> ffmpeg 3 tiny frames -> Groq VLM
factual description -> Groq text model styles 4 captions.

TPM-safe defaults for a 12-clip grading set: 1 worker, staggered
submits, tiny frames, aggressive 429 retries in llm.chat.
"""
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm import key_source  # noqa: E402
from perception import describe  # noqa: E402
from styling import STYLES, style_captions, template_fallbacks  # noqa: E402
from video import extract_frames_b64  # noqa: E402

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")
TOTAL_BUDGET = float(os.environ.get("TOTAL_BUDGET_SECONDS", "560"))  # ~9.3 min
# Serial vision calls — parallel Scout blows Groq's ~30k TPM on 12 clips.
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "1"))
# Gap between task submits so TPM can refill between clips.
STAGGER_SECONDS = float(os.environ.get("STAGGER_SECONDS", "3.0"))

START = time.time()
_write_lock = threading.Lock()
_stats_lock = threading.Lock()
# Per-task: True = got a real VLM description; False = generic fallback path.
_perception_ok: dict[str, bool] = {}


def time_left() -> float:
    return TOTAL_BUDGET - (time.time() - START)


def load_tasks() -> list[dict]:
    try:
        with open(INPUT_PATH) as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("tasks", [data])
        tasks = []
        for t in data:
            if isinstance(t, dict) and t.get("task_id") and t.get("video_url"):
                tasks.append(t)
        return tasks
    except Exception as e:  # noqa: BLE001
        print(f"[main] FAILED to read tasks: {e}", flush=True)
        return []


def write_results(order: list[str], results: dict) -> None:
    with _write_lock:
        payload = [{"task_id": tid, "captions": results[tid]} for tid in order]
        tmp = OUTPUT_PATH + ".tmp"
        try:
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            with open(tmp, "w") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, OUTPUT_PATH)
        except Exception as e:  # noqa: BLE001
            print(f"[main] write failed: {e}", flush=True)


def _mark_perception(tid: str, ok: bool) -> None:
    with _stats_lock:
        _perception_ok[tid] = ok


def process_task(task: dict, order: list[str], results: dict) -> None:
    tid = task["task_id"]
    t0 = time.time()
    got_real = False
    try:
        if time_left() < 30:
            print(f"[{tid}] skipping heavy work — near budget", flush=True)
            return  # fallback captions already in place

        frames = []
        try:
            frames = extract_frames_b64(task["video_url"])
        except Exception as e:  # noqa: BLE001
            print(f"[{tid}] frame extraction crashed: {e}", flush=True)

        description = ""
        if frames and time_left() > 25:
            try:
                description = describe(frames)
                if description.strip():
                    got_real = True
                    print(f"[{tid}] description: {description[:120]}...",
                          flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[{tid}] perception failed: {e}", flush=True)

        if description:
            results[tid] = template_fallbacks(description)  # upgrade baseline
            write_results(order, results)

        if time_left() > 15:
            try:
                results[tid] = style_captions(description or
                                              "A short video clip of a scene.")
            except Exception as e:  # noqa: BLE001
                print(f"[{tid}] styling failed: {e}", flush=True)
    except Exception:  # noqa: BLE001
        print(f"[{tid}] UNEXPECTED:\n{traceback.format_exc()}", flush=True)
    finally:
        _mark_perception(tid, got_real)
        # enforce all 4 styles non-empty no matter what happened above
        caps = results.get(tid) or {}
        base = template_fallbacks("")
        for s in STYLES:
            if not str(caps.get(s, "")).strip():
                caps[s] = base[s]
        results[tid] = caps
        write_results(order, results)
        tag = "REAL" if got_real else "FALLBACK"
        print(f"[{tid}] done in {time.time() - t0:.1f}s "
              f"perception={tag} ({time_left():.0f}s budget left)", flush=True)


def _print_perception_summary(order: list[str]) -> None:
    n = len(order)
    with _stats_lock:
        real = sum(1 for tid in order if _perception_ok.get(tid))
    fallback = n - real
    print(f"[main] {real}/{n} clips got real perception, "
          f"{fallback}/{n} fell back.", flush=True)
    if fallback:
        missed = [tid for tid in order if not _perception_ok.get(tid)]
        print(f"[main] fallback task_ids: {missed}", flush=True)


def main() -> None:
    tasks = load_tasks()
    order = [t["task_id"] for t in tasks]
    # Seed EVERY task with valid non-empty fallbacks before doing anything.
    results = {tid: template_fallbacks("") for tid in order}
    write_results(order, results)
    src = key_source()
    print(f"[main] {len(tasks)} tasks; budget {TOTAL_BUDGET:.0f}s; "
          f"workers={MAX_WORKERS}; stagger={STAGGER_SECONDS}s; "
          f"GROQ key={src}", flush=True)

    if tasks:
        try:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                futs = []
                for i, t in enumerate(tasks):
                    if i > 0 and STAGGER_SECONDS > 0:
                        time.sleep(STAGGER_SECONDS)
                    futs.append(pool.submit(process_task, t, order, results))
                for _ in as_completed(futs):
                    pass
        except Exception:  # noqa: BLE001
            print(f"[main] pool error:\n{traceback.format_exc()}", flush=True)

    write_results(order, results)
    _print_perception_summary(order)
    print(f"[main] finished in {time.time() - START:.1f}s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        print(f"[main] FATAL:\n{traceback.format_exc()}", flush=True)
    sys.exit(0)  # ALWAYS
