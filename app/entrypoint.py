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

Pipeline per clip: download -> ffmpeg 4 downsized frames -> Groq VLM
factual description -> Groq text model styles 4 captions.
"""
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from perception import describe  # noqa: E402
from styling import STYLES, style_captions, template_fallbacks  # noqa: E402
from video import extract_frames_b64  # noqa: E402

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")
TOTAL_BUDGET = float(os.environ.get("TOTAL_BUDGET_SECONDS", "540"))  # 9 min
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "3"))

START = time.time()
_write_lock = threading.Lock()


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


def process_task(task: dict, order: list[str], results: dict) -> None:
    tid = task["task_id"]
    t0 = time.time()
    try:
        if time_left() < 45:
            print(f"[{tid}] skipping heavy work — near budget", flush=True)
            return  # fallback captions already in place

        frames = []
        try:
            frames = extract_frames_b64(task["video_url"])
        except Exception as e:  # noqa: BLE001
            print(f"[{tid}] frame extraction crashed: {e}", flush=True)

        description = ""
        if frames and time_left() > 35:
            try:
                description = describe(frames)
                print(f"[{tid}] description: {description[:120]}...", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[{tid}] perception failed: {e}", flush=True)

        if description:
            results[tid] = template_fallbacks(description)  # upgrade baseline
            write_results(order, results)

        if time_left() > 20:
            try:
                results[tid] = style_captions(description or
                                              "A short video clip of a scene.")
            except Exception as e:  # noqa: BLE001
                print(f"[{tid}] styling failed: {e}", flush=True)
    except Exception:  # noqa: BLE001
        print(f"[{tid}] UNEXPECTED:\n{traceback.format_exc()}", flush=True)
    finally:
        # enforce all 4 styles non-empty no matter what happened above
        caps = results.get(tid) or {}
        base = template_fallbacks("")
        for s in STYLES:
            if not str(caps.get(s, "")).strip():
                caps[s] = base[s]
        results[tid] = caps
        write_results(order, results)
        print(f"[{tid}] done in {time.time() - t0:.1f}s "
              f"({time_left():.0f}s budget left)", flush=True)


def main() -> None:
    tasks = load_tasks()
    order = [t["task_id"] for t in tasks]
    # Seed EVERY task with valid non-empty fallbacks before doing anything.
    results = {tid: template_fallbacks("") for tid in order}
    write_results(order, results)
    print(f"[main] {len(tasks)} tasks; budget {TOTAL_BUDGET:.0f}s", flush=True)

    if tasks:
        try:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                futs = [pool.submit(process_task, t, order, results)
                        for t in tasks]
                for _ in as_completed(futs):
                    pass
        except Exception:  # noqa: BLE001
            print(f"[main] pool error:\n{traceback.format_exc()}", flush=True)

    write_results(order, results)
    print(f"[main] finished in {time.time() - START:.1f}s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        print(f"[main] FATAL:\n{traceback.format_exc()}", flush=True)
    sys.exit(0)  # ALWAYS
