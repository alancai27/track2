"""Track 2 orchestrator — seed results early, always exit 0."""
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from concurrent.futures import as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import settings  # noqa: E402
from captions import seed_captions, write_styles  # noqa: E402
from frames import sample_frames_b64  # noqa: E402
from vision import ground_clip  # noqa: E402

START = time.time()
_write_lock = threading.Lock()


def _budget_left() -> float:
    return settings.WALL_BUDGET_S - (time.time() - START)


def _load_tasks() -> list[dict]:
    try:
        with open(settings.INPUT_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("tasks", [data])
        return [
            t for t in data
            if isinstance(t, dict) and t.get("task_id") and t.get("video_url")
        ]
    except Exception as e:  # noqa: BLE001
        print(f"[main] cannot read tasks: {e}", flush=True)
        return []


def _write(order: list[str], results: dict[str, dict]) -> None:
    payload = [{"task_id": tid, "captions": results[tid]} for tid in order]
    tmp = settings.OUTPUT_PATH + ".tmp"
    with _write_lock:
        try:
            os.makedirs(os.path.dirname(settings.OUTPUT_PATH) or ".", exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, settings.OUTPUT_PATH)
        except Exception as e:  # noqa: BLE001
            print(f"[main] write failed: {e}", flush=True)


def _run_clip(task: dict) -> dict[str, str]:
    tid = task["task_id"]
    if _budget_left() < 25:
        print(f"[{tid}] near wall budget — placeholders", flush=True)
        return seed_captions("budget")

    frames = sample_frames_b64(task["video_url"])
    if not frames:
        return seed_captions("no frames")

    notes = ground_clip(frames)
    print(f"[{tid}] notes: {notes[:110]}...", flush=True)
    return write_styles(notes or "A short video clip of a scene.")


def _run_clip_timed(task: dict) -> dict[str, str]:
    tid = task["task_id"]
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_run_clip, task)
        try:
            return fut.result(timeout=settings.CLIP_TIMEOUT_S)
        except FuturesTimeout:
            print(f"[{tid}] clip timeout {settings.CLIP_TIMEOUT_S}s", flush=True)
            return seed_captions("timeout")
        except Exception as e:  # noqa: BLE001
            print(f"[{tid}] clip error: {e}", flush=True)
            return seed_captions(str(e)[:80])


def main() -> None:
    try:
        settings.require_api_key()
    except Exception as e:  # noqa: BLE001
        print(f"[main] config: {e}", flush=True)

    tasks = _load_tasks()
    order = [t["task_id"] for t in tasks]
    results = {tid: seed_captions("pending") for tid in order}
    _write(order, results)
    print(
        f"[main] {len(tasks)} tasks; workers={settings.MAX_WORKERS}; "
        f"clip_timeout={settings.CLIP_TIMEOUT_S}s; budget={settings.WALL_BUDGET_S}s",
        flush=True,
    )

    if tasks:
        try:
            with ThreadPoolExecutor(max_workers=settings.MAX_WORKERS) as pool:
                futs = {
                    pool.submit(_run_clip_timed, t): t["task_id"] for t in tasks
                }
                for fut in as_completed(futs):
                    tid = futs[fut]
                    try:
                        caps = fut.result()
                    except Exception as e:  # noqa: BLE001
                        print(f"[{tid}] future error: {e}", flush=True)
                        caps = seed_captions("executor")
                    # Guarantee every style non-empty.
                    base = seed_captions("empty")
                    for s in settings.STYLES:
                        if not str(caps.get(s, "")).strip():
                            caps[s] = base[s]
                    results[tid] = caps
                    _write(order, results)
                    print(f"[{tid}] done ({_budget_left():.0f}s left)", flush=True)
        except Exception:  # noqa: BLE001
            print(f"[main] pool crash:\n{traceback.format_exc()}", flush=True)

    _write(order, results)
    print(f"[main] finished in {time.time() - START:.1f}s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        print(f"[main] FATAL:\n{traceback.format_exc()}", flush=True)
    sys.exit(0)
