"""
Main orchestration for the AMD Track 2 Video Captioning Agent.

Flow per clip:
  download -> scene-aware keyframes + optional audio -> optional local whisper
  -> MiniMax M3 structured brief + verification -> Kimi K2P6 style captions
  -> results.json

Clips are processed concurrently (max 3) with a hard per-clip timeout.
Every task_id and every requested style is guaranteed an entry in the output.
"""
import json
import os
import shutil
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from config import Config
from pipeline.extract import process_video
from pipeline.transcribe import transcribe_audio
from pipeline.analyze import analyze_video
from pipeline.caption import generate_captions


def load_tasks(input_path: str) -> list[dict]:
    """Load /input/tasks.json."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _placeholder_captions(reason: str) -> dict:
    """Return a safe placeholder for every required style."""
    msg = f"Unable to process this video clip ({reason})."
    return {
        "formal": msg,
        "sarcastic": msg,
        "humorous_tech": msg,
        "humorous_non_tech": msg,
    }


def process_single_task(task: dict) -> dict:
    """
    Process one clip end-to-end.

    Never raises: failures are converted to placeholder captions so the
    task_id is always present in results.json.
    """
    task_id = task.get("task_id", "unknown")
    video_url = task.get("video_url", "")

    temp_dir = None
    try:
        print(f"[{task_id}] Processing: {video_url}")

        result = process_video(video_url)
        temp_dir = result["temp_dir"]

        transcript = ""
        if Config.AUTO_TRANSCRIBE and result["audio_path"]:
            transcript = transcribe_audio(result["audio_path"])
            if transcript:
                print(f"[{task_id}] Transcript: {transcript[:100]}...")
            else:
                print(f"[{task_id}] No usable transcript.")

        description = analyze_video(result["frames"], transcript)
        print(f"[{task_id}] Verified description: {description[:100]}...")

        captions = generate_captions(description)

        output_captions = {}
        for style in Config.REQUIRED_STYLES:
            output_captions[style] = captions.get(style, "")

        # Ensure any style missing from the model output gets a placeholder.
        for style in Config.REQUIRED_STYLES:
            if not output_captions.get(style):
                output_captions[style] = f"Unable to generate {style} caption."

        return {"task_id": task_id, "captions": output_captions}

    except Exception as e:
        print(f"[{task_id}] FAILED: {e}")
        traceback.print_exc()
        return {"task_id": task_id, "captions": _placeholder_captions(str(e)[:120])}

    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def process_task_with_timeout(task: dict, timeout_seconds: int) -> dict:
    """Run process_single_task with a hard timeout."""
    task_id = task.get("task_id", "unknown")
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(process_single_task, task)
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeoutError:
            print(f"[{task_id}] TIMEOUT after {timeout_seconds}s")
            return {"task_id": task_id, "captions": _placeholder_captions("timeout")}


def main():
    Config.validate()

    input_path = Config.INPUT_PATH
    output_path = Config.OUTPUT_PATH

    print(f"Reading tasks from: {input_path}")
    tasks = load_tasks(input_path)
    print(f"Found {len(tasks)} task(s).\n")

    results = [None] * len(tasks)

    with ThreadPoolExecutor(max_workers=Config.MAX_CONCURRENT_CLIPS) as executor:
        future_to_index = {
            executor.submit(
                process_task_with_timeout, task, Config.PER_CLIP_TIMEOUT_SECONDS
            ): i
            for i, task in enumerate(tasks)
        }
        for future in future_to_index:
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as e:
                task_id = tasks[index].get("task_id", "unknown")
                print(f"[{task_id}] Unexpected executor error: {e}")
                results[index] = {
                    "task_id": task_id,
                    "captions": _placeholder_captions("executor error"),
                }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults written to: {output_path}")
    print(f"Total tasks processed: {len(results)}")


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
