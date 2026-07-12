"""Download clip + extract N evenly-spaced downsized frames as base64 JPEGs.

Fireworks VLMs allow up to 30 images/request; we use 3–6 dynamically
(like the 92% competitor) at 256px JPEG q4. Serial workers + stagger
keep rate limits happy. Every step has a fallback; on total failure
returns [] and the caller degrades gracefully.
"""
import base64
import os
import subprocess
import tempfile

import requests

# Override forces a fixed count; otherwise duration picks 3–6.
_N_FRAMES_OVERRIDE = os.environ.get("N_FRAMES")
FRAME_WIDTH = int(os.environ.get("FRAME_WIDTH", "256"))
DOWNLOAD_TIMEOUT = float(os.environ.get("DOWNLOAD_TIMEOUT", "60"))
MAX_FRAMES = 6  # well under Fireworks' 30-image VLM cap


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _download(url: str, dest: str) -> bool:
    try:
        with requests.get(url, stream=True, timeout=(10, DOWNLOAD_TIMEOUT)) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        return os.path.getsize(dest) > 0
    except Exception as e:  # noqa: BLE001
        print(f"[video] download failed: {e}", flush=True)
        return False


def _duration(path_or_url: str) -> float | None:
    try:
        p = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                  "-of", "default=noprint_wrappers=1:nokey=1", path_or_url],
                 timeout=30)
        return float(p.stdout.strip())
    except Exception:  # noqa: BLE001
        return None


def _n_frames(dur: float | None) -> int:
    if _N_FRAMES_OVERRIDE:
        return max(1, min(int(_N_FRAMES_OVERRIDE), MAX_FRAMES))
    if not dur or dur <= 0:
        return 4
    if dur < 30:
        return 3
    if dur < 60:
        return 4
    if dur < 90:
        return 5
    return 6


def _extract_at(src: str, ts: float, out_path: str) -> bool:
    try:
        p = _run(["ffmpeg", "-y", "-ss", f"{ts:.2f}", "-i", src,
                  "-frames:v", "1", "-q:v", "4",
                  "-vf", f"scale={FRAME_WIDTH}:-2", out_path],
                 timeout=30)
        return p.returncode == 0 and os.path.exists(out_path) \
            and os.path.getsize(out_path) > 0
    except Exception:  # noqa: BLE001
        return False


def extract_frames_b64(url: str) -> list[str]:
    """Return list of base64-encoded JPEG frames (may be empty)."""
    with tempfile.TemporaryDirectory() as tmp:
        vid = os.path.join(tmp, "clip.mp4")
        src = vid if _download(url, vid) else url  # fallback: ffmpeg reads URL

        dur = _duration(src)
        n = _n_frames(dur)
        if dur and dur > 0:
            stamps = [dur * (i + 0.5) / n for i in range(n)]
        else:
            stamps = [1, 5, 10, 20, 40, 70][:n]

        frames = []
        for i, ts in enumerate(stamps):
            out = os.path.join(tmp, f"f{i}.jpg")
            if _extract_at(src, ts, out):
                with open(out, "rb") as f:
                    frames.append(base64.b64encode(f.read()).decode())
        if not frames:
            out = os.path.join(tmp, "f0.jpg")
            if _extract_at(src, 0.0, out):
                with open(out, "rb") as f:
                    frames.append(base64.b64encode(f.read()).decode())
        if dur:
            print(f"[video] extracted {len(frames)}/{n} frames "
                  f"(dur={dur:.1f}s)", flush=True)
        else:
            print(f"[video] extracted {len(frames)}/{n} frames", flush=True)
        return frames
