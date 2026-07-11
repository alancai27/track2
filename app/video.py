"""Download clip + extract N evenly-spaced downsized frames as base64 JPEGs.

Design: 3 frames, 256px wide, JPEG q4. VLMs take images not video;
Groq Llama 4 Scout caps at 5 images/request. Tiny frames cut vision
tokens hard so a 12-clip set stays under ~30k TPM. Every step has a
fallback; on total failure returns [] and the caller degrades gracefully.
"""
import base64
import os
import subprocess
import tempfile

import requests

N_FRAMES = int(os.environ.get("N_FRAMES", "3"))
FRAME_WIDTH = int(os.environ.get("FRAME_WIDTH", "256"))
DOWNLOAD_TIMEOUT = float(os.environ.get("DOWNLOAD_TIMEOUT", "60"))


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
    """Return list of base64-encoded JPEG frames (may be < N_FRAMES, or [])."""
    with tempfile.TemporaryDirectory() as tmp:
        vid = os.path.join(tmp, "clip.mp4")
        src = vid if _download(url, vid) else url  # fallback: ffmpeg reads URL directly

        dur = _duration(src)
        if dur and dur > 0:
            stamps = [dur * (i + 0.5) / N_FRAMES for i in range(N_FRAMES)]
        else:
            stamps = [1, 5, 10, 20, 40, 70][:N_FRAMES]  # blind fallback

        frames = []
        for i, ts in enumerate(stamps):
            out = os.path.join(tmp, f"f{i}.jpg")
            if _extract_at(src, ts, out):
                with open(out, "rb") as f:
                    frames.append(base64.b64encode(f.read()).decode())
        # last resort: try frame 0 if nothing worked
        if not frames:
            out = os.path.join(tmp, "f0.jpg")
            if _extract_at(src, 0.0, out):
                with open(out, "rb") as f:
                    frames.append(base64.b64encode(f.read()).decode())
        print(f"[video] extracted {len(frames)} frames from {url}", flush=True)
        return frames
