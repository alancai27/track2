"""Download clip + extract scene-change keyframes as base64 JPEGs.

Primary: ffmpeg scene detection (select='gt(scene,0.3)') for visually
distinct frames. Fallback: evenly spaced if scene detect yields <3.
Cap at 8 frames (Fireworks VLM allows 30). Frames scaled to FRAME_WIDTH.
"""
import base64
import glob
import os
import subprocess
import tempfile
from contextlib import contextmanager

import requests

FRAME_WIDTH = int(os.environ.get("FRAME_WIDTH", "256"))
DOWNLOAD_TIMEOUT = float(os.environ.get("DOWNLOAD_TIMEOUT", "60"))
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "8"))
SCENE_THRESHOLD = float(os.environ.get("SCENE_THRESHOLD", "0.3"))
MIN_SCENE_FRAMES = 3


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


def _read_jpegs(paths: list[str]) -> list[str]:
    out = []
    for path in paths:
        try:
            with open(path, "rb") as f:
                data = f.read()
            if data:
                out.append(base64.b64encode(data).decode())
        except OSError:
            continue
    return out


def _subsample(paths: list[str], n: int) -> list[str]:
    if len(paths) <= n:
        return paths
    if n == 1:
        return [paths[len(paths) // 2]]
    idxs = [round(i * (len(paths) - 1) / (n - 1)) for i in range(n)]
    return [paths[i] for i in idxs]


def _scene_frame_paths(src: str, out_dir: str) -> list[str]:
    """Extract scene-change frames; return sorted JPEG paths."""
    pattern = os.path.join(out_dir, "scene_%04d.jpg")
    try:
        _run([
            "ffmpeg", "-y", "-i", src,
            "-vf", f"select='gt(scene,{SCENE_THRESHOLD})',"
                   f"scale={FRAME_WIDTH}:-2",
            "-vsync", "vfr",
            "-q:v", "4",
            pattern,
        ], timeout=90)
    except Exception as e:  # noqa: BLE001
        print(f"[video] scene detect failed: {e}", flush=True)
        return []
    paths = sorted(glob.glob(os.path.join(out_dir, "scene_*.jpg")))
    return [p for p in paths if os.path.getsize(p) > 0]


def _even_frame_paths(src: str, out_dir: str, n: int,
                      dur: float | None) -> list[str]:
    if dur and dur > 0:
        stamps = [dur * (i + 0.5) / n for i in range(n)]
    else:
        stamps = [1, 4, 8, 12, 18, 25, 35, 45][:n]
    paths = []
    for i, ts in enumerate(stamps):
        out = os.path.join(out_dir, f"even_{i:04d}.jpg")
        if _extract_at(src, ts, out):
            paths.append(out)
    if not paths:
        out = os.path.join(out_dir, "even_0000.jpg")
        if _extract_at(src, 0.0, out):
            paths.append(out)
    return paths


def _keyframe_paths(src: str, work: str) -> tuple[list[str], str]:
    """Return (jpeg paths, method label). Prefer scene; else even spacing."""
    scene_dir = os.path.join(work, "scene")
    os.makedirs(scene_dir, exist_ok=True)
    scene = _scene_frame_paths(src, scene_dir)
    if len(scene) >= MIN_SCENE_FRAMES:
        picked = _subsample(scene, MAX_FRAMES)
        return picked, f"scene({len(scene)}->{len(picked)})"

    even_dir = os.path.join(work, "even")
    os.makedirs(even_dir, exist_ok=True)
    dur = _duration(src)
    n = MAX_FRAMES
    even = _even_frame_paths(src, even_dir, n, dur)
    return even, f"even({len(even)})"


@contextmanager
def open_clip(url: str):
    """Download once; yield (frames_b64, local_video_path)."""
    with tempfile.TemporaryDirectory() as tmp:
        vid = os.path.join(tmp, "clip.mp4")
        src = vid if _download(url, vid) else url
        if src == url:
            # Still try to materialize a local copy for ASR when stream works.
            try:
                _run(["ffmpeg", "-y", "-i", url, "-c", "copy",
                      "-t", "180", vid], timeout=90)
                if os.path.exists(vid) and os.path.getsize(vid) > 0:
                    src = vid
            except Exception:  # noqa: BLE001
                pass

        work = os.path.join(tmp, "frames")
        os.makedirs(work, exist_ok=True)
        paths, method = _keyframe_paths(src, work)
        frames = _read_jpegs(paths)
        dur = _duration(src)
        dur_s = f" dur={dur:.1f}s" if dur else ""
        print(f"[video] extracted {len(frames)} frames via {method}{dur_s}",
              flush=True)
        yield frames, (src if os.path.isfile(src) else "")


def extract_frames_b64(url: str) -> list[str]:
    """Back-compat: keyframes only (no shared download for ASR)."""
    with open_clip(url) as (frames, _vid):
        return list(frames)
