"""Video download + temporal-midpoint keyframes (+ optional scene extras).

Primary (accuracy-safe): timestamps at (i+0.5)/n * duration — avoids
fade/title-card first/last frames.

Secondary: FFmpeg scene cuts (gt(scene,THRESHOLD)), 0.5s-deduped; add up
to 2 cuts only if meaningfully away from existing mids.

Counts: 4 short / 6 medium / 8 long. Long-edge ~768px, JPEG q:v 4.
Download uses retries. No audio extraction (Whisper disabled).
"""
import base64
import os
import re
import subprocess
import tempfile

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

FRAME_LONG_EDGE = int(os.environ.get("FRAME_LONG_EDGE", "768"))
DOWNLOAD_TIMEOUT = float(os.environ.get("DOWNLOAD_TIMEOUT", "60"))
SCENE_THRESHOLD = float(os.environ.get("SCENE_THRESHOLD", "0.3"))
SCENE_DEDUPE_GAP = 0.5
MAX_SCENE_EXTRAS = 2
JPEG_QV = int(os.environ.get("JPEG_QV", "4"))


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _download(url: str, dest: str) -> bool:
    """Download with retries (competitor-style HTTPAdapter)."""
    try:
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.8,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=1, pool_maxsize=1)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        with session.get(url, stream=True, timeout=(10, DOWNLOAD_TIMEOUT)) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
        return os.path.getsize(dest) > 0
    except Exception as e:  # noqa: BLE001
        print(f"[video] download failed: {e}", flush=True)
        return False


def _duration(path_or_url: str) -> float | None:
    """Prefer ffprobe; fall back to parsing ffmpeg banner Duration."""
    try:
        p = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                  "-of", "default=noprint_wrappers=1:nokey=1", path_or_url],
                 timeout=30)
        return float(p.stdout.strip())
    except Exception:  # noqa: BLE001
        pass
    try:
        p = _run(["ffmpeg", "-hide_banner", "-i", path_or_url], timeout=30)
        m = re.search(r"Duration:\s+(\d+):(\d+):(\d+\.\d+)", p.stderr or "")
        if not m:
            return None
        h, mi, s = m.groups()
        return float(h) * 3600 + float(mi) * 60 + float(s)
    except Exception:  # noqa: BLE001
        return None


def _n_frames(dur: float | None) -> int:
    override = os.environ.get("N_FRAMES")
    if override:
        return max(2, min(int(override), 8))
    if not dur or dur <= 0:
        return 6
    if dur <= 30:
        return 4
    if dur <= 60:
        return 6
    return 8


def _scale_vf() -> str:
    # Fit inside long_edge x long_edge, keep aspect (competitor scale style).
    e = FRAME_LONG_EDGE
    return f"scale={e}:{e}:force_original_aspect_ratio=decrease"


def _extract_at(src: str, ts: float, out_path: str) -> bool:
    try:
        p = _run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{ts:.3f}", "-i", src,
            "-frames:v", "1",
            "-vf", _scale_vf(),
            "-q:v", str(JPEG_QV),
            out_path,
        ], timeout=30)
        return p.returncode == 0 and os.path.exists(out_path) \
            and os.path.getsize(out_path) > 0
    except Exception:  # noqa: BLE001
        return False


def _read_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _midpoint_stamps(dur: float, n: int) -> list[float]:
    # (i+0.5)/n * dur — never first/last endpoints.
    # Clamp away from EOF (reported duration can overshoot decodeable stream).
    end = max(dur - 0.5, dur * 0.9)
    stamps = [dur * (i + 0.5) / n for i in range(n)]
    return [max(0.05, min(ts, end)) for ts in stamps]


def _detect_scene_changes(src: str) -> list[float]:
    """FFmpeg scene detect + 0.5s dedupe. Empty list on failure."""
    try:
        result = _run([
            "ffmpeg", "-hide_banner", "-loglevel", "info",
            "-i", src,
            "-vf", f"select='gt(scene,{SCENE_THRESHOLD})',showinfo",
            "-f", "null", "-",
        ], timeout=60)
    except Exception as e:  # noqa: BLE001
        print(f"[video] scene detection failed: {e}", flush=True)
        return []

    timestamps = []
    for line in (result.stderr or "").splitlines():
        if "pts_time:" not in line:
            continue
        m = re.search(r"pts_time:([\d.]+)", line)
        if m:
            timestamps.append(float(m.group(1)))
    timestamps.sort()
    deduped = []
    for ts in timestamps:
        if not deduped or ts - deduped[-1] > SCENE_DEDUPE_GAP:
            deduped.append(ts)
    return deduped


def _pick_scene_extras(scene_changes: list[float], mid_stamps: list[float],
                       dur: float) -> list[float]:
    """Up to MAX_SCENE_EXTRAS cuts far from mids and away from fade ends."""
    if not scene_changes or not mid_stamps:
        return []
    min_gap = max(0.4, dur / (len(mid_stamps) * 4))
    end = max(dur - 0.5, dur * 0.9)
    candidates = []
    for t in scene_changes:
        if t <= 0.2 or t >= end - 0.05:
            continue  # skip open/close fades
        if all(abs(t - m) >= min_gap for m in mid_stamps):
            candidates.append(t)
    if not candidates:
        return []
    # Prefer cuts spread across the clip: pick by distance from nearest mid.
    scored = sorted(
        candidates,
        key=lambda t: min(abs(t - m) for m in mid_stamps),
        reverse=True,
    )
    picked = []
    for t in scored:
        if all(abs(t - p) >= SCENE_DEDUPE_GAP for p in picked):
            picked.append(t)
        if len(picked) >= MAX_SCENE_EXTRAS:
            break
    return sorted(picked)


def extract_frames_b64(url: str) -> list[str]:
    """Return midpoint JPEG frames (+ optional scene extras) as base64."""
    with tempfile.TemporaryDirectory() as tmp:
        vid = os.path.join(tmp, "clip.mp4")
        src = vid if _download(url, vid) else url

        dur = _duration(src)
        n = _n_frames(dur)
        if dur and dur > 0:
            stamps = _midpoint_stamps(dur, n)
        else:
            stamps = [2.0, 6.0, 12.0, 20.0, 30.0, 45.0][:n]

        # Optional scene extras (not first/last-driven).
        scene_extras: list[float] = []
        if dur and dur > 0:
            scenes = _detect_scene_changes(src)
            scene_extras = _pick_scene_extras(scenes, stamps, dur)
            if scene_extras:
                print(f"[video] scene cuts={len(scenes)} extras="
                      f"{[round(t, 2) for t in scene_extras]}", flush=True)

        all_stamps = sorted(set(round(t, 3) for t in stamps + scene_extras))

        paths = []
        for i, ts in enumerate(all_stamps):
            out = os.path.join(tmp, f"frame_{i:03d}.jpg")
            if _extract_at(src, ts, out):
                paths.append(out)

        if not paths and dur:
            # Last resort: single true midpoint (still not endpoint).
            out = os.path.join(tmp, "frame_000.jpg")
            if _extract_at(src, max(dur * 0.5, 0.2), out):
                paths.append(out)

        frames = [_read_b64(p) for p in paths]
        dur_s = f" dur={dur:.1f}s" if dur else ""
        print(f"[video] extracted {len(frames)} frames "
              f"(mid={n}, +scene={len(scene_extras)}){dur_s}", flush=True)
        return frames
