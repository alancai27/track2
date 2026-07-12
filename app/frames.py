"""Download + scene-aware keyframe sampling (3–6 frames, endpoints included)."""
import base64
import os
import re
import subprocess
import tempfile

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import settings


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _download(url: str, dest: str) -> bool:
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
        with session.get(url, stream=True, timeout=(10, settings.DOWNLOAD_TIMEOUT)) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
        return os.path.getsize(dest) > 0
    except Exception as e:  # noqa: BLE001
        print(f"[frames] download failed: {e}", flush=True)
        return False


def _duration(src: str) -> float | None:
    try:
        p = _run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", src],
            timeout=30,
        )
        return float(p.stdout.strip())
    except Exception:  # noqa: BLE001
        pass
    try:
        p = _run(["ffmpeg", "-hide_banner", "-i", src], timeout=30)
        m = re.search(r"Duration:\s+(\d+):(\d+):(\d+\.\d+)", p.stderr or "")
        if not m:
            return None
        h, mi, s = m.groups()
        return float(h) * 3600 + float(mi) * 60 + float(s)
    except Exception:  # noqa: BLE001
        return None


def _frame_budget(dur: float | None) -> int:
    if not dur or dur <= 0:
        return min(settings.FRAMES_LE_60S, settings.FRAME_CAP)
    if dur <= 30:
        n = settings.FRAMES_LE_30S
    elif dur <= 60:
        n = settings.FRAMES_LE_60S
    else:
        n = settings.FRAMES_GT_60S
    return max(2, min(n, settings.FRAME_CAP))


def _scene_cuts(src: str) -> list[float]:
    try:
        result = _run([
            "ffmpeg", "-hide_banner", "-loglevel", "info",
            "-i", src,
            "-vf", f"select='gt(scene,{settings.SCENE_CUT_THRESHOLD})',showinfo",
            "-f", "null", "-",
        ], timeout=60)
    except Exception as e:  # noqa: BLE001
        print(f"[frames] scene detect failed: {e}", flush=True)
        return []
    stamps = []
    for line in (result.stderr or "").splitlines():
        if "pts_time:" not in line:
            continue
        m = re.search(r"pts_time:([\d.]+)", line)
        if m:
            stamps.append(float(m.group(1)))
    stamps.sort()
    out: list[float] = []
    for t in stamps:
        if not out or t - out[-1] > 0.5:
            out.append(t)
    return out


def _pick_timestamps(dur: float, scenes: list[float], n: int) -> list[float]:
    """Always keep first/last; fill with scene cuts / even spacing to reach n."""
    # Match 0.92 clamp: duration - 0.5 (not a 0.9*dur mix).
    end = max(dur - 0.5, 0.0)
    picks = [0.0] + [t for t in scenes if 0.0 < t < dur] + [dur]
    picks = sorted(set(round(t, 3) for t in picks))

    if len(picks) < n:
        need = n - len(picks)
        step = dur / (need + 1)
        extra = [round(step * i, 3) for i in range(1, need + 1)]
        picks = sorted(set(picks + extra))

    if len(picks) > n:
        first, last = picks[0], picks[-1]
        mid = picks[1:-1]
        keep = n - 2
        if keep <= 0:
            picks = [first, last]
        elif len(mid) <= keep:
            picks = [first] + mid + [last]
        else:
            idxs = [
                int(round(i * (len(mid) - 1) / (keep - 1)))
                for i in range(keep)
            ]
            picks = [first] + [mid[i] for i in sorted(set(idxs))] + [last]

    return sorted(set(
        round(max(0.0, min(t, end)), 3) for t in picks
    ))


def _extract_one(src: str, ts: float, out_path: str) -> bool:
    e = settings.LONG_EDGE_PX
    scale = f"scale={e}:{e}:force_original_aspect_ratio=decrease"
    try:
        p = _run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{ts:.3f}", "-i", src,
            "-frames:v", "1",
            "-vf", scale,
            "-q:v", str(settings.JPEG_QV),
            out_path,
        ], timeout=30)
        return p.returncode == 0 and os.path.exists(out_path) \
            and os.path.getsize(out_path) > 0
    except Exception:  # noqa: BLE001
        return False


def sample_frames_b64(url: str) -> list[str]:
    """Return JPEG keyframes as base64 (no data-URI prefix)."""
    with tempfile.TemporaryDirectory() as tmp:
        vid = os.path.join(tmp, "clip.mp4")
        src = vid if _download(url, vid) else url
        dur = _duration(src)
        n = _frame_budget(dur)
        if dur and dur > 0:
            stamps = _pick_timestamps(dur, _scene_cuts(src), n)
        else:
            stamps = [0.5, 2.0, 5.0, 10.0, 20.0, 40.0][:n]

        paths = []
        for i, ts in enumerate(stamps):
            out = os.path.join(tmp, f"f_{i:03d}.jpg")
            if _extract_one(src, ts, out):
                paths.append(out)

        if not paths and dur:
            out = os.path.join(tmp, "f_000.jpg")
            if _extract_one(src, max(dur * 0.5, 0.2), out):
                paths.append(out)

        frames = []
        for p in paths:
            with open(p, "rb") as f:
                frames.append(base64.b64encode(f.read()).decode())
        print(
            f"[frames] n={len(frames)} target={n}"
            + (f" dur={dur:.1f}s" if dur else ""),
            flush=True,
        )
        return frames
