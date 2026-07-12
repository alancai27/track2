"""
Video download, scene-aware keyframe extraction, and audio extraction.

Keyframes:
- Detect shot boundaries with FFmpeg select=gt(scene,THRESHOLD).
- Always include first and last frames.
- Fill in extra evenly-spaced frames if there are too few shots.
- Sample evenly across detected shots if there are too many.
- Resize to max 1024 px on the long side, JPEG quality ~85 (ffmpeg -q:v 4).
"""
import os
import re
import shutil
import subprocess
import tempfile

import requests

from config import Config


def _get_ffmpeg_binary() -> str:
    """Return the ffmpeg executable, falling back to imageio-ffmpeg if needed."""
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        return get_ffmpeg_exe()
    except Exception:
        raise RuntimeError(
            "ffmpeg not found in PATH and imageio_ffmpeg is not installed."
        )


FFMPEG = _get_ffmpeg_binary()


def download_video(video_url: str, dest_dir: str) -> str:
    """Download the video to a local temporary path with retries."""
    local_path = os.path.join(dest_dir, "input_video.mp4")

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        max_retries=3,
        pool_connections=1,
        pool_maxsize=1,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    with session:
        response = session.get(
            video_url,
            stream=True,
            timeout=Config.VIDEO_DOWNLOAD_TIMEOUT,
        )
        response.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return local_path


def get_video_duration(video_path: str) -> float:
    """Return video duration in seconds using ffmpeg."""
    cmd = [
        FFMPEG, "-hide_banner",
        "-i", video_path,
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=30
    )
    match = re.search(r"Duration:\s+(\d+):(\d+):(\d+\.\d+)", result.stderr)
    if not match:
        raise RuntimeError("Could not determine video duration")
    hours, minutes, seconds = match.groups()
    return float(hours) * 3600 + float(minutes) * 60 + float(seconds)


def compute_dynamic_frame_count(duration: float, max_frames: int | None = None) -> int:
    """Choose a frame budget based on clip length. Shorter clips need fewer frames."""
    cap = max_frames if max_frames is not None else Config.ABSOLUTE_MAX_FRAMES
    if duration <= 30:
        return min(Config.FRAMES_SHORT, cap)
    if duration <= 60:
        return min(Config.FRAMES_MEDIUM, cap)
    return min(Config.FRAMES_LONG, cap)


def detect_scene_changes(video_path: str, threshold: float) -> list[float]:
    """
    Run FFmpeg scene detection and return a sorted list of shot-boundary
    timestamps (seconds). Falls back to empty list on any failure.
    """
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "info",
        "-i", video_path,
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60
        )
    except Exception as e:
        print(f"  Scene detection failed: {e}")
        return []

    timestamps = []
    for line in result.stderr.splitlines():
        if "pts_time:" in line:
            match = re.search(r"pts_time:([\d.]+)", line)
            if match:
                timestamps.append(float(match.group(1)))

    timestamps.sort()
    # Remove near-duplicates caused by consecutive scene frames.
    deduped = []
    for ts in timestamps:
        if not deduped or ts - deduped[-1] > 0.5:
            deduped.append(ts)
    return deduped


def build_keyframe_timestamps(
    duration: float,
    scene_changes: list[float],
    min_frames: int,
    max_frames: int,
) -> list[float]:
    """
    Build a list of representative timestamps.

    Always includes the first (0.0) and last (duration) frames, adds scene
    changes, and either fills or sub-samples to stay inside [min_frames,
    max_frames].
    """
    # Start with boundaries and scene changes.
    candidates = [0.0] + [ts for ts in scene_changes if 0.0 < ts < duration] + [duration]
    candidates = sorted(set(round(ts, 3) for ts in candidates))

    if len(candidates) < min_frames:
        # Fill with evenly spaced frames inside the clip.
        needed = min_frames - len(candidates)
        step = duration / (needed + 1)
        extra = [round(step * i, 3) for i in range(1, needed + 1)]
        candidates = sorted(set(candidates + extra))

    if len(candidates) > max_frames:
        # Sample evenly across candidate timestamps while keeping first/last.
        first, last = candidates[0], candidates[-1]
        middle = candidates[1:-1]
        keep_count = max_frames - 2
        if keep_count <= 0:
            return [first, last]
        if len(middle) <= keep_count:
            sampled = middle
        else:
            indices = [
                int(round(i * (len(middle) - 1) / (keep_count - 1)))
                for i in range(keep_count)
            ]
            sampled = [middle[i] for i in sorted(set(indices))]
        candidates = [first] + sampled + [last]

    # Clamp to valid range so the last frame never sits too close to EOF.
    # Use duration - 0.5 to leave room for containers whose reported duration
    # is slightly longer than the actual decodeable stream.
    candidates = [max(0.0, min(ts, duration - 0.5)) for ts in candidates]
    return sorted(set(round(ts, 3) for ts in candidates))


def extract_keyframes(
    video_path: str, output_dir: str, timestamps: list[float]
) -> list[str]:
    """Extract frames at the chosen timestamps and resize them."""
    frame_paths = []
    max_side = Config.KEYFRAME_MAX_LONG_SIDE
    quality = Config.KEYFRAME_JPEG_QUALITY
    # Scale keeps aspect ratio and fits within max_side x max_side.
    scale_filter = f"scale={max_side}:{max_side}:force_original_aspect_ratio=decrease"

    for idx, ts in enumerate(timestamps):
        out_path = os.path.join(output_dir, f"frame_{idx:03d}.jpg")
        cmd = [
            FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(ts),
            "-i", video_path,
            "-frames:v", "1",
            "-vf", scale_filter,
            "-q:v", str(quality),
            out_path,
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=30)
            frame_paths.append(out_path)
        except subprocess.CalledProcessError as e:
            print(f"  Warning: failed to extract frame at {ts}s: {e}")

    if not frame_paths:
        raise RuntimeError("Could not extract any keyframes from the video.")
    return frame_paths


def extract_audio(video_path: str, output_dir: str) -> str:
    """
    Extract 16kHz mono WAV audio for transcription.
    Returns an empty string if the clip has no audio track.
    """
    audio_path = os.path.join(output_dir, "audio.wav")
    cmd = [
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0 or not os.path.exists(audio_path):
        return ""
    return audio_path


def process_video(video_url: str) -> dict:
    """
    Download a clip and produce keyframes + audio.

    Returns:
        {"frames": [paths], "audio_path": str, "temp_dir": str}
    """
    temp_dir = tempfile.mkdtemp(prefix="clip_")

    video_path = download_video(video_url, temp_dir)
    duration = get_video_duration(video_path)

    target_frames = compute_dynamic_frame_count(duration)
    scene_changes = detect_scene_changes(video_path, Config.SCENE_THRESHOLD)
    timestamps = build_keyframe_timestamps(
        duration,
        scene_changes,
        min_frames=target_frames,
        max_frames=target_frames,
    )
    frames = extract_keyframes(video_path, temp_dir, timestamps)
    audio_path = extract_audio(video_path, temp_dir)

    print(f"  Extracted {len(frames)} keyframes from {len(scene_changes)} scene changes.")
    return {
        "frames": frames,
        "audio_path": audio_path,
        "temp_dir": temp_dir,
    }
