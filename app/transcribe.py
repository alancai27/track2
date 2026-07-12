"""Local audio transcription via faster-whisper (tiny / int8 / CPU).

HARD GUARD: every failure path returns "" — transcription must never block
or crash a clip. Toggle with AUTO_TRANSCRIBE (default true).
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout

AUTO_TRANSCRIBE = os.environ.get("AUTO_TRANSCRIBE", "true").lower() in (
    "1", "true", "yes", "on",
)
TRANSCRIBE_TIMEOUT = float(os.environ.get("TRANSCRIBE_TIMEOUT", "25"))
WHISPER_MODEL_PATH = os.environ.get(
    "WHISPER_MODEL_PATH", "/models/whisper-tiny"
)
# HuggingFace id used when local path is missing (local/dev only).
WHISPER_MODEL_ID = os.environ.get("WHISPER_MODEL_ID", "tiny")

_model = None
_model_lock = threading.Lock()
_model_failed = False


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _extract_wav(video_path: str, wav_path: str) -> bool:
    try:
        p = _run([
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-ac", "1", "-ar", "16000",
            "-f", "wav", wav_path,
        ], timeout=20)
        return p.returncode == 0 and os.path.exists(wav_path) \
            and os.path.getsize(wav_path) > 44
    except Exception as e:  # noqa: BLE001
        print(f"[asr] wav extract failed: {e}", flush=True)
        return False


def _load_model():
    global _model, _model_failed
    with _model_lock:
        if _model is not None:
            return _model
        if _model_failed:
            return None
        try:
            from faster_whisper import WhisperModel  # noqa: WPS
            path = WHISPER_MODEL_PATH
            if os.path.isdir(path):
                _model = WhisperModel(
                    path, device="cpu", compute_type="int8"
                )
            else:
                # Dev fallback: download tiny on first use.
                _model = WhisperModel(
                    WHISPER_MODEL_ID, device="cpu", compute_type="int8"
                )
            print(f"[asr] whisper ready ({path or WHISPER_MODEL_ID})",
                  flush=True)
            return _model
        except Exception as e:  # noqa: BLE001
            _model_failed = True
            print(f"[asr] model load failed (disabled): {e}", flush=True)
            return None


def _transcribe_sync(video_path: str) -> str:
    if not video_path or not os.path.isfile(video_path):
        return ""
    model = _load_model()
    if model is None:
        return ""
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "audio.wav")
        if not _extract_wav(video_path, wav):
            return ""
        segments, _info = model.transcribe(
            wav,
            beam_size=1,
            vad_filter=True,
            language=None,  # auto-detect; ignore if empty
        )
        parts = []
        for seg in segments:
            t = (seg.text or "").strip()
            if t:
                parts.append(t)
        text = " ".join(parts).strip()
        if text:
            print(f"[asr] transcript ({len(text)} chars): {text[:100]}...",
                  flush=True)
        else:
            print("[asr] empty transcript", flush=True)
        return text


def transcribe_audio(video_path: str) -> str:
    """Return transcript text, or "" on any failure / timeout / disable."""
    if not AUTO_TRANSCRIBE:
        return ""
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_transcribe_sync, video_path)
            return fut.result(timeout=TRANSCRIBE_TIMEOUT) or ""
    except FuturesTimeout:
        print(f"[asr] timed out after {TRANSCRIBE_TIMEOUT:.0f}s — skipping",
              flush=True)
        return ""
    except Exception as e:  # noqa: BLE001
        print(f"[asr] failed (skipping): {e}", flush=True)
        return ""
