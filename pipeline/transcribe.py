"""
Local audio transcription using faster-whisper.

- Primary model size is configurable (default: small).
- Falls back to tiny if the primary model fails or returns junk.
- Returns an empty string when audio is missing, silent, or has < 3 words.
"""
import os
import threading

from faster_whisper import WhisperModel

from config import Config

# Thread-safe cache so models are loaded only once across concurrent clips.
_model_cache: dict[str, WhisperModel] = {}
_model_lock = threading.Lock()


def _load_model(model_size: str) -> WhisperModel:
    """Load a faster-whisper model from the configured cache directory."""
    with _model_lock:
        if model_size not in _model_cache:
            _model_cache[model_size] = WhisperModel(
                model_size,
                device=Config.WHISPER_DEVICE,
                compute_type=Config.WHISPER_COMPUTE_TYPE,
                download_root=Config.WHISPER_CACHE_DIR,
                local_files_only=False,
            )
        return _model_cache[model_size]


def _transcribe_with_model(audio_path: str, model_size: str) -> str:
    """Transcribe audio with a specific model size."""
    model = _load_model(model_size)
    segments, _ = model.transcribe(
        audio_path,
        beam_size=5,
        language="en",
        condition_on_previous_text=False,
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return text


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe the audio file locally.

    Returns a non-empty string only when a transcript with at least
    WHISPER_MIN_WORDS words is produced.
    """
    if not audio_path or not os.path.exists(audio_path):
        return ""

    for model_size in (Config.WHISPER_MODEL, Config.WHISPER_FALLBACK_MODEL):
        if not model_size:
            continue
        try:
            text = _transcribe_with_model(audio_path, model_size)
            word_count = len(text.split())
            print(f"  Whisper [{model_size}] transcript ({word_count} words): {text[:80]}...")
            if word_count >= Config.WHISPER_MIN_WORDS:
                return text
            # Too short; treat as no useful audio.
            return ""
        except Exception as e:
            print(f"  Transcription [{model_size}] failed (non-fatal): {e}")
            # If tiny also fails, fall through to empty transcript.

    return ""
