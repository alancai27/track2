"""
Central configuration loaded from environment variables.
No secrets are hardcoded; all runtime behaviour is configurable.
"""
import os


class Config:
    # --- Fireworks AI (OpenAI-compatible) ---
    FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY", "")
    FIREWORKS_BASE_URL = os.environ.get(
        "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"
    )
    # MiniMax M3 is the multimodal vision model used for scene understanding.
    FIREWORKS_VISION_MODEL = os.environ.get(
        "FIREWORKS_VISION_MODEL", "accounts/fireworks/models/minimax-m3"
    )
    # Kimi K2P6 is used for style-specific captions (set reasoning_effort=none for clean output).
    FIREWORKS_TEXT_MODEL = os.environ.get(
        "FIREWORKS_TEXT_MODEL", "accounts/fireworks/models/kimi-k2p6"
    )
    REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "none")

    # --- Local faster-whisper transcription ---
    AUTO_TRANSCRIBE = os.environ.get("AUTO_TRANSCRIBE", "false").strip().lower() in {
        "1", "true", "yes", "y", "on"
    }
    WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "tiny")
    WHISPER_FALLBACK_MODEL = os.environ.get("WHISPER_FALLBACK_MODEL", "tiny")
    WHISPER_CACHE_DIR = os.environ.get("WHISPER_CACHE_DIR", "/app/models")
    WHISPER_MIN_WORDS = int(os.environ.get("WHISPER_MIN_WORDS", "3"))
    WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
    WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")

    # --- Keyframe extraction ---
    SCENE_THRESHOLD = float(os.environ.get("SCENE_THRESHOLD", "0.3"))
    # Dynamic frame budget: how many frames to aim for at different durations.
    FRAMES_SHORT = int(os.environ.get("FRAMES_SHORT", "3"))      # <= 30 s
    FRAMES_MEDIUM = int(os.environ.get("FRAMES_MEDIUM", "5"))    # 30-60 s
    FRAMES_LONG = int(os.environ.get("FRAMES_LONG", "6"))        # 60-120 s
    ABSOLUTE_MAX_FRAMES = int(os.environ.get("ABSOLUTE_MAX_FRAMES", "6"))
    KEYFRAME_JPEG_QUALITY = int(os.environ.get("KEYFRAME_JPEG_QUALITY", "4"))
    KEYFRAME_MAX_LONG_SIDE = int(os.environ.get("KEYFRAME_MAX_LONG_SIDE", "1024"))

    # --- Caption generation ---
    CAPTION_MAX_TOKENS = int(os.environ.get("CAPTION_MAX_TOKENS", "200"))
    BRIEF_MAX_TOKENS = int(os.environ.get("BRIEF_MAX_TOKENS", "1500"))

    # --- Orchestration ---
    PER_CLIP_TIMEOUT_SECONDS = int(os.environ.get("PER_CLIP_TIMEOUT_SECONDS", "120"))
    MAX_CONCURRENT_CLIPS = int(os.environ.get("MAX_CONCURRENT_CLIPS", "3"))
    VIDEO_DOWNLOAD_TIMEOUT = int(os.environ.get("VIDEO_DOWNLOAD_TIMEOUT", "60"))

    # --- Paths ---
    INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
    OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")

    REQUIRED_STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]

    @classmethod
    def validate(cls):
        missing = []
        if not cls.FIREWORKS_API_KEY:
            missing.append("FIREWORKS_API_KEY")
        if missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
        return True
