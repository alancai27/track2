"""Runtime knobs for the Track 2 caption agent (env-overridable)."""
import os


def _bool(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in {
        "1", "true", "yes", "y", "on",
    }


FIREWORKS_API_KEY = (os.environ.get("FIREWORKS_API_KEY") or "").strip()
FIREWORKS_BASE_URL = os.environ.get(
    "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"
)

VISION_MODEL = os.environ.get(
    "VISION_MODEL", "accounts/fireworks/models/minimax-m3"
)
CAPTION_MODEL = os.environ.get(
    "CAPTION_MODEL", "accounts/fireworks/models/kimi-k2p6"
)
REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "none")

# Frame budget mirrors the 0.92 design: short/medium/long dynamic counts.
FRAMES_LE_30S = int(os.environ.get("FRAMES_LE_30S", "3"))
FRAMES_LE_60S = int(os.environ.get("FRAMES_LE_60S", "5"))
FRAMES_GT_60S = int(os.environ.get("FRAMES_GT_60S", "6"))
FRAME_CAP = int(os.environ.get("FRAME_CAP", "6"))
SCENE_CUT_THRESHOLD = float(os.environ.get("SCENE_CUT_THRESHOLD", "0.3"))
JPEG_QV = int(os.environ.get("JPEG_QV", "4"))
LONG_EDGE_PX = int(os.environ.get("LONG_EDGE_PX", "1024"))

BRIEF_TOKENS = int(os.environ.get("BRIEF_TOKENS", "1500"))
CAPTION_TOKENS = int(os.environ.get("CAPTION_TOKENS", "200"))
CALL_TIMEOUT = float(os.environ.get("CALL_TIMEOUT", "60"))

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")
CLIP_TIMEOUT_S = float(os.environ.get("CLIP_TIMEOUT_S", "120"))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "3"))
DOWNLOAD_TIMEOUT = float(os.environ.get("DOWNLOAD_TIMEOUT", "60"))
WALL_BUDGET_S = float(os.environ.get("WALL_BUDGET_S", "560"))

STYLES = ("formal", "sarcastic", "humorous_tech", "humorous_non_tech")


def require_api_key() -> None:
    if not FIREWORKS_API_KEY:
        raise RuntimeError("FIREWORKS_API_KEY is missing")
