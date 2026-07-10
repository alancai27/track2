"""Perception: frames -> one rich FACTUAL description via a Groq VLM.

Model selection:
  1. VISION_MODEL env override, if set.
  2. Hardcoded Groq vision ladder (Llama 4 Scout first).
Each candidate is tried in order; first success wins and is cached.
"""
import os
import threading

from llm import chat

# Groq vision model IDs — Scout confirmed working with image_url blocks.
CANDIDATES = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
]

PROMPT = (
    "These are evenly spaced frames from one short video clip, in order. "
    "Write a rich, purely FACTUAL description of the clip in 4-6 sentences: "
    "the setting/location, main subjects, what they are doing, notable "
    "objects, colors/lighting/weather, and how things change or move across "
    "the frames. Describe motion and progression, not each frame separately. "
    "No opinions, no humor, no speculation beyond what is visible."
)

_lock = threading.Lock()
_working_model: str | None = None


def _candidate_models() -> list[str]:
    override = os.environ.get("VISION_MODEL")
    out = [override] if override else []
    for m in CANDIDATES:
        if m not in out:
            out.append(m)
    return out


def describe(frames_b64: list[str]) -> str:
    """Return factual description; raises only if every model fails."""
    global _working_model
    content = [{"type": "text", "text": PROMPT}]
    for b64 in frames_b64[:12]:  # hard cap, well under 30-image limit
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    messages = [{"role": "user", "content": content}]

    with _lock:
        models = ([_working_model] if _working_model else []) + \
            [m for m in _candidate_models() if m != _working_model]

    last_err = None
    for model in models:
        try:
            text = chat(model, messages, max_tokens=350, temperature=0.2)
            with _lock:
                _working_model = model
            print(f"[perception] ok via {model}", flush=True)
            return text
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[perception] {model} failed: {e}", flush=True)
    raise last_err or RuntimeError("no vision model available")
