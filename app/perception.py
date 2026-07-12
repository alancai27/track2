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
    "These are evenly spaced frames from one short video clip, in temporal "
    "order. Write a detailed, purely FACTUAL description in 5-8 sentences "
    "that a caption writer can rely on. Cover, with concrete specifics:\n"
    "- Setting/location and environment (indoor/outdoor, place type)\n"
    "- Main subjects (people/animals/objects): appearance, clothing, colors\n"
    "- Actions and motion: what is happening and how it progresses across "
    "the frames\n"
    "- Notable objects, props, text/signage if clearly readable\n"
    "- Lighting, weather, time-of-day cues, and overall visual mood "
    "(only what is visible)\n"
    "- Distinctive details that make THIS clip unique\n"
    "Describe the clip as one continuous scene, not frame-by-frame. "
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
    for b64 in frames_b64[:5]:  # Groq Scout hard-caps at 5 images/request
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
            text = chat(model, messages, max_tokens=500, temperature=0.2)
            with _lock:
                _working_model = model
            print(f"[perception] ok via {model}", flush=True)
            return text
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[perception] {model} failed: {e}", flush=True)
    raise last_err or RuntimeError("no vision model available")
