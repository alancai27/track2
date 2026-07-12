"""Perception: frames -> verified FACTUAL description via Fireworks minimax-m3.

Flow:
  1. Draft description from frames.
  2. Verification pass: same frames + draft → corrected description.
Captions are written from the verified description only.

minimax-m3 is a reasoning model — llm.chat sends reasoning_effort=none.
Fireworks VLM image cap is 30/request; we use 3–6 frames dynamically.
"""
import os
import threading

from llm import chat

CANDIDATES = [
    "accounts/fireworks/models/minimax-m3",
]

PROMPT = (
    "These are evenly spaced frames from one short video clip, in order. "
    "Write a concise FACTUAL description in 3-5 sentences covering: "
    "setting, main subjects (appearance/colors), actions and how the scene "
    "progresses, notable objects, and lighting/weather. Be specific but "
    "compact. One continuous scene, not frame-by-frame. No opinions, humor, "
    "or speculation beyond what is visible. Do not invent city/country names, "
    "landmarks, brand names, or identity labels that are not clearly visible."
)

VERIFY_PROMPT = (
    "You are verifying a draft description of this video against the frames.\n"
    "Draft description:\n{draft}\n\n"
    "Compare the draft to what is ACTUALLY visible in these frames. "
    "Rewrite a corrected FACTUAL description in 3-5 sentences that:\n"
    "- Keeps only details supported by the frames\n"
    "- Removes or fixes any hallucinations, guesses, or invented specifics "
    "(wrong objects, actions, locations, brands, identities)\n"
    "- Adds any clearly visible important detail the draft missed\n"
    "- Stays compact and concrete\n"
    "Reply with ONLY the corrected description, no preamble."
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


def _frame_messages(prompt_text: str, frames_b64: list[str]) -> list[dict]:
    # Fireworks VLM: up to 30 images; we send at most 6.
    content: list[dict] = [{"type": "text", "text": prompt_text}]
    for b64 in frames_b64[:6]:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    return [{"role": "user", "content": content}]


def _call_vision(messages: list[dict], max_tokens: int) -> str:
    """Try model ladder; cache first success. Raises if all fail."""
    global _working_model
    with _lock:
        models = ([_working_model] if _working_model else []) + \
            [m for m in _candidate_models() if m != _working_model]

    last_err = None
    for model in models:
        try:
            text = chat(model, messages, max_tokens=max_tokens, temperature=0.2)
            with _lock:
                _working_model = model
            return text
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[perception] {model} failed: {e}", flush=True)
    raise last_err or RuntimeError("no vision model available")


def _verify(frames_b64: list[str], draft: str) -> str:
    """Second vision pass: correct hallucinations against the same frames."""
    messages = _frame_messages(VERIFY_PROMPT.format(draft=draft), frames_b64)
    try:
        corrected = _call_vision(messages, max_tokens=280)
        if corrected.strip():
            print("[perception] verify ok", flush=True)
            return corrected.strip()
    except Exception as e:  # noqa: BLE001
        print(f"[perception] verify failed (keeping draft): {e}", flush=True)
    return draft


def describe(frames_b64: list[str]) -> str:
    """Return verified factual description; raises only if draft call fails."""
    draft = _call_vision(_frame_messages(PROMPT, frames_b64), max_tokens=280)
    print(f"[perception] draft ok via {_working_model}", flush=True)
    return _verify(frames_b64, draft)
