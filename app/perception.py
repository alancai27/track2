"""Perception: frames (+ optional audio transcript) -> verified description.

Flow:
  1. Draft description from keyframes (+ transcript if present).
  2. Verification pass against the same frames.
Uses Fireworks minimax-m3 with reasoning_effort=none (via llm.chat).
"""
import os
import threading

from llm import chat

CANDIDATES = [
    "accounts/fireworks/models/minimax-m3",
]

PROMPT = (
    "These are keyframes from one short video clip, in temporal order. "
    "Write a concise FACTUAL description in 3-5 sentences covering: "
    "setting, main subjects (appearance/colors), actions and how the scene "
    "progresses, notable objects, and lighting/weather. Be specific but "
    "compact. One continuous scene, not frame-by-frame. No opinions, humor, "
    "or speculation beyond what is visible. Do not invent city/country names, "
    "landmarks, brand names, or identity labels that are not clearly visible."
)

TRANSCRIPT_ADDENDUM = (
    "\n\nAudio transcript from the clip (may be incomplete or noisy):\n"
    "\"{transcript}\"\n"
    "If the transcript contains clearly relevant speech or sounds described "
    "as words, weave that into the factual description. Do NOT invent "
    "dialogue that is not in the transcript. If the transcript is empty or "
    "gibberish, ignore it."
)

VERIFY_PROMPT = (
    "You are verifying a draft description of this video against the frames.\n"
    "Draft description:\n{draft}\n\n"
    "{transcript_block}"
    "Compare the draft to what is ACTUALLY visible in these frames"
    "{audio_note}. "
    "Rewrite a corrected FACTUAL description in 3-5 sentences that:\n"
    "- Keeps only details supported by the frames"
    "{audio_keep}\n"
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
    # Cap at 8 to match MAX_FRAMES; Fireworks allows up to 30.
    content: list[dict] = [{"type": "text", "text": prompt_text}]
    for b64 in frames_b64[:8]:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    return [{"role": "user", "content": content}]


def _call_vision(messages: list[dict], max_tokens: int) -> str:
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


def _draft_prompt(transcript: str) -> str:
    prompt = PROMPT
    t = (transcript or "").strip()
    if t:
        prompt += TRANSCRIPT_ADDENDUM.format(transcript=t[:1500])
    return prompt


def _verify(frames_b64: list[str], draft: str, transcript: str) -> str:
    t = (transcript or "").strip()
    if t:
        t_block = (
            f"Audio transcript (for speech-only facts):\n\"{t[:1500]}\"\n\n"
        )
        audio_note = " and spoken content supported by the transcript"
        audio_keep = " or clearly supported by the transcript"
    else:
        t_block = ""
        audio_note = ""
        audio_keep = ""
    messages = _frame_messages(
        VERIFY_PROMPT.format(
            draft=draft,
            transcript_block=t_block,
            audio_note=audio_note,
            audio_keep=audio_keep,
        ),
        frames_b64,
    )
    try:
        corrected = _call_vision(messages, max_tokens=280)
        if corrected.strip():
            print("[perception] verify ok", flush=True)
            return corrected.strip()
    except Exception as e:  # noqa: BLE001
        print(f"[perception] verify failed (keeping draft): {e}", flush=True)
    return draft


def describe(frames_b64: list[str], transcript: str = "") -> str:
    """Return verified factual description; raises only if draft call fails."""
    draft = _call_vision(
        _frame_messages(_draft_prompt(transcript), frames_b64),
        max_tokens=280,
    )
    print(f"[perception] draft ok via {_working_model}", flush=True)
    return _verify(frames_b64, draft, transcript)
