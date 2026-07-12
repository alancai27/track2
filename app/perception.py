"""Two-step visual grounding (competitor prompts, Fireworks minimax-m3):

1. MiniMax M3 writes a dense structured JSON brief from sampled frames.
2. MiniMax M3 checks that description against the same frames and removes
   anything unsupported or too generic.

The final verified description is what the caption model uses.
"""
import json
import os
import re
import threading

from llm import chat

CANDIDATES = [
    "accounts/fireworks/models/minimax-m3",
]

BRIEF_MAX_TOKENS = int(os.environ.get("BRIEF_MAX_TOKENS", "600"))
BRIEF_FIELDS = (
    "setting",
    "subjects",
    "actions",
    "objects",
    "mood",
    "sounds",
    "dialogue_summary",
    "notable_details",
    "overall_summary",
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
    content: list[dict] = [{"type": "text", "text": prompt_text}]
    for b64 in frames_b64[:9]:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    return [{"role": "user", "content": content}]


def _call_vision(frames_b64: list[str], prompt: str, max_tokens: int) -> str:
    """Send frames + prompt to the vision model; cache first working model."""
    global _working_model
    messages = _frame_messages(prompt, frames_b64)
    with _lock:
        models = ([_working_model] if _working_model else []) + \
            [m for m in _candidate_models() if m != _working_model]

    last_err = None
    for model in models:
        try:
            text = chat(
                model, messages, max_tokens=max_tokens, temperature=0.1
            )
            with _lock:
                _working_model = model
            return text
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[perception] {model} failed: {e}", flush=True)
    raise last_err or RuntimeError("no vision model available")


def _strip_to_json(text: str) -> dict:
    """Extract a JSON object from model output, tolerating markdown fences."""
    if not text:
        raise ValueError("Empty response from model.")

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        obj = json.loads(match.group(0))
        if isinstance(obj, dict):
            return obj

    raise ValueError(f"Could not extract JSON from response: {text[:200]}")


def _build_brief_prompt(transcript: str) -> str:
    # Exact competitor prompt (transcript section empty when ASR is off).
    transcript_section = ""
    if transcript:
        transcript_section = f"""
The clip's audio was transcribed as follows (may include dialogue, narration, or ambient sounds):
"{transcript}"
"""

    return f"""You are analyzing a short video clip using the provided keyframes in chronological order.{transcript_section}

Produce a structured JSON brief that captures ONLY what is actually visible in the frames. Use exactly these fields:
- setting: where and when the video takes place
- subjects: the main people, animals, or entities visible
- actions: what the subjects are doing
- objects: notable objects, props, or environmental details
- mood: atmosphere or emotional tone
- sounds: notable non-speech sounds (or "none" if none)
- dialogue_summary: summary of spoken dialogue or narration, if any
- notable_details: any other distinctive visual details
- overall_summary: a concise 2-3 sentence summary

Rules:
- Describe ONLY what you can see in the provided frames.
- Do NOT invent animals, vehicles, objects, landmarks, locations, or people that are not clearly visible.
- If something is unclear or partially visible, describe it generically or omit it.
- Do not include explanations, markdown, or reasoning outside the JSON.

Output ONLY valid JSON matching this structure exactly:

{{
  "setting": "...",
  "subjects": "...",
  "actions": "...",
  "objects": "...",
  "mood": "...",
  "sounds": "...",
  "dialogue_summary": "...",
  "notable_details": "...",
  "overall_summary": "..."
}}
"""


def _normalize_brief(parsed: dict) -> dict[str, str]:
    out = {}
    for key in BRIEF_FIELDS:
        val = parsed.get(key, "")
        if val is None:
            val = ""
        out[key] = str(val).strip()
    if not out["overall_summary"] and not out["subjects"]:
        raise ValueError("brief missing overall_summary and subjects")
    return out


def _brief_to_paragraph(brief: dict[str, str]) -> str:
    """Flatten a structured brief into a dense factual paragraph."""
    parts = [
        brief["overall_summary"],
        f"Setting: {brief['setting']}",
        f"Subjects: {brief['subjects']}",
        f"Actions: {brief['actions']}",
        f"Objects/details: {brief['objects']}",
        f"Mood: {brief['mood']}",
    ]
    dialogue = brief.get("dialogue_summary", "")
    if dialogue and dialogue.lower() not in {"none", "n/a", ""}:
        parts.append(f"Audio/dialogue: {dialogue}")
    notable = brief.get("notable_details", "")
    if notable and notable.lower() not in {"none", "n/a", ""}:
        parts.append(f"Notable details: {notable}")
    return " ".join(p for p in parts if p)


def _build_verify_prompt(draft: str) -> str:
    # Exact competitor verify prompt.
    return f"""Here is a draft description of the video frames:

{draft}

First, critique the draft by listing each specific concrete claim (objects, animals, vehicles, locations, text, landmarks). For each claim, decide if it is: (a) clearly a real visible object/scene, (b) a graphical overlay, watermark, dissolve, or transition effect, (c) partially visible or unclear, or (d) not supported by the frames.

Then rewrite the description as plain text, keeping only claims in category (a). For category (b), describe the graphical element generically only if it is central. Remove or generalize categories (c) and (d). Never describe overlays or transition effects as if they are real-world objects or scenes.

Also remove or generalize:
- Exact quoted text, brand names, signs, slogans
- Ethnicity, identity labels, religion markers
- Location claims (city names, countries, landmarks)

Output only the final rewritten factual description. Do not output the critique list. Do not mention frames, AI, uncertainty, or analysis."""


def _generate_structured_brief(frames_b64: list[str], transcript: str) -> dict[str, str]:
    """First call: produce a structured brief from frames (+ optional transcript)."""
    last_error = None
    prompt = _build_brief_prompt(transcript)
    for attempt in range(2):
        try:
            raw_text = _call_vision(frames_b64, prompt, BRIEF_MAX_TOKENS)
            parsed = _strip_to_json(raw_text)
            brief = _normalize_brief(parsed)
            print(f"[perception] brief ok via {_working_model} "
                  f"(attempt {attempt + 1})", flush=True)
            return brief
        except Exception as e:  # noqa: BLE001
            last_error = e
            print(f"[perception] brief attempt {attempt + 1} failed: {e}",
                  flush=True)
    raise RuntimeError(f"Could not produce valid video brief: {last_error}")


def _verify_description(frames_b64: list[str], draft_paragraph: str) -> str:
    """Second call: verify the brief against the frames; plain-text description."""
    try:
        verified = _call_vision(
            frames_b64,
            _build_verify_prompt(draft_paragraph),
            BRIEF_MAX_TOKENS,
        )
        text = verified.strip()
        if text:
            print("[perception] verify ok", flush=True)
            return text
    except Exception as e:  # noqa: BLE001
        print(f"[perception] verify failed (keeping draft): {e}", flush=True)
    return draft_paragraph


def describe(frames_b64: list[str], transcript: str = "") -> str:
    """
    Generate a dense, verified plain-text description of the video.

    Returns a single paragraph the caption model can use as factual grounding.
    """
    try:
        brief = _generate_structured_brief(frames_b64, transcript)
        draft = _brief_to_paragraph(brief)
    except Exception as e:  # noqa: BLE001
        # Reliability fallback: free-text draft so the clip is never zeroed.
        print(f"[perception] structured brief failed, free-text fallback: {e}",
              flush=True)
        draft = _call_vision(
            frames_b64,
            "Describe this video clip factually in 3-5 sentences based only "
            "on what is visible in the frames. No speculation.",
            280,
        )
        print(f"[perception] free-text draft ok via {_working_model}",
              flush=True)

    return _verify_description(frames_b64, draft)
