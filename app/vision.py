"""Two-pass visual grounding: structured brief, then frame-checked rewrite."""
import json
import re

from client import complete
import settings

BRIEF_KEYS = (
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


def _messages(prompt: str, frames_b64: list[str]) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": prompt}]
    for b64 in frames_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    return [{"role": "user", "content": content}]


def _strip_to_json(text: str) -> dict:
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


def _build_brief_prompt(transcript: str = "") -> str:
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
    out: dict[str, str] = {}
    for key in BRIEF_KEYS:
        val = parsed.get(key, "")
        out[key] = "" if val is None else str(val).strip()
    if not out["overall_summary"] and not out["subjects"]:
        raise ValueError("brief missing overall_summary and subjects")
    return out


def _brief_to_paragraph(brief: dict[str, str]) -> str:
    parts = [
        brief["overall_summary"],
        f"Setting: {brief['setting']}",
        f"Subjects: {brief['subjects']}",
        f"Actions: {brief['actions']}",
        f"Objects/details: {brief['objects']}",
        f"Mood: {brief['mood']}",
    ]
    if brief["dialogue_summary"] and brief["dialogue_summary"].lower() not in {
        "none", "n/a", "",
    }:
        parts.append(f"Audio/dialogue: {brief['dialogue_summary']}")
    if brief["notable_details"] and brief["notable_details"].lower() not in {
        "none", "n/a", "",
    }:
        parts.append(f"Notable details: {brief['notable_details']}")
    return " ".join(parts)


def _build_verify_prompt(draft: str) -> str:
    return f"""Here is a draft description of the video frames:

{draft}

First, critique the draft by listing each specific concrete claim (objects, animals, vehicles, locations, text, landmarks). For each claim, decide if it is: (a) clearly a real visible object/scene, (b) a graphical overlay, watermark, dissolve, or transition effect, (c) partially visible or unclear, or (d) not supported by the frames.

Then rewrite the description as plain text, keeping only claims in category (a). For category (b), describe the graphical element generically only if it is central. Remove or generalize categories (c) and (d). Never describe overlays or transition effects as if they are real-world objects or scenes.

Also remove or generalize:
- Exact quoted text, brand names, signs, slogans
- Ethnicity, identity labels, religion markers
- Location claims (city names, countries, landmarks)

Output only the final rewritten factual description. Do not output the critique list. Do not mention frames, AI, uncertainty, or analysis."""


def _call_vision(frames_b64: list[str], prompt: str, max_tokens: int) -> str:
    return complete(
        settings.VISION_MODEL,
        _messages(prompt, frames_b64),
        max_tokens=max_tokens,
        temperature=0.1,
    )


def ground_clip(frames_b64: list[str], transcript: str = "") -> str:
    """
    Dense verified prose description for the caption model.

    Pass 1: structured JSON brief. Pass 2: verify against the same frames.
    """
    if not frames_b64:
        raise ValueError("no frames to ground")

    last_error: Exception | None = None
    brief: dict[str, str] | None = None
    for attempt in range(2):
        try:
            raw = _call_vision(
                frames_b64,
                _build_brief_prompt(transcript),
                settings.BRIEF_TOKENS,
            )
            brief = _normalize_brief(_strip_to_json(raw))
            print(f"[vision] brief ok (attempt {attempt + 1})", flush=True)
            break
        except Exception as e:  # noqa: BLE001
            last_error = e
            print(f"[vision] brief attempt {attempt + 1} failed: {e}", flush=True)

    if brief is None:
        raise RuntimeError(f"Could not produce valid video brief: {last_error}")

    draft = _brief_to_paragraph(brief)
    verified = _call_vision(
        frames_b64, _build_verify_prompt(draft), settings.BRIEF_TOKENS
    ).strip()
    print("[vision] verify ok", flush=True)
    return verified
