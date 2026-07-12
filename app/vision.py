"""Two-pass visual grounding: structured scene card → verified prose."""
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
    for b64 in frames_b64[: settings.FRAME_CAP + 2]:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    return [{"role": "user", "content": content}]


def _extract_json(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON in vision reply: {text[:180]}")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("JSON root is not an object")
    return obj


def _scene_card_prompt() -> str:
    return (
        "Look at these ordered stills from one short clip.\n"
        "Return a JSON object with ONLY these string fields:\n"
        "setting, subjects, actions, objects, mood, sounds, "
        "dialogue_summary, notable_details, overall_summary.\n\n"
        "Constraints:\n"
        "- Stick to what is visibly present.\n"
        "- Prefer generic wording when unsure; never invent people, animals, "
        "vehicles, landmarks, or places.\n"
        "- overall_summary should be 2–3 tight sentences.\n"
        "- sounds / dialogue_summary may be \"none\".\n"
        "- No markdown fences, no commentary — JSON only.\n\n"
        "Example shape:\n"
        '{"setting":"...","subjects":"...","actions":"...","objects":"...",'
        '"mood":"...","sounds":"none","dialogue_summary":"none",'
        '"notable_details":"...","overall_summary":"..."}'
    )


def _normalize(card: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for key in BRIEF_KEYS:
        val = card.get(key, "")
        out[key] = "" if val is None else str(val).strip()
    if not out["overall_summary"] and not out["subjects"]:
        raise ValueError("scene card missing summary and subjects")
    return out


def _card_to_prose(card: dict[str, str]) -> str:
    chunks = [
        card["overall_summary"],
        f"Setting: {card['setting']}",
        f"Subjects: {card['subjects']}",
        f"Actions: {card['actions']}",
        f"Objects/details: {card['objects']}",
        f"Mood: {card['mood']}",
    ]
    for label, key in (
        ("Audio/dialogue", "dialogue_summary"),
        ("Notable details", "notable_details"),
    ):
        val = card.get(key, "")
        if val and val.lower() not in {"none", "n/a", ""}:
            chunks.append(f"{label}: {val}")
    return " ".join(c for c in chunks if c)


def _audit_prompt(draft: str) -> str:
    return (
        "Draft captioning notes for these stills:\n\n"
        f"{draft}\n\n"
        "Audit every concrete claim (objects, animals, vehicles, places, "
        "on-screen text). Classify each as: (a) clearly visible real content, "
        "(b) overlay/watermark/dissolve/transition, (c) unclear/partial, or "
        "(d) unsupported.\n\n"
        "Rewrite a single plain-text factual description keeping only (a). "
        "For (b), mention a graphic overlay only if it dominates the shot. "
        "Drop or soften (c) and (d). Do not treat transitions as real scenery.\n\n"
        "Also strip or generalize: quoted signage/brands, identity/ethnicity/"
        "religion labels, and named cities/countries/landmarks.\n\n"
        "Reply with the cleaned description only — no critique list, no talk "
        "about frames or AI."
    )


def ground_clip(frames_b64: list[str]) -> str:
    """
    Produce a verified prose description grounded in the frames.

    Pass 1: structured scene card. Pass 2: vision audit against the same stills.
    """
    if not frames_b64:
        raise ValueError("no frames to ground")

    last_err: Exception | None = None
    card: dict[str, str] | None = None
    for attempt in range(2):
        try:
            raw = complete(
                settings.VISION_MODEL,
                _messages(_scene_card_prompt(), frames_b64),
                max_tokens=settings.BRIEF_TOKENS,
                temperature=0.1,
            )
            card = _normalize(_extract_json(raw))
            print(f"[vision] scene card ok (try {attempt + 1})", flush=True)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[vision] scene card try {attempt + 1} failed: {e}",
                  flush=True)
    if card is None:
        # Free-text fallback so the clip is not zeroed.
        print(f"[vision] falling back to free-text draft: {last_err}",
              flush=True)
        draft = complete(
            settings.VISION_MODEL,
            _messages(
                "In 3–5 factual sentences, describe only what is visible in "
                "these ordered stills. No speculation.",
                frames_b64,
            ),
            max_tokens=280,
            temperature=0.1,
        )
    else:
        draft = _card_to_prose(card)

    try:
        verified = complete(
            settings.VISION_MODEL,
            _messages(_audit_prompt(draft), frames_b64),
            max_tokens=settings.BRIEF_TOKENS,
            temperature=0.1,
        ).strip()
        if verified:
            print("[vision] audit ok", flush=True)
            return verified
    except Exception as e:  # noqa: BLE001
        print(f"[vision] audit failed; keeping draft: {e}", flush=True)
    return draft
