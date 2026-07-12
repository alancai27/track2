"""Styling: factual description -> 4 captions via Groq text models.

Strategy: ONE call returning strict JSON with all 4 styles (fast,
consistent). If JSON parsing fails, fall back to 4 tiny per-style calls.
If everything fails, template fallbacks built from the description so no
style is ever empty (missing style = 0 for the whole clip).
"""
import json
import os
import re
import threading

from llm import chat

STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]

# Groq text ladder: primary 70B, then fast 8B fallback.
STYLE_CANDIDATES = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

STYLE_SPECS = {
    "formal": (
        "One polished, objective, professional sentence describing the scene. "
        "Factual and precise. No opinion, no humor, no exclamation marks. "
        "Write in grammatically correct, natural, polished English. Re-read "
        "and ensure the sentence is fluent and error-free."
    ),
    "sarcastic": (
        "One dry, ironic caption that lightly mocks the scene while staying "
        "accurate to what's shown. Rotate among DIFFERENT sarcastic angles — "
        "fake enthusiasm, mock profundity, deadpan understatement, or ironic "
        "praise — and vary the sentence structure. BANNED phrases (never use, "
        "in any wording): 'just what the world needed', 'just what was "
        "missing', 'oh great', 'how thrilling', 'because the world was "
        "missing that', or close paraphrases of the same crutch. Invent a "
        "genuinely different sarcastic angle for THIS specific clip. "
        "Write in grammatically correct, natural, polished English. Re-read "
        "and ensure the sentence is fluent and error-free."
    ),
    "humorous_tech": (
        "One genuinely funny caption that maps the scene onto a tech/"
        "programming metaphor that actually fits what's happening. Draw from "
        "a WIDE range — git commits, merge conflicts, APIs, RAM, caching, "
        "compiling, debugging, Stack Overflow, cloud, latency, null pointers, "
        "race conditions, segfaults, unit tests, CI, kernels, etc. — and pick "
        "ONE fresh metaphor that fits THIS clip. Do NOT reuse 'deploy'/"
        "'deploying' (or near-synonyms like 'shipping to prod') as the joke; "
        "clever and specific, not forced or formulaic. "
        "Write in grammatically correct, natural, polished English. Re-read "
        "and ensure the sentence is fluent and error-free."
    ),
    "humorous_non_tech": (
        "One genuinely funny everyday caption — playful, relatable, warm. "
        "Absolutely NO tech or programming references or jargon. "
        "Write in grammatically correct, natural, polished English. Re-read "
        "and ensure the sentence is fluent and error-free."
    ),
}

SYSTEM = (
    "You are an award-winning caption writer. You will get a factual "
    "description of a short video clip and must write 4 captions in 4 sharply "
    "distinct voices. Every caption must be grounded in what is actually in "
    "the clip (accuracy is scored), 1 sentence (max 2), in English.\n\n"
    "Styles:\n"
    + "\n".join(f"- {k}: {v}" for k, v in STYLE_SPECS.items())
    + "\n\nRespond with ONLY a JSON object, no markdown fences, exactly these "
    'keys: {"formal": "...", "sarcastic": "...", "humorous_tech": "...", '
    '"humorous_non_tech": "..."}'
)

_lock = threading.Lock()
_working_model: str | None = None


def _candidate_models() -> list[str]:
    override = os.environ.get("STYLE_MODEL") or os.environ.get("GEMMA_MODEL")
    out = [override] if override else []
    for m in STYLE_CANDIDATES:
        if m not in out:
            out.append(m)
    return out


def _parse_json(text: str) -> dict | None:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:  # noqa: BLE001
        return None


def template_fallbacks(description: str) -> dict[str, str]:
    """Never-empty last resort. Generic but grounded if we have a description."""
    scene = description.strip().split(".")[0].strip() if description.strip() \
        else "A short video clip"
    scene = scene[:140]
    low = scene[0].lower() + scene[1:] if scene else "the scene"
    return {
        "formal": f"{scene}.",
        "sarcastic": f"Ah yes, {low} — truly riveting stuff.",
        "humorous_tech": f"POV: {low}, running smoothly with zero bugs in production for once.",
        "humorous_non_tech": f"Just {low}, living its best life.",
    }


def style_captions(description: str) -> dict[str, str]:
    """Return dict with all 4 styles, guaranteed non-empty."""
    global _working_model
    result = template_fallbacks(description)

    with _lock:
        models = ([_working_model] if _working_model else []) + \
            [m for m in _candidate_models() if m != _working_model]

    user = f"Factual description of the clip:\n{description}\n\nWrite the 4 captions now."
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user}]

    for model in models:
        try:
            text = chat(model, messages, max_tokens=400, temperature=0.8)
            parsed = _parse_json(text)
            if parsed:
                good = 0
                for s in STYLES:
                    v = str(parsed.get(s, "")).strip()
                    if v:
                        result[s] = v
                        good += 1
                if good == 4:
                    with _lock:
                        _working_model = model
                    print(f"[styling] ok via {model}", flush=True)
                    return result
            # JSON came back mangled -> per-style rescue calls on this model
            rescued = _per_style(model, description, result)
            if rescued:
                with _lock:
                    _working_model = model
                return result
        except Exception as e:  # noqa: BLE001
            print(f"[styling] {model} failed: {e}", flush=True)
    print("[styling] all models failed; using template fallbacks", flush=True)
    return result


def _per_style(model: str, description: str, result: dict) -> bool:
    ok_any = False
    for s in STYLES:
        try:
            text = chat(model, [
                {"role": "user", "content":
                    f"Video description: {description}\n\nWrite exactly one "
                    f"caption in this style — {s}: {STYLE_SPECS[s]}\n"
                    "Reply with the caption only, no quotes, no preamble."},
            ], max_tokens=80, temperature=0.8)
            text = text.strip().strip('"').strip()
            if text:
                result[s] = text.split("\n")[0]
                ok_any = True
        except Exception:  # noqa: BLE001
            pass
    return ok_any
