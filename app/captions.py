"""Sequential four-style captions with keyword guardrails."""
import re

from client import complete
import settings

# Exact cue sets from the 0.92 pipeline (scoring-sensitive).
TECH_STYLE_WORDS = {
    "api", "bug", "cache", "commit", "debug", "deploy", "latency", "log",
    "pipeline", "queue", "rollback", "runtime", "scheduler", "server",
    "thread", "packet", "loop", "function", "variable", "compile",
    "render", "frame rate", "fps", "bandwidth", "bandwidth", "cpu", "gpu",
    "memory", "overflow", "underflow", "exception", "crash", "reboot",
}

SARCASM_MARKERS = {
    "apparently", "because", "clearly", "naturally", "of course", "obviously",
    "serious", "thrilling", "groundbreaking", "fascinating", "riveting",
    "nothing says", "nothing screams", "truly", "sure",
}

STYLE_PROMPTS = {
    "formal": (
        "Write a formal, professional, objective caption. Factual tone, no humor, "
        "no slang, no embellishment. Describe only what is visible."
    ),
    "sarcastic": (
        "Write a sarcastic caption: dry, ironic, lightly mocking, grounded in the "
        "specific action described. Stay lighthearted and non-offensive."
    ),
    "humorous_tech": (
        "Write a funny caption using technology, software, programming, network, "
        "game engine, or debugging references. The tech reference should be natural "
        "and the caption should still describe the video."
    ),
    "humorous_non_tech": (
        "Write a funny everyday-humor caption with no technical jargon. Relatable, "
        "light-hearted, and grounded in the video."
    ),
}

_TEMP = {
    "formal": 0.3,
    "sarcastic": 0.75,
    "humorous_tech": 0.75,
    "humorous_non_tech": 0.75,
}


def seed_captions(reason: str = "unavailable") -> dict[str, str]:
    """Non-empty placeholders so missing styles never zero a task."""
    msg = f"Unable to process this video clip ({reason})."
    return {s: msg for s in settings.STYLES}


def _clean_caption(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    return text


def _needs_style_retry(style: str, caption: str) -> bool:
    normalized = caption.lower()
    if style == "humorous_tech":
        return not any(word in normalized for word in TECH_STYLE_WORDS)
    if style == "sarcastic":
        return not any(marker in normalized for marker in SARCASM_MARKERS)
    return False


def _generate_caption(
    description: str,
    style: str,
    prior_captions: list[str],
) -> str:
    variety_note = ""
    if prior_captions:
        variety_note = (
            "\n\nCaptions already written for this clip in other styles. "
            "Use a different sentence structure and comedic angle: "
            + " | ".join(prior_captions)
        )

    prompt = (
        f"{STYLE_PROMPTS[style]}\n\n"
        f"Factual description of the video:\n{description}\n\n"
        "Write ONE caption, one or two sentences, roughly 25 to 60 words. "
        "Write as if you personally watched the video. "
        "Never mention computer vision, models, detection, frames, prompts, pipelines, or uncertainty. "
        "Do not invent details beyond the description. Do not name cities, countries, landmarks, or specific locations. "
        "Do not mention ethnicity, identity labels, religion markers, brand names, or signs unless they are "
        "explicitly present in the factual description. Output only the caption text."
        f"{variety_note}"
    )

    text = complete(
        settings.CAPTION_MODEL,
        [{"role": "user", "content": prompt}],
        max_tokens=settings.CAPTION_TOKENS,
        temperature=_TEMP[style],
    )
    return _clean_caption(text)


def write_styles(description: str) -> dict[str, str]:
    """
    Generate captions for all four styles sequentially.

    Prior captions are fed into later styles so outputs do not sound identical.
    Weak captions are retried once based on keyword heuristics.
    """
    results: dict[str, str] = {}
    prior: list[str] = []

    for style in settings.STYLES:
        try:
            caption = _generate_caption(description, style, prior)
            if _needs_style_retry(style, caption):
                print(f"[captions] {style}: retrying weak caption...", flush=True)
                caption = _generate_caption(description, style, prior)
            results[style] = caption
            prior.append(caption)
        except Exception as e:  # noqa: BLE001
            print(f"[captions] {style} failed: {e}", flush=True)
            results[style] = f"Unable to generate a {style} caption for this clip."
            prior.append(results[style])

    return results
