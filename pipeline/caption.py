"""
Style-specific caption generation.

- Uses Kimi K2P6 with reasoning_effort=none for clean, non-reasoning output.
- Generates captions sequentially so each style can avoid copying the previous one.
- Verifies captions against keyword heuristics and retries once if the style is weak.
- Optionally falls back to a generic caption if generation fails completely.
"""
from openai import OpenAI

from config import Config


# Keywords that signal the requested style is actually present.
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


def _clean_caption(text: str) -> str:
    """Strip surrounding quotes and whitespace from a caption."""
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    return text


def _needs_style_retry(style: str, caption: str) -> bool:
    """Check if a caption obviously misses its style target."""
    normalized = caption.lower()
    if style == "humorous_tech":
        return not any(word in normalized for word in TECH_STYLE_WORDS)
    if style == "sarcastic":
        return not any(marker in normalized for marker in SARCASM_MARKERS)
    return False


def _generate_caption(
    client: OpenAI,
    description: str,
    style: str,
    prior_captions: list[str],
) -> str:
    """Generate one caption for a style."""
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

    response = client.chat.completions.create(
        model=Config.FIREWORKS_TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=Config.CAPTION_MAX_TOKENS,
        temperature=0.75 if style in {"sarcastic", "humorous_tech", "humorous_non_tech"} else 0.3,
        extra_body={"reasoning_effort": Config.REASONING_EFFORT},
    )
    text = response.choices[0].message.content
    if text is None:
        raise ValueError("Model returned no content.")
    return _clean_caption(text)


def generate_captions(description: str) -> dict[str, str]:
    """
    Generate captions for all four styles sequentially.

    Prior captions are fed into later styles so the four outputs do not sound identical.
    Weak captions are retried once based on simple keyword heuristics.
    """
    client = OpenAI(
        api_key=Config.FIREWORKS_API_KEY,
        base_url=Config.FIREWORKS_BASE_URL,
    )

    results: dict[str, str] = {}
    prior: list[str] = []

    for style in Config.REQUIRED_STYLES:
        try:
            caption = _generate_caption(client, description, style, prior)

            # Retry once if the caption clearly misses the style target.
            if _needs_style_retry(style, caption):
                print(f"  {style}: retrying weak caption...")
                caption = _generate_caption(client, description, style, prior)

            results[style] = caption
            prior.append(caption)
        except Exception as e:
            print(f"  Caption generation failed for {style}: {e}")
            results[style] = f"Unable to generate a {style} caption for this clip."
            prior.append(results[style])

    return results
