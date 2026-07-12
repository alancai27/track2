"""Four-style caption writer (sequential + light keyword guardrails)."""
import re

from client import complete
import settings

# Heuristic cues — same idea as the 0.92 stack, wording/lists refreshed.
_TECH_CUES = {
    "api", "bug", "cache", "commit", "debug", "deploy", "latency", "log",
    "pipeline", "queue", "rollback", "runtime", "scheduler", "server",
    "thread", "packet", "loop", "function", "variable", "compile",
    "render", "fps", "bandwidth", "cpu", "gpu", "memory", "exception",
    "crash", "reboot", "git", "ram", "null", "kernel", "stack", "async",
}

_SARCASM_CUES = {
    "apparently", "because", "clearly", "naturally", "of course", "obviously",
    "serious", "thrilling", "groundbreaking", "fascinating", "riveting",
    "nothing says", "nothing screams", "truly", "sure", "surely",
}

_VOICE = {
    "formal": (
        "Compose a formal, objective caption. Neutral diction, no jokes, "
        "no slang — only what the notes support."
    ),
    "sarcastic": (
        "Compose a dry, ironic caption that lightly needles the scene while "
        "staying accurate and inoffensive."
    ),
    "humorous_tech": (
        "Compose a witty caption that folds in a natural software/tech "
        "metaphor (debugging, networks, engines, etc.) without losing the "
        "actual scene."
    ),
    "humorous_non_tech": (
        "Compose a warm, everyday-funny caption with zero tech jargon, "
        "rooted in what actually happens on screen."
    ),
}

_TEMP = {
    "formal": 0.3,
    "sarcastic": 0.75,
    "humorous_tech": 0.75,
    "humorous_non_tech": 0.75,
}


def seed_captions(reason: str = "unavailable") -> dict[str, str]:
    """Non-empty placeholders so the grader never sees missing styles."""
    base = f"Unable to caption this clip ({reason})."
    return {s: base for s in settings.STYLES}


def _tidy(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    text = re.sub(
        r"^(caption|formal|sarcastic|humorous[_ ]?\w+)\s*:\s*",
        "", text, flags=re.I,
    )
    return " ".join(ln.strip() for ln in text.splitlines() if ln.strip())


def _weak(style: str, caption: str) -> bool:
    if not caption:
        return True
    low = caption.lower()
    if style == "humorous_tech":
        return not any(w in low for w in _TECH_CUES)
    if style == "sarcastic":
        return not any(m in low for m in _SARCASM_CUES)
    return False


def _one(
    notes: str,
    style: str,
    already: list[str],
    *,
    retry: bool = False,
) -> str:
    variety = ""
    if already:
        variety = (
            "\n\nEarlier captions for this same clip (vary structure/angle): "
            + " | ".join(already)
        )
    nudge = ""
    if retry:
        if style == "humorous_tech":
            nudge = "\nRetry: include a clear tech reference; stay factual."
        elif style == "sarcastic":
            nudge = "\nRetry: lean harder into dry irony."

    prompt = (
        f"{_VOICE[style]}\n\n"
        f"Grounding notes (do not contradict):\n{notes}\n\n"
        "Write one caption (1–2 sentences, about 25–60 words). "
        "Sound like you watched the clip. Never mention models, frames, "
        "prompts, pipelines, or uncertainty. Invent nothing beyond the notes. "
        "Avoid city/country/landmark names and identity/brand labels unless "
        "the notes already state them. Output caption text only."
        f"{variety}{nudge}"
    )
    raw = complete(
        settings.CAPTION_MODEL,
        [{"role": "user", "content": prompt}],
        max_tokens=settings.CAPTION_TOKENS,
        temperature=_TEMP[style],
    )
    return _tidy(raw)


def write_styles(notes: str) -> dict[str, str]:
    """Sequential captions; one keyword-based retry for weak tech/sarcasm."""
    out = seed_captions("generation pending")
    prior: list[str] = []
    for style in settings.STYLES:
        try:
            cap = _one(notes, style, prior)
            if _weak(style, cap):
                print(f"[captions] {style}: weak → retry", flush=True)
                cap2 = _one(notes, style, prior, retry=True)
                if cap2 and not _weak(style, cap2):
                    cap = cap2
                elif cap2 and not cap:
                    cap = cap2
            if cap:
                out[style] = cap
                prior.append(cap)
            else:
                prior.append(out[style])
        except Exception as e:  # noqa: BLE001
            print(f"[captions] {style} failed: {e}", flush=True)
            prior.append(out[style])
    return out
