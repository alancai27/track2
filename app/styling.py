"""Styling: verified description -> 4 captions via sequential Fireworks calls.

Strategy (92%-competitor techniques):
  * Generate styles ONE AT A TIME; each call sees prior captions and must
    use a different sentence structure / comedic angle.
  * Shared hard rules (watched-it voice, no CV jargon, no invented details,
    25–60 words).
  * Per-style temperature (formal 0.3; others 0.75).
  * Keyword guardrail + one regenerate for sarcastic / humorous_tech.
kimi-k2p6 is a reasoning model — llm.chat sends reasoning_effort=none.
Template fallbacks if everything fails (missing style = 0 for the clip).
"""
import os
import re
import threading

from llm import chat

STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]

# Fireworks kimi-k2p6 (reasoning; effort=none via llm.chat).
STYLE_CANDIDATES = [
    "accounts/fireworks/models/kimi-k2p6",
]

STYLE_TEMP = {
    "formal": 0.3,
    "sarcastic": 0.75,
    "humorous_tech": 0.75,
    "humorous_non_tech": 0.75,
}

SHARED_RULES = (
    "Write as if you personally watched the video. "
    "Never mention computer vision, models, detection, frames, prompts, "
    "pipelines, or uncertainty. "
    "Do not invent details beyond the description. Do not name cities, "
    "countries, landmarks, or specific locations. Do not mention ethnicity, "
    "identity labels, religion, or brand names unless explicitly in the "
    "description. "
    "Write ONE caption, one or two sentences, roughly 25 to 60 words. "
    "Write in grammatically correct, natural, polished English. Re-read and "
    "ensure the sentence is fluent and error-free. "
    "Reply with the caption only — no quotes, no labels, no preamble."
)

STYLE_SPECS = {
    "formal": (
        "Polished, objective, professional voice. Factual and precise. "
        "No opinion, no humor, no exclamation marks."
    ),
    "sarcastic": (
        "Dry irony — fake enthusiasm, mock profundity, deadpan understatement, "
        "or ironic praise. Lightly mock the scene while staying accurate. "
        "BANNED phrases (any wording): 'just what the world needed', 'just "
        "what was missing', 'oh great', 'how thrilling', 'because the world "
        "was missing that'."
    ),
    "humorous_tech": (
        "Genuinely funny caption using ONE fresh tech/programming metaphor "
        "that fits the scene (git, merge conflicts, APIs, RAM, caching, "
        "compiling, debugging, Stack Overflow, cloud, latency, null pointers, "
        "race conditions, segfaults, unit tests, CI, kernels…). "
        "Do NOT use deploy/deploying/shipping-to-prod as the joke. "
        "Be witty and creative with the metaphor — git/cache/RAM/etc. do NOT "
        "need to be visible in the clip. What MUST stay accurate is the "
        "real subject/action/setting the metaphor is mapped onto: describe "
        "what is actually happening, then land a clever tech joke about it. "
        "Do not invent people, objects, or events that aren't in the "
        "description."
    ),
    "humorous_non_tech": (
        "Genuinely funny everyday caption — playful, relatable, warm. "
        "Absolutely NO tech or programming references or jargon. "
        "ACCURACY FIRST: the joke MUST riff on details actually visible in "
        "the clip (real subjects, actions, setting, objects from the "
        "description). Stay funny, but do NOT invent objects, people, "
        "scenarios, or events that aren't in the description. Every clause "
        "must remain factually true to the clip."
    ),
}

# Guardrail keyword lists (lowercase substring match).
TECH_KEYWORDS = (
    "git", "commit", "merge", "api", "apis", "ram", "cache", "caching",
    "compile", "compiling", "compiled", "debug", "debugging", "debugger",
    "stack overflow", "cloud", "latency", "null", "pointer", "segfault",
    "race condition", "unit test", "ci ", " ci", "kernel", "bug", "cpu",
    "gpu", "server", "code", "coding", "algorithm", "repo", "thread",
    "async", "json", "http", "memory", "buffer", "stack", "heap", "runtime",
    "exception", "timeout", "ping", "bandwidth", "pixel", "bit ", "byte",
    "syntax", "compiler", "interpreter", "protocol", "socket", "query",
    "database", "sql", "regex", "pipeline", "container", "docker", "linux",
    "kernel", "firmware", "hardware", "software", "npm", "python", "java",
)

SARCASM_MARKERS = (
    "clearly", "apparently", "obviously", "sure,", "of course", "nothing says",
    "peak ", "as if", "riveting", "thrilling", "groundbreaking", "masterpiece",
    "fascinating", "who wouldn't", "because nothing", "truly", "absolutely",
    "essential", "exactly what", "just what every", "inspiring", "heroic",
    "legendary", "breathtaking", "nail-biting", "can't wait", "thrilled",
    "delighted", "privileged", "honored", "deeply moving", "life-changing",
    "unprecedented", "bold strategy", "flawless", "perfect timing",
    "what a time", "another day", "priorities", "totally normal",
    "no notes", "chef's kiss", "iconic", "vibes", "surely",
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


def template_fallbacks(description: str) -> dict[str, str]:
    """Never-empty last resort. Generic but grounded if we have a description."""
    scene = description.strip().split(".")[0].strip() if description.strip() \
        else "A short video clip"
    scene = scene[:140]
    low = scene[0].lower() + scene[1:] if scene else "the scene"
    return {
        "formal": f"{scene}.",
        "sarcastic": f"Ah yes, {low} — truly riveting stuff.",
        "humorous_tech": (
            f"POV: {low}, running smoothly with zero bugs in production "
            f"for once."
        ),
        "humorous_non_tech": f"Just {low}, living its best life.",
    }


def _clean_caption(text: str) -> str:
    text = text.strip().strip('"').strip("'").strip()
    text = re.sub(r"^(caption|formal|sarcastic|humorous[_ ]?\w+)\s*:\s*",
                  "", text, flags=re.I)
    # Keep at most two sentences' worth of lines.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return " ".join(lines) if lines else ""


def _has_tech(caption: str) -> bool:
    low = f" {caption.lower()} "
    return any(k in low or k in caption.lower() for k in TECH_KEYWORDS)


def _has_sarcasm(caption: str) -> bool:
    low = caption.lower()
    return any(m in low for m in SARCASM_MARKERS)


def _passes_guardrail(style: str, caption: str) -> bool:
    if not caption or len(caption.split()) < 8:
        return False
    if style == "humorous_tech":
        return _has_tech(caption)
    if style == "sarcastic":
        return _has_sarcasm(caption)
    if style == "humorous_non_tech":
        return not _has_tech(caption)
    return True


def _build_prompt(description: str, style: str,
                  prior: dict[str, str]) -> str:
    parts = [
        f"Video description:\n{description}\n",
        f"Write a {style} caption.\nStyle brief: {STYLE_SPECS[style]}\n",
        f"Rules: {SHARED_RULES}",
    ]
    if prior:
        parts.append(
            "\nAlready-written captions for this SAME clip (do NOT copy "
            "their sentence structure or comedic angle — pick a distinctly "
            "different structure and angle):\n"
        )
        for s, cap in prior.items():
            parts.append(f"- {s}: {cap}\n")
    return "".join(parts)


def _generate_one(model: str, description: str, style: str,
                  prior: dict[str, str], *, retry: bool = False) -> str:
    prompt = _build_prompt(description, style, prior)
    if retry:
        if style == "humorous_tech":
            prompt += (
                "\nRETRY: Your previous attempt lacked a clear tech metaphor. "
                "Include an explicit programming/tech reference that fits."
            )
        elif style == "sarcastic":
            prompt += (
                "\nRETRY: Your previous attempt was not clearly sarcastic. "
                "Lean harder into dry irony or ironic praise."
            )
        elif style == "humorous_non_tech":
            prompt += (
                "\nRETRY: Remove ALL tech/programming jargon; keep it everyday."
            )
    text = chat(
        model,
        [{"role": "user", "content": prompt}],
        max_tokens=120,
        temperature=STYLE_TEMP[style],
    )
    return _clean_caption(text)


def style_captions(description: str) -> dict[str, str]:
    """Return dict with all 4 styles, guaranteed non-empty."""
    global _working_model
    result = template_fallbacks(description)

    with _lock:
        models = ([_working_model] if _working_model else []) + \
            [m for m in _candidate_models() if m != _working_model]

    for model in models:
        try:
            prior: dict[str, str] = {}
            ok = 0
            for style in STYLES:
                caption = _generate_one(model, description, style, prior)
                if not _passes_guardrail(style, caption):
                    print(f"[styling] {style} failed guardrail; retrying",
                          flush=True)
                    caption2 = _generate_one(
                        model, description, style, prior, retry=True
                    )
                    if _passes_guardrail(style, caption2):
                        caption = caption2
                    elif caption2 and not caption:
                        caption = caption2
                if caption:
                    result[style] = caption
                    prior[style] = caption
                    ok += 1
            if ok == 4:
                with _lock:
                    _working_model = model
                print(f"[styling] sequential ok via {model}", flush=True)
                return result
            if ok > 0:
                with _lock:
                    _working_model = model
                print(f"[styling] partial ({ok}/4) via {model}", flush=True)
                return result
        except Exception as e:  # noqa: BLE001
            print(f"[styling] {model} failed: {e}", flush=True)

    print("[styling] all models failed; using template fallbacks", flush=True)
    return result
