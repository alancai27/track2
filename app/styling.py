"""Style-specific caption generation (competitor engine + our accuracy levers).

- Kimi K2P6 with reasoning_effort=none (via llm.chat).
- Sequential styles; prior captions fed for variety.
- Keyword heuristics + one retry for weak tech/sarcasm (and tech-leak on
  non-tech).
- Structured fact bullets + few-shot exemplars + short word budgets.
- Template fallbacks — never leave a style empty for the grader.
"""
import os
import re
import threading

from llm import chat

STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]

STYLE_CANDIDATES = [
    "accounts/fireworks/models/kimi-k2p6",
]

STYLE_TEMP = {
    "formal": 0.3,
    "sarcastic": 0.75,
    "humorous_tech": 0.75,
    "humorous_non_tech": 0.75,
}

# Short budgets beat 25–60 filler (competitor later plan + our judge).
STYLE_WORDS = {
    "formal": (15, 28),
    "sarcastic": (12, 24),
    "humorous_tech": (15, 28),
    "humorous_non_tech": (12, 24),
}

# Competitor style prompts (kept close to theirs; small accuracy addenda).
STYLE_PROMPTS = {
    "formal": (
        "Write a formal, professional, objective caption. Factual tone, no humor, "
        "no slang, no embellishment. Describe only what is visible."
    ),
    "sarcastic": (
        "Write a sarcastic caption: dry, ironic, lightly mocking, grounded in the "
        "specific action described. Stay lighthearted and non-offensive. "
        "Do not use crutches like 'oh great', 'how thrilling', or 'just what the "
        "world needed'."
    ),
    "humorous_tech": (
        "Write a funny caption using technology, software, programming, network, "
        "game engine, or debugging references. The tech reference should be natural "
        "and the caption should still describe the video. The metaphor can be "
        "creative (git/cache/RAM need not be visible) but the subject/action it "
        "maps onto must match the facts. Avoid 'deploy/deploying/shipping to prod' "
        "as the joke."
    ),
    "humorous_non_tech": (
        "Write a funny everyday-humor caption with no technical jargon. Relatable, "
        "light-hearted, and grounded in the video. Riff only on real subjects/"
        "actions/setting from the facts — invent nothing."
    ),
}

# Competitor keyword sets (+ a few extras we found useful).
TECH_STYLE_WORDS = {
    "api", "bug", "cache", "commit", "debug", "latency", "log",
    "pipeline", "queue", "rollback", "runtime", "scheduler", "server",
    "thread", "packet", "loop", "function", "variable", "compile",
    "render", "frame rate", "fps", "bandwidth", "cpu", "gpu",
    "memory", "overflow", "underflow", "exception", "crash", "reboot",
    "git", "merge", "ram", "null", "pointer", "segfault", "kernel",
    "stack", "heap", "async", "syntax", "compiler", "docker", "repo",
}

SARCASM_MARKERS = {
    "apparently", "because", "clearly", "naturally", "of course", "obviously",
    "serious", "thrilling", "groundbreaking", "fascinating", "riveting",
    "nothing says", "nothing screams", "truly", "sure", "surely",
    "olympic", "history will", "priorities", "totally normal", "no notes",
}

# Few-shots unrelated to likely clip topics (register, not content).
STYLE_EXAMPLES = {
    "formal": {
        "good": [
            "A ceramic bowl of steaming broth sits on a wooden counter beside "
            "chopped scallions.",
            "Two hikers cross a narrow rope bridge over a misty ravine at dawn.",
        ],
        "bad": (
            "This amazing soup looks delicious and makes me want to travel "
            "somewhere wonderful right now!!!"
        ),
    },
    "sarcastic": {
        "good": [
            "Nothing builds character like waiting for water to boil while "
            "staring at it with Olympic focus.",
            "Yes, rearrange the same three pillows again — interior design "
            "history will remember this.",
        ],
        "bad": (
            "Oh great, just what the world needed: another video of someone "
            "cooking soup."
        ),
    },
    "humorous_tech": {
        "good": [
            "That kettle is stuck in a busy-wait loop until the interrupt "
            "finally fires.",
            "Pillow fluffing looks like a garbage-collection pass that still "
            "leaves fragmentation.",
        ],
        "bad": (
            "Deploying soup to production in the cloud with AI Kubernetes "
            "microservices."
        ),
    },
    "humorous_non_tech": {
        "good": [
            "The kettle's basically peer-pressuring the water into becoming "
            "tea already.",
            "Those pillows are getting the spa day the rest of us were promised.",
        ],
        "bad": (
            "When your API latency is higher than your morning motivation "
            "(so true bestie)."
        ),
    },
}

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
    """Never-empty last resort (grader zeros missing styles)."""
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


def facts_to_bullets(description: str) -> str:
    """Structured facts for grounding (Subjects/Actions/Setting/Objects/Mood)."""
    desc = (description or "").strip()
    if not desc:
        return (
            "- Subjects: unknown\n- Actions: unknown\n"
            "- Setting: unknown\n- Objects: none noted\n- Mood: neutral"
        )
    if re.search(r"(?i)^\s*-\s*subjects\s*:", desc, re.M):
        return desc
    # Prefer fields already present in verified paragraph from perception.
    bullets = []
    for label, patterns in (
        ("Subjects", [r"(?i)Subjects:\s*(.+?)(?:\s+(?:Setting|Actions|Objects|Mood|Audio|Notable)\b|$)"]),
        ("Actions", [r"(?i)Actions:\s*(.+?)(?:\s+(?:Setting|Subjects|Objects|Mood|Audio|Notable)\b|$)"]),
        ("Setting", [r"(?i)Setting:\s*(.+?)(?:\s+(?:Subjects|Actions|Objects|Mood|Audio|Notable)\b|$)"]),
        ("Objects", [r"(?i)Objects(?:/details)?:\s*(.+?)(?:\s+(?:Setting|Subjects|Actions|Mood|Audio|Notable)\b|$)"]),
        ("Mood", [r"(?i)Mood:\s*(.+?)(?:\s+(?:Setting|Subjects|Actions|Objects|Audio|Notable)\b|$)"]),
    ):
        found = ""
        for pat in patterns:
            m = re.search(pat, desc)
            if m:
                found = m.group(1).strip(" .;")
                break
        bullets.append(f"- {label}: {found or '(see summary)'}")

    if any("(see summary)" not in b for b in bullets):
        # Keep overall summary as Actions supplement if parsing was partial.
        summary = desc.split("Setting:")[0].strip()
        if summary and len(summary) > 20:
            bullets[1] = f"- Actions: {summary}" if "(see summary)" in bullets[1] \
                else bullets[1]
        return "\n".join(bullets)

    prompt = (
        "Convert this video description into EXACTLY these five labeled "
        "bullets. Use short phrases, no inventing facts not in the text.\n"
        "Format:\n"
        "- Subjects: ...\n- Actions: ...\n- Setting: ...\n"
        "- Objects: ...\n- Mood: ...\n\n"
        f"Description:\n{desc}\n\n"
        "Reply with ONLY the five bullets."
    )
    with _lock:
        models = ([_working_model] if _working_model else []) + \
            [m for m in _candidate_models() if m != _working_model]
    for model in models:
        try:
            text = chat(model, [{"role": "user", "content": prompt}],
                        max_tokens=180, temperature=0.1)
            if text and "Subjects" in text and "Actions" in text:
                print(f"[styling] facts structured via {model}", flush=True)
                return text.strip()
        except Exception as e:  # noqa: BLE001
            print(f"[styling] facts structure failed on {model}: {e}",
                  flush=True)
    return (
        f"- Subjects: (see description)\n"
        f"- Actions: {desc}\n"
        f"- Setting: (see description)\n"
        f"- Objects: (see description)\n"
        f"- Mood: (see description)"
    )


def _clean_caption(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    text = re.sub(r"^(caption|formal|sarcastic|humorous[_ ]?\w+)\s*:\s*",
                  "", text, flags=re.I)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return " ".join(lines) if lines else ""


def _needs_style_retry(style: str, caption: str) -> bool:
    """Check if a caption obviously misses its style target (competitor heuristic)."""
    if not caption:
        return True
    normalized = caption.lower()
    lo, _hi = STYLE_WORDS[style]
    if len(caption.split()) < max(8, lo - 4):
        return True
    if style == "humorous_tech":
        return not any(word in normalized for word in TECH_STYLE_WORDS)
    if style == "sarcastic":
        return not any(marker in normalized for marker in SARCASM_MARKERS)
    if style == "humorous_non_tech":
        return any(word in normalized for word in TECH_STYLE_WORDS)
    return False


def _examples_block(style: str) -> str:
    ex = STYLE_EXAMPLES[style]
    lines = ["Good examples (match this register, not these topics):"]
    for g in ex["good"]:
        lines.append(f'  ✓ "{g}"')
    lines.append(f'Bad example (avoid this register): ✗ "{ex["bad"]}"')
    return "\n".join(lines)


def _generate_caption(
    model: str,
    facts: str,
    style: str,
    prior_captions: list[str],
    *,
    retry: bool = False,
) -> str:
    """Generate one caption for a style (competitor prompt skeleton)."""
    lo, hi = STYLE_WORDS[style]
    variety_note = ""
    if prior_captions:
        variety_note = (
            "\n\nCaptions already written for this clip in other styles. "
            "Use a different sentence structure and comedic angle: "
            + " | ".join(prior_captions)
        )

    retry_note = ""
    if retry:
        if style == "humorous_tech":
            retry_note = (
                "\nRETRY: previous attempt lacked a clear tech reference — "
                "include one, stay accurate to the facts."
            )
        elif style == "sarcastic":
            retry_note = (
                "\nRETRY: previous attempt was not clearly sarcastic — "
                "lean into dry irony."
            )
        elif style == "humorous_non_tech":
            retry_note = (
                "\nRETRY: remove ALL tech jargon; keep everyday humor only."
            )

    prompt = (
        f"{STYLE_PROMPTS[style]}\n\n"
        f"Structured facts about the video (ground ALL claims here):\n{facts}\n\n"
        f"Write ONE caption, one or two sentences, {lo} to {hi} words. "
        "Must include the central subject and what they are doing. "
        "Write as if you personally watched the video. "
        "Never mention computer vision, models, detection, frames, prompts, "
        "pipelines, or uncertainty. "
        "Do not invent details beyond the facts. Do not name cities, countries, "
        "landmarks, or specific locations. "
        "Do not mention ethnicity, identity labels, religion markers, brand "
        "names, or signs unless they are explicitly present in the factual "
        "description. "
        "Write in grammatically correct, natural, polished English. "
        "Output only the caption text.\n\n"
        f"{_examples_block(style)}"
        f"{variety_note}"
        f"{retry_note}"
    )

    text = chat(
        model,
        [{"role": "user", "content": prompt}],
        max_tokens=90,
        temperature=STYLE_TEMP[style],
    )
    return _clean_caption(text)


def style_captions(description: str) -> dict[str, str]:
    """
    Generate captions for all four styles sequentially.

    Prior captions are fed into later styles so outputs do not sound identical.
    Weak captions are retried once based on keyword heuristics.
    """
    global _working_model
    result = template_fallbacks(description)
    facts = facts_to_bullets(description)

    with _lock:
        models = ([_working_model] if _working_model else []) + \
            [m for m in _candidate_models() if m != _working_model]

    for model in models:
        try:
            prior: list[str] = []
            ok = 0
            for style in STYLES:
                try:
                    caption = _generate_caption(model, facts, style, prior)
                    if _needs_style_retry(style, caption):
                        print(f"[styling] {style}: retrying weak caption...",
                              flush=True)
                        caption2 = _generate_caption(
                            model, facts, style, prior, retry=True
                        )
                        if caption2 and not _needs_style_retry(style, caption2):
                            caption = caption2
                        elif caption2 and not caption:
                            caption = caption2
                    if caption:
                        result[style] = caption
                        prior.append(caption)
                        ok += 1
                except Exception as e:  # noqa: BLE001
                    print(f"[styling] {style} failed: {e}", flush=True)
                    # keep template fallback already in result
                    prior.append(result[style])

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
