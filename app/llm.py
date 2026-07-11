"""Shared Groq (OpenAI-compatible) client + hardened chat helper.

Env:
  GROQ_API_KEY   (required; set locally via .env, or baked into the image
                  at build time via Dockerfile ARG/ENV — never committed)
  GROQ_BASE_URL  (default: https://api.groq.com/openai/v1)
"""
import os
import re
import time

from openai import OpenAI

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"

# Groq: "Please try again in 1.23s" / "try again in 500ms"
_RETRY_AFTER_RE = re.compile(
    r"try again in\s+([\d.]+)\s*(ms|s|m)?", re.I
)

_client = None


def _api_key() -> str:
    return (os.environ.get("GROQ_API_KEY") or "").strip() or "missing-key"


def key_source() -> str:
    """'set' or 'missing' — for startup diagnostics (never prints the key)."""
    return "set" if (os.environ.get("GROQ_API_KEY") or "").strip() else "missing"


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=_api_key(),
            base_url=os.environ.get("GROQ_BASE_URL", DEFAULT_BASE_URL),
            timeout=float(os.environ.get("PER_CALL_TIMEOUT", "45")),
            max_retries=0,  # we do our own retries so we control the clock
        )
    return _client


def _is_rate_limit(err: Exception) -> bool:
    msg = str(err)
    low = msg.lower()
    return (
        "429" in msg
        or "rate limit" in low
        or "rate_limit" in low
        or "tokens per minute" in low
        or "tpm" in low
    )


def _rate_limit_wait(err: Exception, attempt: int) -> float:
    """Seconds to sleep on 429. Prefer Groq's hint; else exponential backoff."""
    m = _RETRY_AFTER_RE.search(str(err))
    if m:
        val = float(m.group(1))
        unit = (m.group(2) or "s").lower()
        if unit == "ms":
            wait = val / 1000.0
        elif unit == "m":
            wait = val * 60.0
        else:
            wait = val
        wait = wait + 0.75  # cushion past the window
    else:
        # 2, 4, 8, 16, 20, 20...
        wait = min(2.0 * (2 ** attempt), 20.0)
    return max(1.0, min(wait, 45.0))


def chat(model: str, messages: list, max_tokens: int = 400,
         temperature: float = 0.6, response_format: dict | None = None,
         retries: int = 1, rate_limit_retries: int = 10) -> str:
    """Call chat.completions.

    On 429: parse Groq's "try again in Xs", sleep, retry (up to
    rate_limit_retries — default 10, enough for a 12-clip TPM grind).
    On 400: retry once WITHOUT optional params (Track 1 lesson).
    Raises on final failure — caller decides fallback.
    """
    kwargs = dict(model=model, messages=messages,
                  max_tokens=max_tokens, temperature=temperature)
    if response_format:
        kwargs["response_format"] = response_format

    last_err = None
    soft_left = retries
    rate_left = rate_limit_retries
    rate_attempt = 0

    while True:
        try:
            resp = client().chat.completions.create(**kwargs)
            text = (resp.choices[0].message.content or "").strip()
            if text:
                return text
            last_err = RuntimeError("empty completion")
            if soft_left <= 0:
                break
            soft_left -= 1
            time.sleep(1.0)
            continue
        except Exception as e:  # noqa: BLE001
            last_err = e
            if _is_rate_limit(e) and rate_left > 0:
                wait = _rate_limit_wait(e, rate_attempt)
                rate_left -= 1
                rate_attempt += 1
                print(f"[llm] 429 on {model}; sleep {wait:.1f}s "
                      f"({rate_left} rate-retries left)", flush=True)
                time.sleep(wait)
                continue
            msg = str(e)
            # 400 → strip optional params and try bare-bones once
            if "400" in msg or "invalid" in msg.lower():
                kwargs.pop("response_format", None)
            if soft_left <= 0:
                break
            soft_left -= 1
            time.sleep(1.0)

    raise last_err
