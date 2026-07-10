"""Shared Groq (OpenAI-compatible) client + hardened chat helper.

Env:
  GROQ_API_KEY   (your key locally; injected by grader if applicable)
  GROQ_BASE_URL  (default: https://api.groq.com/openai/v1)
"""
import os
import re
import time

from openai import OpenAI

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
# Groq: "Please try again in 1.23s" / "try again in 2s"
_RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)\s*s", re.I)

_client = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ.get("GROQ_API_KEY", "missing-key"),
            base_url=os.environ.get("GROQ_BASE_URL", DEFAULT_BASE_URL),
            timeout=float(os.environ.get("PER_CALL_TIMEOUT", "25")),
            max_retries=0,  # we do our own retries so we control the clock
        )
    return _client


def _rate_limit_wait(err: Exception) -> float | None:
    """Seconds to sleep on 429, or None if not a rate-limit error."""
    msg = str(err)
    low = msg.lower()
    if "429" not in msg and "rate limit" not in low and "rate_limit" not in low:
        return None
    m = _RETRY_AFTER_RE.search(msg)
    if m:
        return float(m.group(1)) + 0.35  # small cushion past the window
    return 2.0


def chat(model: str, messages: list, max_tokens: int = 400,
         temperature: float = 0.6, response_format: dict | None = None,
         retries: int = 1, rate_limit_retries: int = 3) -> str:
    """Call chat.completions.

    On 429: parse Groq's "try again in Xs", sleep, retry (up to
    rate_limit_retries). On 400: retry once WITHOUT optional params
    (Track 1 lesson). Raises on final failure — caller decides fallback.
    """
    kwargs = dict(model=model, messages=messages,
                  max_tokens=max_tokens, temperature=temperature)
    if response_format:
        kwargs["response_format"] = response_format

    last_err = None
    soft_left = retries
    rate_left = rate_limit_retries

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
            wait = _rate_limit_wait(e)
            if wait is not None and rate_left > 0:
                rate_left -= 1
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
