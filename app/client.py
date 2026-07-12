"""Thin Fireworks OpenAI-compatible client with 429 backoff."""
import re
import threading
import time

from openai import OpenAI

import settings

_RETRY_HINT = re.compile(
    r"try again in\s+(?:(\d+)m\s*)?([\d.]+)\s*(ms|s)?", re.I
)
_lock = threading.Lock()
_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    with _lock:
        if _client is None:
            _client = OpenAI(
                api_key=settings.FIREWORKS_API_KEY or "missing-key",
                base_url=settings.FIREWORKS_BASE_URL,
                timeout=settings.CALL_TIMEOUT,
                max_retries=0,
            )
        return _client


def _rate_wait(err: Exception, attempt: int) -> float:
    m = _RETRY_HINT.search(str(err))
    if m:
        minutes = float(m.group(1) or 0)
        val = float(m.group(2))
        unit = (m.group(3) or "s").lower()
        wait = minutes * 60.0 + (val / 1000.0 if unit == "ms" else val) + 0.75
    else:
        wait = min(2.0 * (2 ** attempt), 20.0)
    return max(1.0, min(wait, 180.0))


def _is_429(err: Exception) -> bool:
    msg = str(err).lower()
    return "429" in msg or "rate limit" in msg or "rate_limit" in msg


def complete(
    model: str,
    messages: list,
    *,
    max_tokens: int,
    temperature: float,
    rate_retries: int = 8,
) -> str:
    """Chat completion; always sends reasoning_effort=none for clean answers."""
    kwargs = dict(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={"reasoning_effort": settings.REASONING_EFFORT},
    )
    last: Exception | None = None
    rate_left = rate_retries
    attempt = 0
    soft = 1
    while True:
        try:
            resp = get_client().chat.completions.create(**kwargs)
            text = (resp.choices[0].message.content or "").strip()
            text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.I).strip()
            if text:
                return text
            last = RuntimeError("empty completion")
            if soft <= 0:
                break
            soft -= 1
            time.sleep(1.0)
            continue
        except Exception as e:  # noqa: BLE001
            last = e
            if _is_429(e) and rate_left > 0:
                wait = _rate_wait(e, attempt)
                rate_left -= 1
                attempt += 1
                print(f"[client] 429; sleep {wait:.1f}s ({rate_left} left)",
                      flush=True)
                time.sleep(wait)
                continue
            if soft <= 0:
                break
            soft -= 1
            time.sleep(1.0)
    raise last or RuntimeError("completion failed")
