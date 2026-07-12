"""Shared Fireworks (OpenAI-compatible) client + hardened chat helper.

Env:
  FIREWORKS_API_KEY   (required; set locally via .env, or baked into the
                       image at build time via Dockerfile ARG/ENV)
  FIREWORKS_BASE_URL  (default: https://api.fireworks.ai/inference/v1)
"""
import os
import re
import threading
import time

from openai import OpenAI

DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"

# Default for reasoning models (minimax-m3, kimi-k2p6): clean answer, no traces.
DEFAULT_EXTRA_BODY = {"reasoning_effort": "none"}

# "Please try again in 1.23s" / "2m43.1232s"
_RETRY_AFTER_RE = re.compile(
    r"try again in\s+(?:(\d+)m\s*)?([\d.]+)\s*(ms|s)?",
    re.I,
)

_client = None
_usage_lock = threading.Lock()
_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
          "calls": 0}


def _api_key() -> str:
    return (os.environ.get("FIREWORKS_API_KEY") or "").strip() or "missing-key"


def key_source() -> str:
    """'set' or 'missing' — for startup diagnostics (never prints the key)."""
    return "set" if (os.environ.get("FIREWORKS_API_KEY") or "").strip() else "missing"


def token_usage() -> dict[str, int]:
    """Cumulative prompt/completion/total tokens from successful API calls."""
    with _usage_lock:
        return dict(_usage)


def _record_usage(resp) -> None:
    u = getattr(resp, "usage", None)
    if u is None:
        return
    with _usage_lock:
        _usage["prompt_tokens"] += int(getattr(u, "prompt_tokens", 0) or 0)
        _usage["completion_tokens"] += int(
            getattr(u, "completion_tokens", 0) or 0
        )
        total = getattr(u, "total_tokens", None)
        if total is None:
            total = (getattr(u, "prompt_tokens", 0) or 0) + (
                getattr(u, "completion_tokens", 0) or 0
            )
        _usage["total_tokens"] += int(total or 0)
        _usage["calls"] += 1


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=_api_key(),
            base_url=os.environ.get("FIREWORKS_BASE_URL", DEFAULT_BASE_URL),
            timeout=float(os.environ.get("PER_CALL_TIMEOUT", "60")),
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
        or "tokens per day" in low
        or "tpm" in low
        or "tpd" in low
    )


def _rate_limit_wait(err: Exception, attempt: int) -> float:
    """Seconds to sleep on 429. Prefer provider hint; else exponential backoff."""
    m = _RETRY_AFTER_RE.search(str(err))
    if m:
        minutes = float(m.group(1) or 0)
        val = float(m.group(2))
        unit = (m.group(3) or "s").lower()
        if unit == "ms":
            wait = minutes * 60.0 + val / 1000.0
        else:
            wait = minutes * 60.0 + val
        wait = wait + 0.75
    else:
        wait = min(2.0 * (2 ** attempt), 20.0)
    return max(1.0, min(wait, 180.0))


def _message_text(resp) -> str:
    """Prefer message.content; strip accidental reasoning wrappers."""
    msg = resp.choices[0].message
    text = (getattr(msg, "content", None) or "").strip()
    if not text:
        # Some reasoning models expose a separate field; try common attrs.
        for attr in ("reasoning_content", "reasoning"):
            alt = getattr(msg, attr, None)
            if isinstance(alt, str) and alt.strip() and not text:
                # Prefer not to use reasoning as the answer.
                pass
        return ""
    # Defensive: drop <think>...</think> / similar if a model leaks them.
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.I).strip()
    text = re.sub(r"<reasoning>[\s\S]*?</reasoning>", "", text, flags=re.I).strip()
    return text


def chat(model: str, messages: list, max_tokens: int = 400,
         temperature: float = 0.6, response_format: dict | None = None,
         extra_body: dict | None = None,
         retries: int = 1, rate_limit_retries: int = 10) -> str:
    """Call chat.completions.

    Always sends reasoning_effort=none by default (minimax/kimi reasoning
    models). On 429: sleep + retry. On 400: strip optional params and retry
    bare-bones once. Raises on final failure — caller decides fallback.
    """
    body = dict(DEFAULT_EXTRA_BODY)
    if extra_body:
        body.update(extra_body)

    kwargs: dict = dict(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body=body,
    )
    if response_format:
        kwargs["response_format"] = response_format

    last_err = None
    soft_left = retries
    rate_left = rate_limit_retries
    rate_attempt = 0

    while True:
        try:
            resp = client().chat.completions.create(**kwargs)
            _record_usage(resp)
            text = _message_text(resp)
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
                # Keep reasoning_effort if possible; only drop if it was blamed
                if "reasoning" in msg.lower():
                    kwargs.pop("extra_body", None)
            if soft_left <= 0:
                break
            soft_left -= 1
            time.sleep(1.0)

    raise last_err
