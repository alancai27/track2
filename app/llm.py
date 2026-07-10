"""Shared Groq (OpenAI-compatible) client + hardened chat helper.

Env:
  GROQ_API_KEY   (your key locally; injected by grader if applicable)
  GROQ_BASE_URL  (default: https://api.groq.com/openai/v1)
"""
import os
import time

from openai import OpenAI

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"

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


def chat(model: str, messages: list, max_tokens: int = 400,
         temperature: float = 0.6, response_format: dict | None = None,
         retries: int = 1) -> str:
    """Call chat.completions. On 400, retry once WITHOUT optional params
    (Track 1 lesson: some models reject response_format / extra params).
    Raises on final failure — caller decides fallback."""
    kwargs = dict(model=model, messages=messages,
                  max_tokens=max_tokens, temperature=temperature)
    if response_format:
        kwargs["response_format"] = response_format

    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = client().chat.completions.create(**kwargs)
            text = (resp.choices[0].message.content or "").strip()
            if text:
                return text
            last_err = RuntimeError("empty completion")
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = str(e)
            # 400 → strip optional params and try bare-bones once
            if "400" in msg or "invalid" in msg.lower():
                kwargs.pop("response_format", None)
            if attempt < retries:
                time.sleep(1.0)
    raise last_err
