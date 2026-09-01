"""Load API keys from `.env` and call OpenAI / Gemini with minimal helpers."""

from __future__ import annotations

import os
import sys
import warnings
import base64
from pathlib import Path
from typing import Any, List, Mapping, Optional

# Avoid importing urllib3 here (import can emit this warning). Match by message instead.
warnings.filterwarnings(
    "ignore",
    message=r"urllib3 v2 only supports OpenSSL 1\.1\.1\+.*",
)

# Default model ids as requested (API strings may differ slightly from marketing names).
DEFAULT_OPENAI_MODEL = "gpt-5.4-nano"
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite-preview"


def _scripts_dir() -> Path:
    return Path(__file__).resolve().parent


def _gemini_thinking_config(
    *,
    thinking_level: Optional[str] = None,
    thinking_budget: Optional[int] = None,
    include_thoughts: Optional[bool] = None,
) -> Any:
    """Build ``types.ThinkingConfig`` for :class:`google.genai.types.GenerateContentConfig`.

    If ``thinking_level`` is set (e.g. ``\"high\"``), it is preferred over
    ``thinking_budget`` / ``include_thoughts`` and matches the Gemini API style::

        ThinkingConfig(thinking_level=ThinkingLevel('high'))
    """
    try:
        from google.genai import types
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Install the Google Gen AI SDK: pip install google-genai"
        ) from e

    if thinking_level is not None and str(thinking_level).strip() != "":
        tl = str(thinking_level).strip()
        return types.ThinkingConfig(
            thinking_level=types.ThinkingLevel(tl),
        )
    if thinking_budget is not None or include_thoughts is not None:
        return types.ThinkingConfig(
            thinking_budget=thinking_budget,
            include_thoughts=include_thoughts,
        )
    return None


def load_env(dotenv_path: Optional[Path] = None) -> None:
    """Load `OPENAI_API_KEY` / `GEMINI_API_KEY` from `.env` if python-dotenv is installed.

    Search order when ``dotenv_path`` is omitted:
    1. Repository root: ``<repo>/.env``
    2. This directory: ``scripts/.env``

    Later files only fill keys not already set (``override=False``).
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    if dotenv_path is not None:
        if dotenv_path.is_file():
            load_dotenv(dotenv_path, override=False)
        return
    here = _scripts_dir()
    for path in (here.parent / ".env", here / ".env"):
        if path.is_file():
            load_dotenv(path, override=False)


def _require_key(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        raise ValueError(
            f"Missing {name}. Set it in `.env` or the environment before calling the API."
        )
    return value


def _extract_text_from_gemini_response(response: Any) -> str:
    """Extract response text from Gemini SDK response object."""
    text = getattr(response, "text", None)
    if text:
        return text

    parts: List[str] = []
    for cand in getattr(response, "candidates", []) or []:
        content = getattr(cand, "content", None)
        for p in getattr(content, "parts", []) or []:
            t = getattr(p, "text", None)
            if t:
                parts.append(t)
    return "".join(parts)


def _extract_usage_from_openai_response(response: Any) -> dict:
    """Extract usage from OpenAI ChatCompletion. Aligns loosely with Gemini key names."""
    usage = getattr(response, "usage", None)
    if not usage:
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "reasoning_tokens": None,
        }
    ctd = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(ctd, "reasoning_tokens", None) if ctd is not None else None
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "reasoning_tokens": reasoning,
    }


def _extract_usage_from_gemini_response(response: Any) -> dict:
    """Extract key usage fields from Gemini response. Missing fields become None."""
    usage = getattr(response, "usage_metadata", None)
    return {
        "prompt_token_count": getattr(usage, "prompt_token_count", None),
        "candidates_token_count": getattr(usage, "candidates_token_count", None),
        "total_token_count": getattr(usage, "total_token_count", None),
        "thoughts_token_count": getattr(usage, "thoughts_token_count", None),
    }


def call_openai_with_usage(
    messages: List[Mapping[str, Any]],
    *,
    model: str = DEFAULT_OPENAI_MODEL,
    temperature: float = 0.0,
    extra_create_params: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Call OpenAI Chat Completions; return assistant text plus ``usage`` dict.

    Unlike Gemini's ``count_tokens``, OpenAI does not expose a separate pre-flight
    multimodal token counter in this stack: token counts come from the completion
    response's ``usage`` field.

    ``usage`` includes at least ``prompt_tokens``, ``completion_tokens``,
    ``total_tokens``. Reasoning models may set ``reasoning_tokens`` inside
    ``completion_tokens_details`` (surfaced here as ``reasoning_tokens``).

    ``extra_create_params`` are merged into ``client.chat.completions.create``,
    e.g. ``{"modalities": ["text"]}`` for audio-in / text-out.
    """
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Install the OpenAI SDK: pip install openai"
        ) from e

    load_env()
    client = OpenAI(api_key=_require_key("OPENAI_API_KEY"))
    params: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "temperature": temperature,
    }
    if extra_create_params:
        params.update(dict(extra_create_params))
    response = client.chat.completions.create(**params)
    choice = response.choices[0].message
    content = choice.content
    if content is None:
        text = ""
    elif isinstance(content, str):
        text = content
    else:
        text = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in (content if isinstance(content, list) else [content])
        )
    return {"text": text, "usage": _extract_usage_from_openai_response(response)}


def call_openai(
    messages: List[Mapping[str, Any]],
    *,
    model: str = DEFAULT_OPENAI_MODEL,
    temperature: float = 0.0,
) -> str:
    """Call OpenAI Chat Completions and return the assistant text."""
    return call_openai_with_usage(
        messages, model=model, temperature=temperature
    )["text"]


# Transient-failure policy for the per-QUERY calls (parser, predicate matcher).
_RETRY_ATTEMPTS = 4
_RETRY_BASE_S = 2.0


def call_gemini_with_usage(
    prompt: str,
    *,
    model: str = DEFAULT_GEMINI_MODEL,
    temperature: float = 0.0,
    thinking_level: Optional[str] = None,
    thinking_budget: Optional[int] = None,
    include_thoughts: Optional[bool] = None,
) -> dict:
    """Call Gemini (text) and return text + usage metadata."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Install the Google Gen AI SDK: pip install google-genai"
        ) from e

    load_env()
    client = genai.Client(api_key=_require_key("GEMINI_API_KEY"))
    thinking_config = _gemini_thinking_config(
        thinking_level=thinking_level,
        thinking_budget=thinking_budget,
        include_thoughts=include_thoughts,
    )
    # Retry transient server-side failures. This path carries the PARSER and the
    # predicate matcher -- one call per query, so a 503 here kills a whole query
    # (and, in a multi-repetition run, a whole repetition) for a fault that clears
    # in a second. The per-row cascade path already retries (MAX_ATTEMPTS in
    # cascade_live); this one did not, and a single 503 aborted repetition 2 of a
    # 5-repetition run at Q8.
    # Only retry SERVER errors: a 4xx (bad key, bad request) will not fix itself
    # and must surface immediately.
    import time as _time
    last = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            response = client.models.generate_content(
                model=model,
                contents=types.Part.from_text(text=prompt),
                config=types.GenerateContentConfig(
                    temperature=temperature, thinking_config=thinking_config
                ),
            )
            break
        except Exception as exc:                                  # noqa: BLE001
            code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            retryable = code in (429, 500, 502, 503, 504) or "UNAVAILABLE" in str(exc)
            if not retryable or attempt == _RETRY_ATTEMPTS - 1:
                raise
            last = exc
            wait = _RETRY_BASE_S * (2 ** attempt)
            print(f"[connect_api] {code or type(exc).__name__} on attempt "
                  f"{attempt+1}/{_RETRY_ATTEMPTS}; retrying in {wait:.1f}s",
                  file=sys.stderr)
            _time.sleep(wait)
    return {
        "text": _extract_text_from_gemini_response(response),
        "usage": _extract_usage_from_gemini_response(response),
    }


def call_gemini_with_parts_and_usage(
    parts: List[Any],
    *,
    model: str = DEFAULT_GEMINI_MODEL,
    temperature: float = 0.0,
    thinking_level: Optional[str] = None,
    thinking_budget: Optional[int] = None,
    include_thoughts: Optional[bool] = None,
) -> dict:
    """Call Gemini with a pre-built list of ``types.Part`` (text, images, etc.)."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Install the Google Gen AI SDK: pip install google-genai"
        ) from e

    if not parts:
        raise ValueError("parts must be non-empty")

    load_env()
    client = genai.Client(api_key=_require_key("GEMINI_API_KEY"))
    thinking_config = _gemini_thinking_config(
        thinking_level=thinking_level,
        thinking_budget=thinking_budget,
        include_thoughts=include_thoughts,
    )
    response = client.models.generate_content(
        model=model,
        contents=parts,
        config=types.GenerateContentConfig(
            temperature=temperature, thinking_config=thinking_config
        ),
    )
    return {
        "text": _extract_text_from_gemini_response(response),
        "usage": _extract_usage_from_gemini_response(response),
    }


def call_gemini(
    prompt: str,
    *,
    model: str = DEFAULT_GEMINI_MODEL,
    temperature: float = 0.0,
) -> str:
    """Call Google Gemini via the unified ``google-genai`` SDK (not deprecated ``google.generativeai``)."""
    result = call_gemini_with_usage(
        prompt,
        model=model,
        temperature=temperature,
    )
    return result["text"]


def _image_mime_type(path: Path) -> str:
    """Best-effort MIME type from file extension for Gemini image parts."""
    suf = path.suffix.lower()
    if suf in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suf == ".png":
        return "image/png"
    if suf == ".webp":
        return "image/webp"
    if suf == ".gif":
        return "image/gif"
    return "image/jpeg"


def _openai_audio_input_format(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".mp3":
        return "mp3"
    return "wav"


def call_openai_with_image_and_usage(
    prompt: str,
    image_path: str,
    *,
    model: str = DEFAULT_OPENAI_MODEL,
    temperature: float = 0.0,
) -> dict:
    """Chat Completions with text + local image (data URL); returns text + usage."""
    image_file = Path(image_path)
    if not image_file.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    mime = _image_mime_type(image_file)
    b64 = base64.standard_b64encode(image_file.read_bytes()).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"
    messages: List[Mapping[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]
    return call_openai_with_usage(messages, model=model, temperature=temperature)


def call_openai_with_audio_and_usage(
    prompt: str,
    audio_path: str,
    *,
    model: str = DEFAULT_OPENAI_MODEL,
    temperature: float = 0.0,
) -> dict:
    """Chat Completions with text + local audio (base64 ``input_audio``); returns text + usage."""
    audio_file = Path(audio_path)
    if not audio_file.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    fmt = _openai_audio_input_format(audio_file)
    b64 = base64.standard_b64encode(audio_file.read_bytes()).decode("ascii")
    messages: List[Mapping[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "input_audio",
                    "input_audio": {"data": b64, "format": fmt},
                },
            ],
        }
    ]
    # Do not pass ``modalities`` here: many chat models reject it while still
    # accepting ``input_audio`` (text-only assistant output remains the default).
    return call_openai_with_usage(messages, model=model, temperature=temperature)


def call_gemini_with_image_and_usage(
    prompt: str,
    image_path: str,
    *,
    model: str = DEFAULT_GEMINI_MODEL,
    temperature: float = 0.0,
    thinking_level: Optional[str] = None,
    thinking_budget: Optional[int] = None,
    include_thoughts: Optional[bool] = None,
) -> dict:
    """Call Gemini with text + local image file and return text + usage."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Install the Google Gen AI SDK: pip install google-genai"
        ) from e

    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    load_env()
    client = genai.Client(api_key=_require_key("GEMINI_API_KEY"))
    image_bytes = image_file.read_bytes()
    mime_type = _image_mime_type(image_file)
    thinking_config = _gemini_thinking_config(
        thinking_level=thinking_level,
        thinking_budget=thinking_budget,
        include_thoughts=include_thoughts,
    )

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ],
        config=types.GenerateContentConfig(
            temperature=temperature, thinking_config=thinking_config
        ),
    )
    return {
        "text": _extract_text_from_gemini_response(response),
        "usage": _extract_usage_from_gemini_response(response),
    }


def call_gemini_with_image(
    prompt: str,
    image_path: str,
    *,
    model: str = DEFAULT_GEMINI_MODEL,
    temperature: float = 0.0,
) -> str:
    """Call Gemini with text + local image file and return text only."""
    result = call_gemini_with_image_and_usage(
        prompt, image_path, model=model, temperature=temperature
    )
    return result["text"]


def call_gemini_with_audio_and_usage(
    prompt: str,
    audio_path: str,
    *,
    model: str = DEFAULT_GEMINI_MODEL,
    temperature: float = 0.0,
    thinking_level: Optional[str] = None,
    thinking_budget: Optional[int] = None,
    include_thoughts: Optional[bool] = None,
) -> dict:
    """Call Gemini with text + local audio file and return text + usage."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Install the Google Gen AI SDK: pip install google-genai"
        ) from e

    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    load_env()
    client = genai.Client(api_key=_require_key("GEMINI_API_KEY"))
    audio_bytes = audio_file.read_bytes()
    thinking_config = _gemini_thinking_config(
        thinking_level=thinking_level,
        thinking_budget=thinking_budget,
        include_thoughts=include_thoughts,
    )

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
        ],
        config=types.GenerateContentConfig(
            temperature=temperature, thinking_config=thinking_config
        ),
    )
    return {
        "text": _extract_text_from_gemini_response(response),
        "usage": _extract_usage_from_gemini_response(response),
    }


def call_gemini_with_audio(
    prompt: str,
    audio_path: str,
    *,
    model: str = DEFAULT_GEMINI_MODEL,
    temperature: float = 0.0,
) -> str:
    """Call Gemini with text + local audio file and return text only."""
    result = call_gemini_with_audio_and_usage(
        prompt, audio_path, model=model, temperature=temperature
    )
    return result["text"]


def test_api_connections() -> dict:
    """Smoke-test both providers with a tiny prompt. Run: `python connect_api.py`"""
    load_env()
    results: dict = {"openai": None, "gemini": None, "errors": []}

    try:
        results["openai"] = call_openai(
            [{"role": "user", "content": "Reply with exactly: OK"}],
            model=DEFAULT_OPENAI_MODEL,
        ).strip()
    except Exception as e:  # pragma: no cover
        results["errors"].append(f"openai: {e}")

    try:
        results["gemini"] = call_gemini(
            "Reply with exactly: OK",
            model=DEFAULT_GEMINI_MODEL,
        ).strip()
    except Exception as e:  # pragma: no cover
        results["errors"].append(f"gemini: {e}")

    return results


def demo_capital_question() -> dict:
    """End-to-end demo: ask both providers a visible factual question."""
    load_env()
    question = "What is the capital of Australia?"
    results: dict = {"question": question, "openai": None, "gemini": None, "errors": []}

    try:
        results["openai"] = call_openai(
            [{"role": "user", "content": question}],
            model=DEFAULT_OPENAI_MODEL,
        ).strip()
    except Exception as e:  # pragma: no cover
        results["errors"].append(f"openai: {e}")

    try:
        results["gemini"] = call_gemini(
            question,
            model=DEFAULT_GEMINI_MODEL,
        ).strip()
    except Exception as e:  # pragma: no cover
        results["errors"].append(f"gemini: {e}")

    return results


def _silence_known_env_warnings() -> None:
    """Reduce noisy warnings when running this file as a script (API still works)."""
    # google-auth: Python 3.9 is past EOL — upgrade Python for long-term support.
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r".*Python version 3\.9 past its end of life.*",
    )


if __name__ == "__main__":
    _silence_known_env_warnings()
    smoke = test_api_connections()
    demo = demo_capital_question()
    print("smoke_test:", smoke)
    print("demo_capital_question:", demo)
