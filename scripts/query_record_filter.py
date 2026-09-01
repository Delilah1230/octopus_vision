#!/usr/bin/env python3
"""
Single-product LLM judge: does this Amazon listing (+ reviews) satisfy a QUERY?

Input is one text block (listing + reviews; image fields are URLs inlined in text).
The model answers in **plain text**: first line ``true`` or ``false``; optional second line
with justification (≤99 characters). The CLI still prints a small JSON object with
``matches``, ``evidence``, and ``parent_asin`` (the latter comes from the bundle, not the model).

Use ``parent_asin`` as the stable product id (unique per product in McAuley Amazon Reviews 2023;
review rows may use a different ``asin`` when variants exist).

Requires:
  pip install openai google-genai python-dotenv

API keys: repo-root ``.env`` or ``scripts/.env`` (see ``connect_api.load_env``).

Example bundle JSON::

  {
    "meta": { ... one meta_* row ... },
    "reviews": [ ... review rows for same parent_asin ... ]
  }

  python scripts/query_record_filter.py \\
    --provider openai --model gpt-4o-mini \\
    --query "..." \\
    --bundle-json product_bundle.json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from connect_api import (
    call_gemini_with_usage,
    call_openai_with_usage,
    load_env,
    _require_key,
    _extract_usage_from_gemini_response,
)

Provider = Literal["openai", "gemini"]

EVIDENCE_MAX_LEN = 99

_IMAGE_PROMPT = (
    "You are a strict relevance judge. You are shown one product image and a QUERY about that product. "
    "Answer based only on what is visible in the image. If uncertain, answer false.\n\n"
    "Response format:\n"
    "- Output exactly one word: `true` or `false` (lowercase).\n"
    "- Do NOT include any explanation, justification, or extra text.\n"
    "- Plain text only. No JSON, no markdown fences."
)

RESPONSE_FORMAT_FOOTER = "Answer with exactly one word: `true` or `false`. No explanation."

_SYSTEM_PROMPT = """You are a strict relevance judge for one Amazon product.

You receive exactly one product: listing fields and a list of customer reviews (text and optional image URLs inlined). You also receive a QUERY.

Rules:
- Base your decision only on the provided title, features, review texts, and any image URLs in this request. Do not invent facts from the open web or prior knowledge about the brand.
- If the QUERY is only weakly related or ambiguous, answer false (be conservative).

Response format:
- Output exactly one word: `true` or `false` (lowercase).
- Do NOT include any explanation, justification, or extra text.
- Plain text only. No JSON, no markdown fences."""


def _extract_json_object(raw: str) -> dict[str, Any]:
    s = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", s)
    if fence:
        s = fence.group(1).strip()
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"No JSON object in model output: {raw[:800]!r}")
    return json.loads(s[start : end + 1])


def _meta_main_image_url(meta: Mapping[str, Any]) -> str | None:
    imgs = meta.get("images")
    if not isinstance(imgs, list):
        return None
    best_non_main: str | None = None
    for it in imgs:
        if not isinstance(it, dict):
            continue
        variant = it.get("variant")
        url = it.get("large") or it.get("hi_res") or it.get("thumb")
        if isinstance(url, str) and url.strip():
            if variant == "MAIN":
                return url.strip()
            if best_non_main is None:
                best_non_main = url.strip()
    return best_non_main


def _review_small_url(rev: Mapping[str, Any]) -> str | None:
    imgs = rev.get("images")
    if not isinstance(imgs, list) or not imgs:
        return None
    first = imgs[0]
    if not isinstance(first, dict):
        return None
    u = first.get("small_image_url") or first.get("medium_image_url") or first.get("large_image_url")
    return u.strip() if isinstance(u, str) and u.strip() else None


def format_amazon_product_block(
    meta: Mapping[str, Any],
    reviews: Sequence[Mapping[str, Any]],
    *,
    max_feature_bullets: int = 12,
    max_review_chars: int = 1200,
) -> tuple[str, str]:
    """
    Build the single text block for the prompt. Returns (text, parent_asin).

    ``parent_asin`` is taken from ``meta``; raises if missing.
    """
    pid = meta.get("parent_asin")
    if not isinstance(pid, str) or not pid.strip():
        raise ValueError("meta must contain a non-empty string parent_asin")
    pid = pid.strip()

    title = meta.get("title")
    title_s = title.strip() if isinstance(title, str) else "(no title)"

    lines: list[str] = [
        f"PRODUCT_ID (parent_asin): {pid}",
        "",
        "## PRODUCT LISTING",
        f"Title: {title_s}",
        "",
    ]

    feats = meta.get("features")
    if isinstance(feats, list) and feats:
        lines.append("Features:")
        for i, f in enumerate(feats[:max_feature_bullets]):
            if isinstance(f, str) and f.strip():
                lines.append(f"- {f.strip()}")
        if len(feats) > max_feature_bullets:
            lines.append(f"- … ({len(feats) - max_feature_bullets} more omitted)")
        lines.append("")
    else:
        lines.append("Features: (none)")
        lines.append("")

    main_img = _meta_main_image_url(meta)
    lines.append(f"Main product image URL (listing): {main_img or '(none)'}")
    lines.append("")
    lines.append("## CUSTOMER REVIEWS")
    lines.append("")

    if not reviews:
        lines.append("(no reviews in this bundle)")
    else:
        for j, rev in enumerate(reviews, start=1):
            rt = rev.get("title")
            rx = rev.get("text")
            lines.append(f"### Review {j}")
            lines.append(f"Review title: {rt.strip() if isinstance(rt, str) else '(none)'}")
            body = rx.strip() if isinstance(rx, str) else ""
            if len(body) > max_review_chars:
                body = body[: max_review_chars - 3] + "..."
            lines.append(f"Review text: {body or '(empty)'}")
            su = _review_small_url(rev)
            lines.append(f"Review image (small) URL: {su or '(none)'}")
            lines.append("")

    return "\n".join(lines), pid


@dataclass
class ProductQueryResult:
    matches: bool
    evidence: str
    parent_asin: str
    raw_response: str
    usage: dict[str, Any] | None
    model_parent_asin: str | None


def _parse_match_response(raw: str) -> tuple[bool, str, str | None]:
    """Parse model output: preferred plain `true`/`false` (+ optional justification); fallback JSON."""
    stripped = raw.strip()
    if "{" in stripped:
        try:
            obj = _extract_json_object(raw)
            if isinstance(obj, dict):
                m = obj.get("matches")
                if isinstance(m, str):
                    matches = m.strip().lower() in ("true", "1", "yes")
                else:
                    matches = bool(m)
                ev = obj.get("evidence", obj.get("justification", ""))
                if ev is None:
                    ev = ""
                evidence = str(ev).strip().replace("\n", " ")
                if len(evidence) > EVIDENCE_MAX_LEN:
                    evidence = evidence[: EVIDENCE_MAX_LEN - 1] + "…"
                model_pa = obj.get("parent_asin")
                model_pa_s = (
                    model_pa.strip() if isinstance(model_pa, str) else None
                )
                return matches, evidence, model_pa_s
        except (ValueError, json.JSONDecodeError):
            pass

    # Plain text: first token true/false; rest is justification
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"Empty model response: {raw[:400]!r}")
    first = lines[0]
    parts = first.split(None, 1)
    head = parts[0].lower().rstrip(".,;:")
    if head not in ("true", "false"):
        raise ValueError(
            f"First line must start with true or false; got: {first[:200]!r}"
        )
    matches = head == "true"
    chunks: list[str] = []
    if len(parts) > 1:
        chunks.append(parts[1])
    chunks.extend(lines[1:])
    evidence = " ".join(chunks).strip().replace("\n", " ")
    if len(evidence) > EVIDENCE_MAX_LEN:
        evidence = evidence[: EVIDENCE_MAX_LEN - 1] + "…"
    return matches, evidence, None


def evaluate_product_image_against_query(
    query: str,
    image_url: str,
    parent_asin: str,
    *,
    provider: Provider,
    model: str,
    temperature: float = 0.0,
    timeout_s: int = 15,
) -> ProductQueryResult:
    """
    Judge whether a product matches the QUERY using only its image.

    Downloads image bytes from `image_url` and sends them to Gemini as an
    image Part alongside the text prompt. OpenAI is not supported here yet.
    """
    if provider != "gemini":
        raise NotImplementedError(
            f"Image evaluator currently only supports Gemini, got provider={provider!r}"
        )
    import requests
    from google import genai
    from google.genai import types

    resp = requests.get(image_url, timeout=timeout_s)
    resp.raise_for_status()
    image_bytes = resp.content
    # Infer mime from URL extension; default jpeg
    url_lower = image_url.lower().split("?")[0]
    if url_lower.endswith(".png"):
        mime_type = "image/png"
    elif url_lower.endswith(".webp"):
        mime_type = "image/webp"
    elif url_lower.endswith(".gif"):
        mime_type = "image/gif"
    else:
        mime_type = "image/jpeg"

    load_env()
    from connect_api import _require_key   # local import to avoid top-level cycle
    client = genai.Client(api_key=_require_key("GEMINI_API_KEY"))

    full_prompt = f"{_IMAGE_PROMPT}\n\n## QUERY\n\n{query.strip()}"
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_text(text=full_prompt),
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ],
        config=types.GenerateContentConfig(temperature=temperature),
    )

    from connect_api import (
        _extract_text_from_gemini_response,
        _extract_usage_from_gemini_response,
    )
    raw   = _extract_text_from_gemini_response(response) or ""
    usage = _extract_usage_from_gemini_response(response)
    matches, evidence, model_pa = _parse_match_response(raw)
    return ProductQueryResult(
        matches=matches,
        evidence=evidence,
        parent_asin=parent_asin.strip(),
        raw_response=raw,
        usage=usage,
        model_parent_asin=model_pa,
    )


def evaluate_product_against_query(
    query: str,
    product_context_text: str,
    parent_asin: str,
    *,
    provider: Provider,
    model: str,
    temperature: float = 0.0,
) -> ProductQueryResult:
    """Call LLM; canonical ``parent_asin`` in the result is the one you passed in (not the model's)."""
    load_env()
    user_body = (
        f"{product_context_text}\n\n"
        "## QUERY\n\n"
        f"{query.strip()}\n\n"
        f"{RESPONSE_FORMAT_FOOTER}"
    )

    if provider == "openai":
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_body},
        ]
        resp = call_openai_with_usage(messages, model=model, temperature=temperature)
        raw = resp.get("text") or ""
        usage = resp.get("usage")
    else:
        prompt = _SYSTEM_PROMPT + "\n\n---\n\n" + user_body
        resp = call_gemini_with_usage(prompt, model=model, temperature=temperature)
        raw = resp.get("text") or ""
        usage = resp.get("usage")

    matches, evidence, model_pa = _parse_match_response(raw)
    return ProductQueryResult(
        matches=matches,
        evidence=evidence,
        parent_asin=parent_asin.strip(),
        raw_response=raw,
        usage=usage,
        model_parent_asin=model_pa,
    )


# ── batched (multi-predicate) evaluators ──────────────────────────────────────
# One LLM call per (asin, modality, model) carrying ALL predicates at once.
# Returns dict[pred_idx → bool] plus usage. Format compliance is enforced by
# Gemini's response_schema (grammar-constrained decoding).

# Default timeout for a single batched generate_content call. If exceeded,
# the call is abandoned and re-issued up to MAX_RETRIES_DEFAULT times.
# (The underlying HTTP request is not actually cancelled by the SDK — we just
# stop waiting and re-submit a new thread. Orphan responses are discarded.)
DEFAULT_TIMEOUT_S = 15
DEFAULT_MAX_RETRIES = 1


def _generate_with_timeout(
    client, *,
    model: str,
    contents,
    config,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    max_retries: int = DEFAULT_MAX_RETRIES,
    label: str = "",
):
    """
    Run client.models.generate_content with a hard wall-clock deadline.
    On timeout: abandon the wait, retry up to `max_retries` times, then raise
    `concurrent.futures.TimeoutError` to the caller.

    Note: the SDK call itself is not interrupted at the network layer — we
    just stop waiting and start a new thread. The previous response, if it
    eventually arrives, is silently discarded.
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

    last_exc = None
    for attempt in range(max_retries + 1):
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            client.models.generate_content,
            model=model, contents=contents, config=config,
        )
        try:
            response = future.result(timeout=timeout_s)
            executor.shutdown(wait=False)
            if attempt > 0:
                print(f"    [timeout-retry OK] {label} {model} succeeded on attempt {attempt+1}")
            return response
        except FutureTimeoutError as exc:
            executor.shutdown(wait=False)
            last_exc = exc
            if attempt < max_retries:
                print(f"    [timeout {timeout_s}s] {label} {model}, retrying "
                      f"({attempt+1}/{max_retries+1})...")
            else:
                print(f"    [timeout {timeout_s}s] {label} {model}, "
                      f"max retries exceeded — raising")
                raise

# SLIM schema — only the bare minimum. Dropping `confidence` (no consumer)
# and `evidence` (forces the model to over-rationalize, which shifts the
# answer distribution and inflates output tokens).
_PRED_BATCHED_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "predicate_idx": {"type": "integer"},
                    "answer":        {"type": "boolean"},
                },
                "required": ["predicate_idx", "answer"],
            },
        },
    },
    "required": ["answers"],
}


_PRED_BATCHED_INSTRUCTION = """\
You are evaluating ONE product against multiple yes/no questions at the same
time. Answer EACH question independently and strictly True or False — do NOT
let any answer affect another.

Output ONLY the schema-conformant JSON: a list of {predicate_idx, answer}
objects, one per question, in any order. No explanation, no extra fields,
no extra text.
"""


def _build_pred_batched_text_prompt(
    product_context_text: str,
    predicates: list,    # list of objects with .agent_prompt
    predicate_indices: list[int],
) -> str:
    questions_block = "\n".join(
        f"  Q{idx}: {p.agent_prompt}"
        for idx, p in zip(predicate_indices, predicates)
    )
    return (
        f"{_PRED_BATCHED_INSTRUCTION}\n"
        f"PRODUCT CONTEXT:\n{product_context_text}\n\n"
        f"QUESTIONS (answer all):\n{questions_block}"
    )


def _build_pred_batched_image_prompt(
    query_intro: str,
    predicates: list,
    predicate_indices: list[int],
) -> str:
    questions_block = "\n".join(
        f"  Q{idx}: {p.agent_prompt}"
        for idx, p in zip(predicate_indices, predicates)
    )
    return (
        f"{_PRED_BATCHED_INSTRUCTION}\n"
        f"{query_intro}\n\n"
        f"QUESTIONS (answer all, evaluating the IMAGE):\n{questions_block}"
    )


def _parse_pred_batched_response(raw_text: str,
                            expected_pred_indices: set[int]) -> tuple[dict[int, bool], list[str]]:
    """Parse JSON-schema'd output. Returns (pred_idx → bool, errors)."""
    answers: dict[int, bool] = {}
    errors: list[str] = []
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return {}, [f"JSON parse failure: {exc}"]
    items = parsed.get("answers", [])
    seen: set[int] = set()
    for item in items:
        if not isinstance(item, dict):
            errors.append(f"answer item not an object: {item!r}")
            continue
        idx = item.get("predicate_idx")
        if idx in seen:
            errors.append(f"duplicate predicate_idx={idx}")
            continue
        seen.add(idx)
        if idx not in expected_pred_indices:
            errors.append(f"unexpected predicate_idx={idx} "
                          f"(valid: {sorted(expected_pred_indices)})")
            continue
        ans = item.get("answer")
        if not isinstance(ans, bool):
            errors.append(f"predicate_idx={idx}: answer not bool: {ans!r}")
            continue
        answers[idx] = ans
    missing = expected_pred_indices - seen
    if missing:
        errors.append(f"missing answers for predicate_idxs={sorted(missing)}")
    return answers, errors


def evaluate_predicate_batched_text(
    product_context_text: str,
    predicates: list,
    predicate_indices: list[int],
    parent_asin: str,
    *,
    model: str,
    temperature: float = 0.0,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict:
    """
    Send ONE Gemini call evaluating all `predicates` on the product's text
    block. Returns:
        {"answers": dict[int, bool],
         "usage": dict,
         "raw_response": str,
         "errors": list[str]}
    On parse / validation failure `answers` may be partial; caller can decide
    to retry per-predicate.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:  # pragma: no cover
        raise ImportError("Install google-genai") from e

    load_env()
    client = genai.Client(api_key=_require_key("GEMINI_API_KEY"))

    prompt = _build_pred_batched_text_prompt(product_context_text, predicates, predicate_indices)
    response = _generate_with_timeout(
        client,
        model=model,
        contents=types.Part.from_text(text=prompt),
        config=types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=_PRED_BATCHED_RESPONSE_SCHEMA,
        ),
        timeout_s=timeout_s,
        max_retries=max_retries,
        label=f"pred-batched-text {parent_asin}",
    )
    raw   = getattr(response, "text", "") or ""
    usage = _extract_usage_from_gemini_response(response)
    answers, errors = _parse_pred_batched_response(raw, set(predicate_indices))
    return {
        "answers":      answers,
        "usage":        usage,
        "raw_response": raw,
        "errors":       errors,
        "parent_asin":  parent_asin.strip(),
    }


# ── megabatch: N products × M predicates in ONE call ─────────────────────────

# Schema for asin-batched: per (asin, predicate, modality). The model
# answers each combination separately, keeping the modality split.
_ASIN_BATCHED_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "parent_asin":   {"type": "string"},
                    "predicate_idx": {"type": "integer"},
                    "modality":      {"type": "string", "enum": ["text", "image"]},
                    "answer":        {"type": "boolean"},
                },
                "required": ["parent_asin", "predicate_idx", "modality", "answer"],
            },
        },
    },
    "required": ["results"],
}


_ASIN_BATCHED_INSTRUCTION = """\
You are evaluating MULTIPLE products against MULTIPLE yes/no questions in a
single call. Each product has BOTH text context AND (when available) an
image. For EACH (product, question) you must give TWO separate answers:
  - one based on the TEXT context only  (modality="text")
  - one based on the IMAGE only          (modality="image")

If a product has no image, give only its text-modality answers.
DO NOT let any one answer affect any other.

Each product is identified by its `parent_asin`. Each question is
identified by its `predicate_idx`. Your output is a flat array of result
objects, one per (parent_asin, predicate_idx, modality) tuple.

Output ONLY the schema-conformant JSON: a list of
{parent_asin, predicate_idx, modality, answer} objects. No explanation,
no extra fields, no extra text.
"""


def _parse_asin_batched_response(
    raw_text: str,
    expected_asins: set[str],
    expected_pred_indices: set[int],
    expected_modality_by_asin: dict[str, set[str]],
) -> tuple[dict[tuple[str, int, str], bool], list[str]]:
    """Parse JSON output. Returns ({(asin, pred_idx, modality) → bool}, errors).

    `expected_modality_by_asin` says which modalities each asin should produce
    answers for (e.g., asins without images only produce text-modality).
    """
    answers: dict[tuple[str, int, str], bool] = {}
    errors: list[str] = []
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return {}, [f"JSON parse failure: {exc}"]
    items = parsed.get("results", [])
    seen: set[tuple[str, int, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            errors.append(f"item not an object: {item!r}")
            continue
        asin = item.get("parent_asin")
        idx  = item.get("predicate_idx")
        mod  = item.get("modality")
        key  = (asin, idx, mod)
        if key in seen:
            errors.append(f"duplicate {key!r}")
            continue
        seen.add(key)
        if asin not in expected_asins:
            errors.append(f"unexpected parent_asin={asin!r}")
            continue
        if idx not in expected_pred_indices:
            errors.append(f"unexpected predicate_idx={idx}")
            continue
        if mod not in expected_modality_by_asin.get(asin, set()):
            errors.append(f"{key!r}: modality not expected for this asin")
            continue
        ans = item.get("answer")
        if not isinstance(ans, bool):
            errors.append(f"{key!r}: answer not bool: {ans!r}")
            continue
        answers[key] = ans
    missing = {
        (a, i, m)
        for a in expected_asins
        for i in expected_pred_indices
        for m in expected_modality_by_asin.get(a, set())
    } - seen
    if missing:
        errors.append(
            f"missing {len(missing)} (asin, pred, modality) tuples "
            f"(sample: {sorted(missing)[:5]})"
        )
    return answers, errors


# ── pred-modality-batched: 1 call per (asin, model), text + image in one prompt ──

# Schema for pred-modality-batched: each predicate gets ONE answer per modality
# present. The model must answer EACH (predicate, modality) pair separately —
# so we keep the modality split signal that downstream profiling consumes.
_PRED_MODALITY_BATCHED_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "predicate_idx": {"type": "integer"},
                    "modality":      {"type": "string", "enum": ["text", "image"]},
                    "answer":        {"type": "boolean"},
                },
                "required": ["predicate_idx", "modality", "answer"],
            },
        },
    },
    "required": ["answers"],
}


_PRED_MODALITY_BATCHED_INSTRUCTION = """\
You are evaluating ONE product against multiple yes/no questions in a single
call. You receive BOTH the product's textual context AND its image. For
EACH question you must give TWO separate answers:
  - one based on the TEXT context only  (modality="text")
  - one based on the IMAGE only          (modality="image")

If a question cannot be answered from a given modality (e.g. an ingredient
question on the image alone), give your best Boolean using only that signal.
DO NOT let one answer affect another.

Output ONLY the schema-conformant JSON: a list of
{predicate_idx, modality, answer} objects. Provide exactly 2 × N items
(N = number of questions), one for each (predicate, modality) pair. No
explanation, no extra fields.
"""


def _parse_pred_modality_batched_response(
    raw_text: str,
    expected_pred_indices: set[int],
    expected_modalities:   set[str],
) -> tuple[dict[tuple[int, str], bool], list[str]]:
    """Parse JSON output. Returns ({(pred_idx, modality) → bool}, errors)."""
    answers: dict[tuple[int, str], bool] = {}
    errors: list[str] = []
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return {}, [f"JSON parse failure: {exc}"]
    items = parsed.get("answers", [])
    seen: set[tuple[int, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            errors.append(f"item not an object: {item!r}")
            continue
        idx = item.get("predicate_idx")
        mod = item.get("modality")
        key = (idx, mod)
        if key in seen:
            errors.append(f"duplicate {key!r}")
            continue
        seen.add(key)
        if idx not in expected_pred_indices:
            errors.append(f"unexpected predicate_idx={idx}")
            continue
        if mod not in expected_modalities:
            errors.append(f"unexpected modality={mod!r}")
            continue
        ans = item.get("answer")
        if not isinstance(ans, bool):
            errors.append(f"{key!r}: answer not bool: {ans!r}")
            continue
        answers[key] = ans
    missing = {(i, m) for i in expected_pred_indices for m in expected_modalities} - seen
    if missing:
        errors.append(f"missing {len(missing)} (pred, modality) pairs (sample: {sorted(missing)[:5]})")
    return answers, errors


def evaluate_pred_modality_batched(
    product_context_text: str,
    image_url: str | None,
    predicates: list,
    predicate_indices: list[int],
    parent_asin: str,
    *,
    model: str,
    temperature: float = 0.0,
    download_timeout_s: int = 15,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict:
    """
    ONE Gemini call that bundles BOTH text and image of a single product
    plus all predicates. The model returns ONE answer PER MODALITY per
    predicate — so for 2 predicates we expect 4 answers (text + image
    × 2 predicates). This preserves the per-modality split that
    downstream profiling consumes.

    If image_url is None or download fails, only text-modality answers
    are expected back (we drop image from the modality set).

    Returns:
      {"answers":  dict[(pred_idx, modality) → bool],
       "usage":    dict,
       "raw_response": str,
       "errors":   list[str],
       "parent_asin": str}
    """
    import requests
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:  # pragma: no cover
        raise ImportError("Install google-genai") from e

    load_env()
    client = genai.Client(api_key=_require_key("GEMINI_API_KEY"))

    questions_block = "\n".join(
        f"  Q{idx}: {p.agent_prompt}"
        for idx, p in zip(predicate_indices, predicates)
    )
    prompt_text = (
        f"{_PRED_MODALITY_BATCHED_INSTRUCTION}\n"
        f"PRODUCT CONTEXT (text):\n{product_context_text}\n\n"
        f"QUESTIONS (answer EACH in both modalities — text and image):\n"
        f"{questions_block}"
    )

    parts: list = [types.Part.from_text(text=prompt_text)]
    has_image = False

    if image_url:
        try:
            resp = requests.get(image_url, timeout=download_timeout_s)
            resp.raise_for_status()
            image_bytes = resp.content
            url_lower = image_url.lower().split("?")[0]
            if url_lower.endswith(".png"):
                mime_type = "image/png"
            elif url_lower.endswith(".webp"):
                mime_type = "image/webp"
            elif url_lower.endswith(".gif"):
                mime_type = "image/gif"
            else:
                mime_type = "image/jpeg"
            parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
            has_image = True
        except Exception as e:
            print(f"  [pred_modality_batched] {parent_asin}: image download "
                  f"failed ({e}); continuing text-only")

    expected_modalities = {"text", "image"} if has_image else {"text"}

    response = _generate_with_timeout(
        client,
        model=model,
        contents=parts,
        config=types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=_PRED_MODALITY_BATCHED_RESPONSE_SCHEMA,
        ),
        timeout_s=timeout_s,
        max_retries=max_retries,
        label=f"pred-modality-batched {parent_asin}",
    )
    raw   = getattr(response, "text", "") or ""
    usage = _extract_usage_from_gemini_response(response)
    answers, errors = _parse_pred_modality_batched_response(
        raw, set(predicate_indices), expected_modalities,
    )
    return {
        "answers":      answers,    # {(pred_idx, modality) → bool}
        "usage":        usage,
        "raw_response": raw,
        "errors":       errors,
        "parent_asin":  parent_asin.strip(),
    }


def evaluate_asin_batched(
    products_text:  dict[str, str],          # asin → product text block
    products_image: dict[str, str | None],   # asin → image_url (or None)
    predicates: list,                        # objects with .agent_prompt
    predicate_indices: list[int],
    *,
    model: str,
    temperature: float = 0.0,
    download_timeout_s: int = 15,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict:
    """
    asin-batched mode — ONE Gemini call per MODEL covering ALL products
    (text + image) × ALL predicates. The most aggressive batching mode.

    Each product's text block and image are bundled into a single multipart
    call with a `parent_asin` header. The model is asked to answer every
    (asin, predicate_idx) pair in one shot.

    Inputs:
      products_text  = {asin → text block} (must contain all asins to evaluate)
      products_image = {asin → image_url}  (optional per asin; None or missing skips image)

    Returns:
      {"answers":  dict[(asin, pred_idx) → bool],
       "usage":    dict,
       "raw_response": str,
       "errors":   list[str]}

    Notes:
      - Image bytes are inline (Part.from_bytes), so total request size is
        bounded by Gemini's 20MB inline-data cap. Practical N ≈ ≤30 with
        typical product image sizes.
      - Token-cost scales with N (text + image both linear in product count).
      - The model picks its own modality per (asin, pred) — no separate
        text-vote vs image-vote signal is recovered.
    """
    import requests
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:  # pragma: no cover
        raise ImportError("Install google-genai") from e

    load_env()
    client = genai.Client(api_key=_require_key("GEMINI_API_KEY"))

    asins = list(products_text.keys())
    questions_block = "\n".join(
        f"  Q{idx}: {p.agent_prompt}"
        for idx, p in zip(predicate_indices, predicates)
    )

    # Track which modalities are expected per asin (text always; image
    # iff download succeeds). Used by the parser to validate completeness.
    expected_modality_by_asin: dict[str, set[str]] = {a: {"text"} for a in asins}

    parts: list = []
    parts.append(types.Part.from_text(text=(
        f"{_ASIN_BATCHED_INSTRUCTION}\n"
        f"You will see {len(asins)} products. Each is introduced by its "
        f"`parent_asin` label, followed by its TEXT context and (if "
        f"available) its IMAGE. After all products, the QUESTIONS list "
        f"applies to EVERY product."
    )))
    for asin in asins:
        text_block = products_text.get(asin, "")
        image_url  = products_image.get(asin)
        parts.append(types.Part.from_text(text=(
            f"\n========== PRODUCT parent_asin = {asin} ==========\n"
            f"TEXT context:\n{text_block}"
        )))
        if image_url:
            try:
                resp = requests.get(image_url, timeout=download_timeout_s)
                resp.raise_for_status()
                image_bytes = resp.content
                url_lower = image_url.lower().split("?")[0]
                if url_lower.endswith(".png"):
                    mime_type = "image/png"
                elif url_lower.endswith(".webp"):
                    mime_type = "image/webp"
                elif url_lower.endswith(".gif"):
                    mime_type = "image/gif"
                else:
                    mime_type = "image/jpeg"
                parts.append(types.Part.from_text(
                    text=f"IMAGE for parent_asin = {asin}:"))
                parts.append(types.Part.from_bytes(
                    data=image_bytes, mime_type=mime_type))
                expected_modality_by_asin[asin].add("image")
            except Exception as e:
                print(f"  [asin_batched] {asin}: image download failed ({e}); "
                      f"continuing without image for this asin")
    # Expected tuple count = sum over asins of (n_preds × n_modalities_present)
    expected_n = sum(
        len(predicate_indices) * len(expected_modality_by_asin[a])
        for a in asins
    )
    parts.append(types.Part.from_text(text=(
        f"\n\nQUESTIONS (apply each to EVERY product, in BOTH modalities "
        f"when image is provided):\n{questions_block}\n\n"
        f"Output exactly {expected_n} result objects — one per "
        f"(parent_asin, predicate_idx, modality) tuple."
    )))

    response = _generate_with_timeout(
        client,
        model=model,
        contents=parts,
        config=types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=_ASIN_BATCHED_RESPONSE_SCHEMA,
        ),
        timeout_s=timeout_s,
        max_retries=max_retries,
        label=f"asin-batched [{len(asins)} products]",
    )
    raw   = getattr(response, "text", "") or ""
    usage = _extract_usage_from_gemini_response(response)
    answers, errors = _parse_asin_batched_response(
        raw, set(asins), set(predicate_indices), expected_modality_by_asin,
    )
    return {
        "answers":      answers,
        "usage":        usage,
        "raw_response": raw,
        "errors":       errors,
    }


def evaluate_predicate_batched_image(
    image_url: str,
    predicates: list,
    predicate_indices: list[int],
    parent_asin: str,
    *,
    model: str,
    temperature: float = 0.0,
    download_timeout_s: int = 15,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict:
    """One Gemini call evaluating all `predicates` against the product's image.

    `download_timeout_s` covers the image GET; `timeout_s` covers the LLM call
    (re-issued on timeout up to `max_retries` times).
    """
    import requests
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:  # pragma: no cover
        raise ImportError("Install google-genai") from e

    # Image download — separate timeout, not retried
    resp = requests.get(image_url, timeout=download_timeout_s)
    resp.raise_for_status()
    image_bytes = resp.content
    url_lower = image_url.lower().split("?")[0]
    if url_lower.endswith(".png"):
        mime_type = "image/png"
    elif url_lower.endswith(".webp"):
        mime_type = "image/webp"
    elif url_lower.endswith(".gif"):
        mime_type = "image/gif"
    else:
        mime_type = "image/jpeg"

    load_env()
    client = genai.Client(api_key=_require_key("GEMINI_API_KEY"))

    query_intro = (
        "You are looking at a product image. Use only the IMAGE to answer "
        "each question."
    )
    prompt = _build_pred_batched_image_prompt(query_intro, predicates, predicate_indices)
    response = _generate_with_timeout(
        client,
        model=model,
        contents=[
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ],
        config=types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=_PRED_BATCHED_RESPONSE_SCHEMA,
        ),
        timeout_s=timeout_s,
        max_retries=max_retries,
        label=f"pred-batched-image {parent_asin}",
    )
    raw   = getattr(response, "text", "") or ""
    usage = _extract_usage_from_gemini_response(response)
    answers, errors = _parse_pred_batched_response(raw, set(predicate_indices))
    return {
        "answers":      answers,
        "usage":        usage,
        "raw_response": raw,
        "errors":       errors,
        "parent_asin":  parent_asin.strip(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Single-product LLM judge: matches query? (returns matches, evidence, parent_asin)"
    )
    ap.add_argument("--query", required=True)
    ap.add_argument(
        "--bundle-json",
        type=Path,
        required=True,
        help='JSON: {"meta": {...}, "reviews": [ {...}, ... ]}',
    )
    ap.add_argument("--provider", choices=("openai", "gemini"), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print raw response and usage on stderr",
    )
    args = ap.parse_args()

    bundle = json.loads(args.bundle_json.read_text(encoding="utf-8"))
    if not isinstance(bundle, dict):
        raise SystemExit("bundle-json must be a JSON object")
    meta = bundle.get("meta")
    reviews = bundle.get("reviews")
    if not isinstance(meta, dict):
        raise SystemExit('bundle-json must contain object "meta"')
    if reviews is None:
        reviews = []
    if not isinstance(reviews, list):
        raise SystemExit('"reviews" must be an array')

    product_text, parent_asin = format_amazon_product_block(meta, reviews)
    result = evaluate_product_against_query(
        args.query,
        product_text,
        parent_asin,
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
    )

    out = {
        "matches": result.matches,
        "evidence": result.evidence,
        "parent_asin": result.parent_asin,
    }
    if args.verbose:
        import sys

        print(
            json.dumps(
                {
                    "raw_response": result.raw_response,
                    "usage": result.usage,
                    "model_parent_asin": result.model_parent_asin,
                },
                indent=2,
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
