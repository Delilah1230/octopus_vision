#!/usr/bin/env python3
"""
Q1 sub-question decomposition probe (records-only, no aggregation).

Runs the cheapest model (gemini-2.5-flash-lite by default) over ALL 200 GT
products for Q1's two predicates:
    A. "is vegan"
    B. "looks indulgent enough to give as a Valentine's gift"

Each predicate is decomposed into 5 sub-questions (3 content + 2 meta). ONE
LLM call per (product, predicate) returns all 5 answers, each strictly one of
{T, F, cannot_determine} (grammar-constrained via response_schema).

This round ONLY records — it does NOT aggregate the 5 sub-answers into a
predicate-level T/F. Every sub-answer, plus per-call latency / tokens / cost /
raw response, is written to CSV for later analysis. The GT soft label
(T_rate from gt_output/gt_soft_labels.csv) is joined in as a reference column.

Product text block matches what the GT voters saw: title + features +
description only (NO reviews, NO image), so the comparison against the GT soft
labels is apples-to-apples on the text side.

Usage:
    # smoke test (3 products × 2 predicates = 6 calls)
    python scripts/subquestion_probe.py --limit 3

    # full run (200 products × 2 predicates = 400 calls)
    python scripts/subquestion_probe.py
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
from connect_api import (  # noqa: E402
    load_env,
    _require_key,
    _extract_text_from_gemini_response,
    _extract_usage_from_gemini_response,
)
from query_record_filter import (  # noqa: E402
    _generate_with_timeout, _meta_main_image_url,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
META_JSONL = (
    REPO_ROOT
    / "dataset/AmazonReviews2023/sample_3x40/Grocery_and_Gourmet_Food"
    / "meta_Grocery_and_Gourmet_Food_sample_200.jsonl"
)
GT_SOFT_LABELS = REPO_ROOT / "gt_output" / "gt_soft_labels.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "subq_probe"

# Pricing (USD per 1M tokens). pro output price includes billed thinking tokens.
PRICING = {
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    # Used for the PARSER (per query) and RELATED-QUESTION GENERATION (per
    # predicate) only -- never for answering. 5x/3x the answering model's rate,
    # which is affordable precisely because neither cost scales with the data.
    "gemini-3.6-flash": {"input": 1.50, "output": 7.50},
}

ANSWER_ENUM = ["T", "F", "cannot_determine"]

# What the answering prompts call the thing being judged. Every instruction here
# was written for Amazon listings and says "product" throughout, which is domain
# leakage on any other table: a movie review is not a product, and telling the
# model it is invites exactly the "product features / benefits" framing that made
# the first movie decomposition misfire.
#
# Set this to the table's own noun (e.g. "review") for non-Amazon data. The
# default is "product", so every prompt is BYTE-IDENTICAL to before for grocery.
RECORD_NOUN: str = "product"


def _localize(text: str) -> str:
    """Swap the record noun into a prompt written with "product"/"PRODUCT"."""
    if RECORD_NOUN == "product":
        return text
    return (text.replace("products", f"{RECORD_NOUN}s")
                .replace("product", RECORD_NOUN)
                .replace("PRODUCTS", f"{RECORD_NOUN.upper()}S")
                .replace("PRODUCT", RECORD_NOUN.upper()))

# Thinking budget for the sub-question calls. None = provider default, which
# means gemini-2.5-flash (tier 2) THINKS (~650 tok/call, ~5x latency, billed at
# the output rate) -- deliberate, see cascade_live.py:63. Set to 0 to match the
# SemBench baseline protocol, which runs every system with
# reasoning_effort="disable"; use that for any head-to-head against LOTUS /
# Palimpzest, otherwise we are comparing a thinking model against non-thinking
# ones. Runners override it at import time; nothing changes by default.
THINKING_BUDGET: int | None = None

# Chain-of-thought for the sub-question calls. OFF by default, which is what every
# number before 2026-08-01 was produced with: the schema is {sub_id, answer} and
# the instruction says "No explanation".
#
# This is the SECOND, independent knob (THINKING_BUDGET is the first). They are not
# the same thing: thinking_budget buys hidden provider-side reasoning tokens; COT
# makes the model WRITE a visible reason before committing. LOTUS and Palimpzest
# both expose exactly this distinction -- LOTUS `strategy=ReasoningStrategy.COT`
# (Reasoning:/Answer:) and PZ `PromptStrategy.COT_BOOL` (which is PZ's DEFAULT for
# filters). We had neither, so a head-to-head against PZ was comparing a no-CoT
# system against a CoT one. Runners set both at import time; nothing changes by
# default.
#
# Implementation note: CoT here is enforced by the RESPONSE SCHEMA, not by prose.
# These calls use response_mime_type="application/json" + response_schema, so the
# model cannot emit free text outside the schema -- an instruction alone would do
# nothing. We add a `reasoning` field and put it FIRST via propertyOrdering, so
# generation order is reason-then-answer. Emitting the answer first and the reason
# second is post-hoc rationalisation, not chain-of-thought, and would measure
# nothing.
COT: bool = False

# COMPACT OUTPUT — the answers come back as ONE STRING, one character per
# sub-question, in the order the sub-questions were listed:
#     {"answers": "TFT?FT"}      instead of
#     {"answers": [{"sub_id": "pos_R1", "answer": "T"}, ... x6]}
#
# Why it matters: output tokens are billed at 2.5/M and input at 0.3/M, so on the
# related-question scheme the ANSWERS dominate the bill -- measured 118 output
# tok/row (66% of per-row cost) at batch=1 and 184 (91%) at batch=6, against
# LOTUS's 3. The verbose form spends almost all of that echoing `sub_id` keys the
# prompt already fixed the order of.
#
# THE RISK IS ALIGNMENT, NOT SIZE. Dropping sub_id means the model must keep
# positional correspondence with the question list; a shifted string is wrong
# data that still parses. So the parser validates length exactly and the
# character set exactly, and the intended use is A/B against the verbose form on
# the same rows (see vision_evaluation/movie/bench_per_row.py --compare).
#
# Off by default: every existing number was produced with the verbose schema.
#
# MEASURED 2026-08-10, 300 rows, same predicate, vs a verbose-vs-verbose control:
#   control (verbose x2) : 95.7% same verdict, 92.3% same r, F1 0.639 -> 0.656
#   verbose vs compact   : 79.0% same verdict, 45.0% same r, F1 0.639 -> 0.524
# The degradation is ~7x the run-to-run noise, so the schema is NOT free
# packaging -- writing `sub_id` before each answer makes the model re-locate the
# question it is answering. "array" exists to separate the two things compact
# does at once: it drops the key echo but keeps ONE JSON ELEMENT PER QUESTION.
ANSWER_FORMAT: str = "verbose"        # "verbose" | "array" | "compact"

_FORMATS = ("verbose", "array", "compact")


def _fmt() -> str:
    if ANSWER_FORMAT not in _FORMATS:
        raise ValueError(f"ANSWER_FORMAT must be one of {_FORMATS}, got {ANSWER_FORMAT!r}")
    return ANSWER_FORMAT

# string char -> ANSWER_ENUM value
_COMPACT_MAP = {"T": "T", "F": "F", "?": "cannot_determine"}


@dataclass(frozen=True)
class LLMConfig:
    """How ONE model is invoked: which weights, and the two reasoning knobs.

    These three belong together — "gemini-2.5-flash with thinking on, no CoT" is a
    single configuration decision, and the cascade makes a DIFFERENT one per tier
    (tier 1 is a cheap non-thinking screen; tier 2 thinks on purpose). Keeping the
    knobs as module globals forced every tier to share one value and made the
    runner monkey-patch the module at import time. Carry this object instead.

    thinking_budget: None = provider default (gemini-2.5-flash thinks, ~650 tok);
                     0    = off, which is the SemBench baseline protocol.
    cot:             write a `reasoning` string before each answer (schema-enforced).
    """
    model: str
    thinking_budget: int | None = None
    cot: bool = False

    def price(self) -> dict:
        return PRICING.get(self.model, {"input": 0.0, "output": 0.0})


# Sentinel: "caller said nothing, fall back to the module global". Lets the new
# per-call arguments coexist with the old globals so existing callers are unchanged.
_UNSET = object()


def _resolve(thinking_budget, cot) -> tuple[int | None, bool]:
    tb = THINKING_BUDGET if thinking_budget is _UNSET else thinking_budget
    c = COT if cot is _UNSET else cot
    return tb, bool(c)


def _thinking_cfg(budget=_UNSET):
    """types.ThinkingConfig(...) or None. Defaults to the THINKING_BUDGET global."""
    budget = THINKING_BUDGET if budget is _UNSET else budget
    if budget is None:
        return None
    from google.genai import types as _t
    return _t.ThinkingConfig(thinking_budget=budget)




# ── sub-question definitions ─────────────────────────────────────────────────
# NOTE: the `text` sent to the model is ENGLISH. Chinese comments are for the
# maintainer only and are never included in the prompt.
#
# POLARITY CONVENTION (uniform across all sub-questions):
#   - content sub-questions: answer "T" == evidence that the PREDICATE HOLDS
#     (i.e. toward "is vegan" / toward "looks indulgent"). So every content
#     question is phrased positively toward the predicate.
#   - meta sub-questions:    answer "T" == FAVORABLE evidence state, i.e. the
#     information is sufficient (M1) and clear/unambiguous (M2).
#
# Each entry: (sub_id, sub_type in {content, meta}, english_question_text)

PREDICATES: list[dict] = [
    {
        "key": "is a positive review",
        "slug": "is_a_positive_review",
        # Movie sentiment (SemBench), added 2026-07-29 for the CIDR eval.
        # Hand-written + DOMAIN-NEUTRAL: only about the REVIEW's tone. The auto-
        # generated decomposition leaked Amazon "product features / benefits /
        # successful use cases" framing, which misfires on movie reviews -> Q2 F1
        # was 0.60. Registering it here makes it a KNOWN predicate (canned path).
        "subquestions": [
            ("pos_C1", "content",
             "Does the review express overall approval, enjoyment, or "
             "satisfaction with the movie?"),
            ("pos_C2", "content",
             "Does the reviewer speak favorably about the movie or recommend it?"),
            ("pos_C3", "content",
             "Is the review largely free of major complaints, disappointment, "
             "or criticism?"),
            ("pos_M1", "meta",
             "Does the review contain enough text to judge its overall "
             "sentiment?"),
            ("pos_M2", "meta",
             "Is the sentiment clear and one-directional (not mixed, sarcastic, "
             "or self-contradictory)?"),
        ],
    },
    {
        "key": "is vegan",
        "slug": "vegan",
        "subquestions": [
            ("veg_C1", "content",
             "Does the listing explicitly state the product is vegan, "
             "plant-based, or 100% plant-derived?"),
            ("veg_C2", "content",
             "Are the listed ingredients and materials free of any "
             "animal-derived component (e.g. milk, egg, honey, gelatin, whey, "
             "casein, carmine, beeswax)?"),
            ("veg_C3", "content",
             "Is this product of a type that is inherently free of "
             "animal-derived ingredients, so it would be vegan by its very "
             "nature?"),
            ("veg_M1", "meta",
             "Does the provided description/features contain enough ingredient "
             "or material information to determine whether the product is "
             "vegan?"),
            ("veg_M2", "meta",
             "Is the available evidence about the product's vegan status "
             "clear and internally consistent (as opposed to ambiguous or "
             "self-contradictory, e.g. 'may contain traces of milk')?"),
        ],
    },
    {
        "key": "looks indulgent enough to give as a Valentine's gift",
        "slug": "indulgent_valentine",
        "subquestions": [
            ("ind_C1", "content",
             "Does the product's presentation or packaging look premium and "
             "gift-worthy (e.g. gift box, ribbon, elegant or luxurious "
             "packaging)?"),
            ("ind_C2", "content",
             "Is this product a type commonly given as a romantic or "
             "Valentine's gift (e.g. chocolate, confectionery, jewelry, "
             "flowers, gourmet treats)?"),
            ("ind_C3", "content",
             "Does the description use language conveying indulgence, luxury, "
             "or romance (e.g. decadent, rich, premium, luxurious, romantic, "
             "gourmet)?"),
            ("ind_M1", "meta",
             "Does the provided description/features contain enough "
             "presentational or aesthetic information to judge how "
             "gift-worthy the product looks?"),
            ("ind_M2", "meta",
             "Is the available evidence about whether the product looks "
             "indulgent clear and unambiguous (as opposed to a plain, purely "
             "functional listing with no aesthetic cues)?"),
        ],
    },
    {
        "key": "is suitable for a Christmas dinner",
        "slug": "christmas_dinner",
        # v4, 2026-08-02. Replaces v3 after an A/B on the full 168-product Q2 pool
        # (cidr_evaluation/grocery_runner/predicate_ab.py). History:
        #   v1  "does the listing MENTION Christmas" -> 97.5% F, recall 11.8%
        #   v2  occasion-agnostic "festive character" -> P=0.37 / R=0.36
        #   v3  deliberately deleted the occasion signal, on the theory that the
        #       panel rewards any plain ingredient. It does not: only 28% of the
        #       pool is GT-true. All three v3 routes ended up near-tautological on
        #       a grocery catalogue (T-rates 88/95/81%), so 91% of candidates
        #       passed and precision was capped at 0.34 -- not fixable by tau
        #       (even theta=1.0 left 118 of 154 passing, F1 ceiling 0.482).
        #
        # The diagnostic that drove v4 is LIFT = T-rate(GT-true) - T-rate(GT-false)
        # per route; a route with lift ~0 carries no signal however sensible it
        # reads. Measured on gemini-2.5-flash-lite, no CoT, no thinking:
        #   v3  C1 +14  C2 +20  C3 +11  -> F1 0.487
        #   v4  C1 +42  C2 +25  C3 +16  -> F1 0.635   (flash: 0.497 -> 0.692)
        #
        # C1 restores the seasonal signal v3 removed, but asks "would people buy
        # this FOR a holiday meal" rather than "is it Christmas-themed" -- the
        # latter was v1 and reads F for the flour/salt/broth/nuts the panel
        # actually rewards. C2 splits a shared cooked meal from an individual
        # snack, which is the axis the panel divides on (top-rated: vanilla beans,
        # coarse salt, in-shell pecans, chicken broth, lasagne sheets;
        # bottom-rated: applesauce pouches, 100-calorie packs, Cheetos, MREs).
        # C3 is kept verbatim from v3: it is the one route that already pointed the
        # right way.
        #
        # CAVEAT for the paper: v4 was written by inspecting v3's error pattern
        # against the same 21-voter labels it is scored on. It is tuned on the
        # eval set and must be reported as such.
        "subquestions": [
            ("chr_C1", "content",
             "Is this the kind of item people specifically buy for a winter-holiday "
             "meal or holiday baking - for example a festive roast or main, a "
             "holiday side, stuffing, gravy or cranberry, festive baking "
             "ingredients or warm spices, nuts or dried fruit for the holiday "
             "table, or a seasonal celebratory drink?"),
            ("chr_C2", "content",
             "Would this be served as part of a prepared, shared, sit-down meal - "
             "cooked, plated, or set on the table for guests - as opposed to an "
             "individually packaged everyday snack, lunchbox item, or grab-and-go "
             "convenience food?"),
            ("chr_C3", "content",
             "Is the product free of cues tying it to a DIFFERENT occasion - it "
             "is not branded or themed for a birthday, wedding, graduation, "
             "baby shower, or Halloween?"),
            ("chr_M1", "meta",
             "Does the provided description/features say enough about what the "
             "product actually is (its type or ingredients) to judge whether it "
             "belongs at a holiday dinner?"),
            ("chr_M2", "meta",
             "Is the available evidence clear and unambiguous (as opposed to a "
             "generic listing that says nothing about what the product is or how "
             "it is used)?"),
        ],
    },
    {
        "key": "is suitable for kids",
        "slug": "kids_suitable",
        # Added 2026-07-30 for the grocery baseline eval. Previously UNREGISTERED,
        # so it fell through to LLM-generated sub-questions -> the cascade passed
        # only 16 of 115 candidates, recall 0.17, Q3 F1 0.076. Exactly the failure
        # mode christmas_dinner v1 had: narrow routes that all skew FALSE.
        #
        # This predicate is PERMISSIVE in the ground truth: 58% of the 200 products
        # have T_rate > 0.5 (mean 0.56). Most grocery food IS fine for a child; the
        # signal is the presence of a DISQUALIFIER, not the presence of kid
        # marketing. Since content_score is the MEAN of the three content answers
        # (cascade_live._score), every route must read TRUE for an ordinary
        # kid-suitable food -- a route like "does the listing market to children"
        # would answer FALSE for plain pasta and drag the mean below tau.
        "subquestions": [
            ("kid_C1", "content",
             "Is this an ordinary food or drink that a child could eat or drink "
             "as part of a normal diet - for example a snack, cereal, fruit, "
             "pasta, bread, dairy, juice, or sweet treat?"),
            ("kid_C2", "content",
             "Is the product free of adults-only content - no alcohol, no "
             "coffee/energy-drink levels of caffeine, no tobacco or nicotine, no "
             "adult dietary supplements, weight-loss or medicinal products?"),
            ("kid_C3", "content",
             "Would a typical child accept and safely handle this product - a "
             "mild, familiar taste rather than an acquired one (not intensely "
             "spicy, bitter, or alcoholic-flavoured), and not an obvious choking "
             "hazard for young children?"),
            ("kid_M1", "meta",
             "Does the provided description/features say enough about what the "
             "product actually is (its type, ingredients, or flavour) to judge "
             "whether a child could have it?"),
            ("kid_M2", "meta",
             "Is the available evidence about the product's suitability for "
             "children clear and unambiguous (as opposed to a generic listing "
             "that says nothing about what the product contains or tastes like)?"),
        ],
    },
]

def _compact_schema(n: int) -> dict:
    """One string, exactly n characters from [TF?]. The pattern is enforced by
    the schema, so a short or malformed answer fails at generation rather than
    silently parsing into a partial verdict."""
    return {"type": "string", "pattern": f"^[TF?]{{{n}}}$", "minLength": n, "maxLength": n}


def _array_schema(n: int) -> dict:
    """One enum element per sub-question, positional, no key echo:
        ["T", "F", "F", "T", "F", "T"]
    Middle ground between the verbose objects and the single string -- the model
    still emits a separate, schema-constrained token per question in order."""
    return {"type": "array", "items": {"type": "string", "enum": ANSWER_ENUM},
            "minItems": n, "maxItems": n}


def _parse_array(value, expected_ids: list[str]) -> tuple[dict[str, str], list[str]]:
    """[...] -> {sub_id: answer}, BY POSITION. Same alignment risk as compact, so
    the same hard validation: exact length, known values, no partial results."""
    if not isinstance(value, list):
        return {}, [f"array_not_a_list: {value!r}"]
    if len(value) != len(expected_ids):
        return {}, [f"array_length: got {len(value)}, expected {len(expected_ids)}"]
    bad = [v for v in value if v not in ANSWER_ENUM]
    if bad:
        return {}, [f"array_bad_values: {bad}"]
    return dict(zip(expected_ids, value)), []


def _answers_schema(cot=_UNSET) -> dict:
    """Per-sub-question answer list. With COT, a `reasoning` string is generated
    BEFORE `answer` (propertyOrdering) so the answer is conditioned on the reason."""
    _, cot = _resolve(_UNSET, cot)
    if not cot:
        return {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sub_id": {"type": "string"},
                    "answer": {"type": "string", "enum": ANSWER_ENUM},
                },
                "required": ["sub_id", "answer"],
                "propertyOrdering": ["sub_id", "answer"],
            },
        }
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "sub_id": {"type": "string"},
                "reasoning": {"type": "string"},
                "answer": {"type": "string", "enum": ANSWER_ENUM},
            },
            "required": ["sub_id", "reasoning", "answer"],
            "propertyOrdering": ["sub_id", "reasoning", "answer"],
        },
    }

# Batched schema (--batch N>1): N products in ONE prompt, same model + same
# predicate, only the product data differs. `product_index` is echoed back so we
# can verify the mapping instead of trusting position alone.
def batch_response_schema(cot=_UNSET, n_sub: int | None = None) -> dict:
    answers = (_compact_schema(n_sub) if (_fmt() == "compact" and n_sub)
               else _array_schema(n_sub) if (_fmt() == "array" and n_sub)
               else _answers_schema(cot))
    return {
        "type": "object",
        "properties": {
            "products": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "product_index": {"type": "integer"},
                        "answers": answers,
                    },
                    "required": ["product_index", "answers"],
                    "propertyOrdering": ["product_index", "answers"],
                },
            }
        },
        "required": ["products"],
    }


def response_schema(cot=_UNSET, n_sub: int | None = None) -> dict:
    if _fmt() != "verbose" and n_sub:
        inner = (_compact_schema(n_sub) if _fmt() == "compact"
                 else _array_schema(n_sub))
        return {"type": "object", "properties": {"answers": inner},
                "required": ["answers"]}
    return {
        "type": "object",
        "properties": {"answers": _answers_schema(cot)},
        "required": ["answers"],
    }

INSTRUCTION = """\
You are carefully analyzing ONE product. You will be asked several independent
yes/no-style sub-questions about it. Answer EACH sub-question on its own, using
ONLY the product information provided below. Do NOT let one answer influence
another.

For each sub-question, answer with exactly one of:
  - "T"                -> yes / true (your best judgment leans this way)
  - "F"                -> no / false (your best judgment leans this way)
  - "cannot_determine" -> LAST RESORT ONLY

Be decisive. Default to committing to "T" or "F" based on your best judgment,
even when the evidence is only partial or indirect. Reserve "cannot_determine"
strictly for the rare case where the product information contains NO relevant
signal whatsoever for that sub-question. If there is any hint, cue, or partial
signal, you MUST pick "T" or "F".

Output ONLY the schema-conformant JSON: a list of {sub_id, answer} objects,
one per sub-question. No explanation, no extra fields, no extra text.
"""

# CoT variant of the closing paragraph. Swapped in by _closing() when COT is set,
# so the prose matches the schema the model is actually constrained to.
INSTRUCTION_TAIL_COT = """\
Output ONLY the schema-conformant JSON: a list of {sub_id, reasoning, answer}
objects, one per sub-question. For EACH sub-question, first write "reasoning": one
or two sentences citing the specific evidence in the product information, and THEN
write "answer". Decide the answer from that reasoning; do not write the reasoning
to justify an answer you already chose. No extra fields, no extra text.
"""

# Batched instruction (--batch N>1). Same sub-questions, same model — ONLY the
# product data differs. The ordering contract is stated explicitly and enforced
# by echoing product_index, because a mis-mapped answer is silently wrong data.
INSTRUCTION_BATCH = """\
You are carefully analyzing {n} DIFFERENT products, numbered 1 to {n}. For EACH
product you will answer the SAME set of independent yes/no-style sub-questions.

Answer EACH sub-question on its own, for EACH product, using ONLY that product's
own information. Do NOT let one answer influence another. Do NOT let one product
influence another — judge every product independently, as if it were the only one.

For each sub-question, answer with exactly one of:
  - "T"                -> yes / true (your best judgment leans this way)
  - "F"                -> no / false (your best judgment leans this way)
  - "cannot_determine" -> LAST RESORT ONLY

Be decisive. Default to committing to "T" or "F" based on your best judgment,
even when the evidence is only partial or indirect. Reserve "cannot_determine"
strictly for the rare case where the product information contains NO relevant
signal whatsoever for that sub-question. If there is any hint, cue, or partial
signal, you MUST pick "T" or "F".

ORDERING CONTRACT — follow exactly:
  - Return EXACTLY {n} entries in the "products" array.
  - Return them in the SAME ORDER as the products are given below.
  - Set "product_index" to that product's number as shown (1 to {n}).
  - Every product must have an answer for EVERY sub-question.
  - Do NOT merge, reorder, skip, deduplicate, or invent products.

Output ONLY the schema-conformant JSON. No explanation, no extra fields, no extra text.
"""

# Same swap for the batched prompt.
INSTRUCTION_BATCH_TAIL_COT = """\
Output ONLY the schema-conformant JSON. For EACH sub-question of EACH product,
first write "reasoning": one or two sentences citing the specific evidence in THAT
product's information, and THEN write "answer". Decide the answer from that
reasoning; do not write the reasoning to justify an answer you already chose.
No extra fields, no extra text.
"""

_TAIL_PLAIN = ("Output ONLY the schema-conformant JSON: a list of {sub_id, answer} objects,\n"
               "one per sub-question. No explanation, no extra fields, no extra text.\n")

# Compact replacement for the closing paragraph. The schema alone would not tell
# the model that position carries the mapping, and that is the one thing it must
# get right, so the prose says it explicitly and twice.
_TAIL_COMPACT = ("Output ONLY the schema-conformant JSON. \"answers\" is a SINGLE STRING with\n"
                 "EXACTLY ONE CHARACTER PER SUB-QUESTION, IN THE ORDER THEY ARE LISTED ABOVE:\n"
                 "  \"T\" = yes/true,  \"F\" = no/false,  \"?\" = cannot_determine.\n"
                 "The 1st character answers the 1st sub-question, the 2nd the 2nd, and so on.\n"
                 "Do not skip, reorder, or pad. No explanation, no extra fields, no extra text.\n")
_TAIL_ARRAY = ("Output ONLY the schema-conformant JSON. \"answers\" is a LIST with EXACTLY ONE\n"
               "ENTRY PER SUB-QUESTION, IN THE ORDER THEY ARE LISTED ABOVE: the 1st entry\n"
               "answers the 1st sub-question, the 2nd the 2nd, and so on. Each entry is\n"
               "\"T\", \"F\", or \"cannot_determine\". Do not skip, reorder, or pad.\n"
               "No explanation, no extra fields, no extra text.\n")
_TAIL_BATCH_ARRAY = ("Output ONLY the schema-conformant JSON. For EACH product, \"answers\" is a\n"
                     "LIST with EXACTLY ONE ENTRY PER SUB-QUESTION, IN THE ORDER THEY ARE\n"
                     "LISTED ABOVE. Each entry is \"T\", \"F\", or \"cannot_determine\".\n"
                     "Do not skip, reorder, or pad. No explanation, no extra fields.\n")
_TAIL_BATCH_COMPACT = ("Output ONLY the schema-conformant JSON. For EACH product, \"answers\" is a\n"
                       "SINGLE STRING with EXACTLY ONE CHARACTER PER SUB-QUESTION, IN THE ORDER\n"
                       "THEY ARE LISTED ABOVE: \"T\" = yes/true, \"F\" = no/false,\n"
                       "\"?\" = cannot_determine. Do not skip, reorder, or pad.\n"
                       "No explanation, no extra fields, no extra text.\n")
_TAIL_BATCH_PLAIN = ("Output ONLY the schema-conformant JSON. No explanation, no extra fields, "
                     "no extra text.\n")


def _instruction(cot=_UNSET) -> str:
    _, cot = _resolve(_UNSET, cot)
    if _fmt() != "verbose":
        if cot:
            raise ValueError(f"ANSWER_FORMAT={_fmt()!r} and COT are incompatible: CoT "
                             "needs a reasoning field per answer")
        tail = _TAIL_COMPACT if _fmt() == "compact" else _TAIL_ARRAY
        return _localize(INSTRUCTION.replace(_TAIL_PLAIN, tail))
    return _localize(INSTRUCTION if not cot
                     else INSTRUCTION.replace(_TAIL_PLAIN, INSTRUCTION_TAIL_COT))


def _instruction_batch(n: int, cot=_UNSET) -> str:
    _, cot = _resolve(_UNSET, cot)
    if _fmt() != "verbose":
        if cot:
            raise ValueError(f"ANSWER_FORMAT={_fmt()!r} and COT are incompatible")
        tail = _TAIL_BATCH_COMPACT if _fmt() == "compact" else _TAIL_BATCH_ARRAY
        base = INSTRUCTION_BATCH.replace(_TAIL_BATCH_PLAIN, tail)
    else:
        base = (INSTRUCTION_BATCH if not cot
                else INSTRUCTION_BATCH.replace(_TAIL_BATCH_PLAIN, INSTRUCTION_BATCH_TAIL_COT))
    return _localize(base).format(n=n)


# ── data loading ─────────────────────────────────────────────────────────────

def load_metas(meta_jsonl: Path = META_JSONL) -> dict[str, dict]:
    metas: dict[str, dict] = {}
    with Path(meta_jsonl).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            metas[d["parent_asin"]] = d
    return metas


def load_gt_soft_labels() -> dict[tuple[str, str], str]:
    """Return {(parent_asin, predicate_slug) -> T_rate string}."""
    out: dict[tuple[str, str], str] = {}
    with GT_SOFT_LABELS.open() as f:
        for r in csv.DictReader(f):
            out[(r["parent_asin"], r["predicate_slug"])] = r["T_rate"]
    return out


def format_text_block(meta: dict) -> str:
    """Match the GT voters' text block: title + features + description only."""
    parts: list[str] = [f"Title: {meta.get('title') or '(none)'}"]
    feats = meta.get("features") or []
    if feats:
        parts.append("Features:\n" + "\n".join(f"  - {x}" for x in feats))
    desc = meta.get("description") or []
    if desc:
        parts.append("Description:\n" + " ".join(desc))
    return "\n\n".join(parts)


def build_prompt(text_block: str, subquestions: list[tuple], cot=_UNSET) -> str:
    q_lines = "\n".join(f"  {sid}: {text}" for sid, _stype, text in subquestions)
    return (
        f"{_instruction(cot)}\n"
        f"{RECORD_NOUN.upper()} INFORMATION (text):\n{text_block}\n\n"
        f"SUB-QUESTIONS (answer every one):\n{q_lines}"
    )


def build_batch_prompt(text_blocks: list[str], subquestions: list[tuple], cot=_UNSET,
                       force_commit: bool = False) -> str:
    """One prompt, N products, ONE predicate. Sub-questions stated once (that is
    the whole point — amortise the ~400-token fixed block); products numbered.

    force_commit=True (terminal oracle step) appends the same must-decide note
    _build_contents uses for the single-item path — see _FORCE_COMMIT_NOTE.
    2026-08-03: previously batch was unconditionally disabled at any
    force_commit step (cascade_live.py forced B=1), which meant tier 2 in the
    text-only route (always terminal, so always force_commit) never batched at
    all -- 98% of a run's cost/latency was going through the unbatched path.
    """
    n = len(text_blocks)
    q_lines = "\n".join(f"  {sid}: {text}" for sid, _stype, text in subquestions)
    prods = "\n\n".join(
        f"--- {RECORD_NOUN.upper()} {i} ---\n{tb}" for i, tb in enumerate(text_blocks, start=1)
    )
    prompt = (
        f"{_instruction_batch(n, cot)}\n"
        f"SUB-QUESTIONS (answer every one, for EVERY {RECORD_NOUN}):\n{q_lines}\n\n"
        f"THE {n} {RECORD_NOUN.upper()}S:\n\n{prods}"
    )
    if force_commit:
        prompt += force_commit_note(subquestions)
    return prompt


# ── output ───────────────────────────────────────────────────────────────────

ANSWERS_COLUMNS = [
    "parent_asin", "predicate", "predicate_slug",
    "sub_id", "sub_type", "answer", "gt_T_rate",
]
CALLS_COLUMNS = [
    "parent_asin", "predicate", "predicate_slug", "model",
    "batch_id", "batch_size",
    "latency_s", "prompt_tokens", "output_tokens", "thinking_tokens", "cost_usd",
    "n_answers", "parse_errors", "raw_response",
]


def _append_rows(path: Path, columns: list[str], rows: list[dict]) -> None:
    if not rows:
        return
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns)
    for r in rows:
        w.writerow(r)
    with path.open("a", newline="") as f:
        f.write(buf.getvalue())
        f.flush()
        os.fsync(f.fileno())


def _ensure_header(path: Path, columns: list[str]) -> None:
    with path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=columns).writeheader()


# ── modality-aware prompt intros (content of sub-questions is unchanged) ──────

# Derived from _instruction(cot) rather than the INSTRUCTION constant, so the CoT
# tail reaches the image modalities too -- otherwise a --cot run would silently be
# CoT for text and non-CoT for image, while the response schema demanded `reasoning`
# in both.
def _instruction_image(cot=_UNSET) -> str:
    return _instruction(cot).replace(
        "using\nONLY the product information provided below.",
        "using\nONLY the product IMAGE provided.",
    ).replace(
        "You are carefully analyzing ONE product.",
        "You are carefully analyzing ONE product from its IMAGE only.",
    )


def _instruction_textimage(cot=_UNSET) -> str:
    return _instruction(cot).replace(
        "using\nONLY the product information provided below.",
        "using BOTH the product text AND the product IMAGE provided below.",
    )

# Appended ONLY at the terminal escalation step (the oracle, (3, text_image)): there
# is no further review, so a CONTENT sub-question must commit to T/F using the model's
# prior rather than defer. META sub-questions stay honest — they may still report the
# evidence is insufficient, which is exactly what keeps the confidence (correctly) low
# on a forced guess. See notes/cascade_escalation_route.md §2.1.
_FORCE_COMMIT_NOTE = (
    "\n\nFINAL VERDICT REQUIRED — this is the last-resort evaluation, there is NO "
    "further review. For every CONTENT sub-question you MUST answer \"T\" or \"F\" "
    "using your best judgment and prior knowledge; \"cannot_determine\" is NOT "
    "permitted for content sub-questions here. META sub-questions MAY still answer "
    "\"cannot_determine\" when the evidence is genuinely insufficient or unclear."
)

# RELATED-QUESTION scheme (paper §1.3): every question is a same-direction related
# question and there are no meta questions, so the carve-out above has nothing to
# apply to — all of them must commit. Keeping the content/meta wording here would
# tell the model about a distinction its question list does not have.
_FORCE_COMMIT_NOTE_RELATED = (
    "\n\nFINAL VERDICT REQUIRED — this is the last-resort evaluation, there is NO "
    "further review. For EVERY sub-question you MUST answer \"T\" or \"F\" using "
    "your best judgment and prior knowledge; \"cannot_determine\" is NOT permitted "
    "here."
)


def force_commit_note(subquestions: list[tuple]) -> str:
    """Pick the must-decide note that matches the predicate's question shape."""
    if subquestions and all(stype == "related" for _sid, stype, _t in subquestions):
        return _FORCE_COMMIT_NOTE_RELATED
    return _FORCE_COMMIT_NOTE


def _download_image(url: str, timeout_s: int = 15):
    """Return (bytes, mime_type). Raises on failure."""
    import requests
    resp = requests.get(url, timeout=timeout_s)
    resp.raise_for_status()
    ul = url.lower().split("?")[0]
    mime = ("image/png" if ul.endswith(".png") else
            "image/webp" if ul.endswith(".webp") else
            "image/gif" if ul.endswith(".gif") else "image/jpeg")
    return resp.content, mime


def _build_contents(meta: dict, subqs: list, modality: str,
                    force_commit: bool = False, cot=_UNSET):
    """Build the list of google.genai Parts for the given modality.

    force_commit=True (only the terminal oracle step) appends the must-decide note
    so content sub-questions commit to T/F. See notes/cascade_escalation_route.md §2.1.
    """
    from google.genai import types

    q_lines = "\n".join(f"  {sid}: {text}" for sid, _st, text in subqs)
    if modality == "image":
        prompt = (f"{_instruction_image(cot)}\n"
                  f"SUB-QUESTIONS (answer every one from the IMAGE):\n{q_lines}")
    elif modality == "text_image":
        prompt = (f"{_instruction_textimage(cot)}\n"
                  f"{RECORD_NOUN.upper()} INFORMATION (text):\n{format_text_block(meta)}\n\n"
                  f"SUB-QUESTIONS (use BOTH the text and the image):\n{q_lines}")
    else:  # text
        prompt = build_prompt(format_text_block(meta), subqs, cot)

    if force_commit:
        prompt += force_commit_note(subqs)

    parts = [types.Part.from_text(text=prompt)]
    if modality in ("image", "text_image"):
        url = _meta_main_image_url(meta)
        if not url:
            raise ValueError("no image_url available for product")
        img_bytes, mime = _download_image(url)
        parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))
    return parts


# ── one call ─────────────────────────────────────────────────────────────────

def _parse_compact(value, expected_ids: list[str]) -> tuple[dict[str, str], list[str]]:
    """"TFT?FT" -> {sub_id: answer}, BY POSITION.

    Position is the only thing tying an answer to its question here, so every
    way that can go wrong is checked rather than absorbed: wrong type, wrong
    length, unknown character. A partially-valid string is rejected outright --
    a half-filled verdict is worse than a missing one, because the caller would
    score it as if it were complete."""
    if not isinstance(value, str):
        return {}, [f"compact_not_a_string: {value!r}"]
    v = value.strip().upper()
    if len(v) != len(expected_ids):
        return {}, [f"compact_length: got {len(v)}, expected {len(expected_ids)}"]
    bad = [c for c in v if c not in _COMPACT_MAP]
    if bad:
        return {}, [f"compact_bad_chars: {bad}"]
    return {sid: _COMPACT_MAP[c] for sid, c in zip(expected_ids, v)}, []


def _parse(raw: str, expected_ids: list[str]) -> tuple[dict[str, str], list[str]]:
    errors: list[str] = []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {}, [f"json_parse_failure: {exc}"]
    if _fmt() == "compact":
        return _parse_compact(parsed.get("answers"), expected_ids)
    if _fmt() == "array":
        return _parse_array(parsed.get("answers"), expected_ids)
    answers: dict[str, str] = {}
    for item in parsed.get("answers", []):
        if not isinstance(item, dict):
            errors.append(f"item_not_object: {item!r}")
            continue
        sid = item.get("sub_id")
        ans = item.get("answer")
        if sid not in expected_ids:
            errors.append(f"unexpected_sub_id: {sid!r}")
            continue
        if sid in answers:
            errors.append(f"duplicate_sub_id: {sid!r}")
            continue
        if ans not in ANSWER_ENUM:
            errors.append(f"bad_answer[{sid}]: {ans!r}")
            continue
        answers[sid] = ans
    missing = [sid for sid in expected_ids if sid not in answers]
    if missing:
        errors.append(f"missing_sub_ids: {missing}")
    return answers, errors


def _parse_batch(raw: str, expected_ids: list[str], n: int
                 ) -> tuple[dict[int, dict[str, str]], list[str]]:
    """Parse a batched response into {product_index -> {sub_id: answer}}.

    Mapping errors are the whole risk of batching — a silently mis-mapped answer
    is wrong data that looks fine. So validate hard: exactly n entries, indices
    exactly 1..n each once, every sub_id present per product.
    """
    errors: list[str] = []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {}, [f"json_parse_failure: {exc}"]

    out: dict[int, dict[str, str]] = {}
    for item in parsed.get("products", []):
        if not isinstance(item, dict):
            errors.append(f"product_not_object: {item!r}")
            continue
        idx = item.get("product_index")
        if not isinstance(idx, int) or not (1 <= idx <= n):
            errors.append(f"bad_product_index: {idx!r}")
            continue
        if idx in out:
            errors.append(f"duplicate_product_index: {idx}")
            continue
        if _fmt() == "compact":
            answers, errs = _parse_compact(item.get("answers"), expected_ids)
        elif _fmt() == "array":
            answers, errs = _parse_array(item.get("answers"), expected_ids)
        else:
            answers, errs = _parse(json.dumps({"answers": item.get("answers", [])}),
                                   expected_ids)
        errors.extend(f"p{idx}:{e}" for e in errs)
        out[idx] = answers

    missing = [i for i in range(1, n + 1) if i not in out]
    if missing:
        errors.append(f"missing_product_index: {missing}")
    return out, errors


def run_batch(
    client, asins: list[str], metas: dict, predicate: dict, gt: dict,
    *, model: str, temperature: float, timeout_s: int, max_retries: int,
    thinking_budget=_UNSET, cot=_UNSET, force_commit: bool = False,
) -> tuple[list[dict], list[dict]]:
    """N products, ONE predicate, ONE call (text modality only).

    Cost/latency/tokens are ATTRIBUTED per product (divided by batch size) so that
    summing the per-product rows still gives the true call total. `batch_size` and
    `batch_id` are recorded so the real calls can be reconstructed.
    """
    from google.genai import types

    thinking_budget, cot = _resolve(thinking_budget, cot)
    subqs = predicate["subquestions"]
    expected_ids = [sid for sid, _t, _q in subqs]
    n = len(asins)
    batch_id = f"{predicate['slug']}:{asins[0]}+{n - 1}"

    t0 = time.time()
    raw = ""
    usage = {"prompt_token_count": None, "candidates_token_count": None}
    errors: list[str] = []
    per_idx: dict[int, dict[str, str]] = {}
    try:
        prompt = build_batch_prompt([format_text_block(metas[a]) for a in asins], subqs, cot,
                                    force_commit=force_commit)
        response = _generate_with_timeout(
            client, model=model, contents=[types.Part.from_text(text=prompt)],
            config=types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json",
                response_schema=batch_response_schema(cot, n_sub=len(subqs)),
                thinking_config=_thinking_cfg(thinking_budget),
            ),
            timeout_s=timeout_s, max_retries=max_retries,
            label=f"subq-batch[{n}] {predicate['slug']}",
        )
        raw = _extract_text_from_gemini_response(response) or ""
        usage = _extract_usage_from_gemini_response(response)
        per_idx, errors = _parse_batch(raw, expected_ids, n)
    except Exception as exc:  # noqa: BLE001 — record and move on
        errors = [f"call_error: {type(exc).__name__}: {exc}"]

    latency = time.time() - t0
    p_tok = usage.get("prompt_token_count") or 0
    o_tok = usage.get("candidates_token_count") or 0
    # Gemini bills thinking at the OUTPUT rate. tier 2 (gemini-2.5-flash) keeps
    # thinking on by design (cascade_live.py), ~650 tok/call, so omitting this
    # understated every tier-2 call ~4.8x. Fixed 2026-07-30.
    t_tok = usage.get("thoughts_token_count") or 0
    price = PRICING.get(model, {"input": 0.0, "output": 0.0})
    cost = (p_tok / 1e6) * price["input"] + ((o_tok + t_tok) / 1e6) * price["output"]

    answer_rows: list[dict] = []
    call_rows: list[dict] = []
    for i, asin in enumerate(asins, start=1):
        answers = per_idx.get(i, {})
        gt_rate = gt.get((asin, predicate["slug"]), "")
        answer_rows.extend({
            "parent_asin": asin, "predicate": predicate["key"],
            "predicate_slug": predicate["slug"], "sub_id": sid, "sub_type": stype,
            "answer": answers.get(sid, "MISSING"), "gt_T_rate": gt_rate,
        } for sid, stype, _q in subqs)
        call_rows.append({
            "parent_asin": asin, "predicate": predicate["key"],
            "predicate_slug": predicate["slug"], "model": model,
            "batch_id": batch_id, "batch_size": n,
            "latency_s": f"{latency / n:.3f}",          # attributed share
            "prompt_tokens": p_tok // n, "output_tokens": o_tok // n,
            "thinking_tokens": t_tok // n,
            "cost_usd": f"{cost / n:.6f}",              # attributed share
            "n_answers": len(answers),
            "parse_errors": "; ".join(errors),
            "raw_response": raw.replace("\n", "\\n").replace("\r", "") if i == 1 else "",
        })
    return answer_rows, call_rows


def run_one(
    client, asin: str, meta: dict, predicate: dict, gt: dict,
    *, model: str, temperature: float, timeout_s: int, max_retries: int,
    modality: str = "text", force_commit: bool = False,
    thinking_budget=_UNSET, cot=_UNSET,
) -> tuple[list[dict], dict]:
    """Returns (answer_rows, call_row). modality ∈ {text, image, text_image}.
    force_commit=True (terminal oracle step) makes content sub-questions commit to T/F."""
    from google.genai import types

    thinking_budget, cot = _resolve(thinking_budget, cot)
    subqs = predicate["subquestions"]
    expected_ids = [sid for sid, _t, _q in subqs]

    t0 = time.time()
    raw = ""
    usage = {"prompt_token_count": None, "candidates_token_count": None}
    errors: list[str] = []
    answers: dict[str, str] = {}
    try:
        contents = _build_contents(meta, subqs, modality, force_commit=force_commit, cot=cot)
        response = _generate_with_timeout(
            client,
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json",
                response_schema=response_schema(cot, n_sub=len(subqs)),
                thinking_config=_thinking_cfg(thinking_budget),
            ),
            timeout_s=timeout_s,
            max_retries=max_retries,
            label=f"subq {asin} [{predicate['slug']}]",
        )
        raw = _extract_text_from_gemini_response(response) or ""
        usage = _extract_usage_from_gemini_response(response)
        answers, errors = _parse(raw, expected_ids)
    except Exception as exc:  # noqa: BLE001 — record and move on
        errors = [f"call_error: {type(exc).__name__}: {exc}"]

    latency = time.time() - t0
    p_tok = usage.get("prompt_token_count") or 0
    o_tok = usage.get("candidates_token_count") or 0
    # Gemini bills thinking at the OUTPUT rate. tier 2 (gemini-2.5-flash) keeps
    # thinking on by design (cascade_live.py), ~650 tok/call, so omitting this
    # understated every tier-2 call ~4.8x. Fixed 2026-07-30.
    t_tok = usage.get("thoughts_token_count") or 0
    price = PRICING.get(model, {"input": 0.0, "output": 0.0})
    cost = (p_tok / 1e6) * price["input"] + ((o_tok + t_tok) / 1e6) * price["output"]

    gt_rate = gt.get((asin, predicate["slug"]), "")
    answer_rows = [
        {
            "parent_asin": asin,
            "predicate": predicate["key"],
            "predicate_slug": predicate["slug"],
            "sub_id": sid,
            "sub_type": stype,
            "answer": answers.get(sid, "MISSING"),
            "gt_T_rate": gt_rate,
        }
        for sid, stype, _q in subqs
    ]
    call_row = {
        "parent_asin": asin,
        "predicate": predicate["key"],
        "predicate_slug": predicate["slug"],
        "model": model,
        "latency_s": f"{latency:.3f}",
        "prompt_tokens": p_tok,
        "output_tokens": o_tok,
        "thinking_tokens": t_tok,
        "cost_usd": f"{cost:.6f}",
        "n_answers": len(answers),
        "parse_errors": "; ".join(errors),
        "raw_response": raw.replace("\n", "\\n").replace("\r", ""),
    }
    return answer_rows, call_row


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gemini-2.5-flash-lite")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N products (smoke test).")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--modality", choices=("text", "image", "text_image"),
                    default="text", help="Input given to the model.")
    ap.add_argument("--predicates", default=None,
                    help="Comma-separated predicate slugs to run (default: all).")
    ap.add_argument("--batch", type=int, default=1,
                    help="Products per prompt. 1 (default) = today's behaviour: "
                         "5 sub-questions + 1 product. N>1 = 5 sub-questions + N "
                         "products in ONE call (same model + same predicate; only "
                         "the product data differs). Amortises the ~400-token "
                         "fixed block. text modality only.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout-s", type=int, default=30)
    ap.add_argument("--max-retries", type=int, default=1)
    ap.add_argument("--meta-jsonl", type=Path, default=META_JSONL,
                    help="Product meta JSONL to run on (default: the 200-sample; "
                         "pass the _sample_1000 file to run on all 1000).")
    args = ap.parse_args()

    load_env()
    from google import genai
    client = genai.Client(api_key=_require_key("GEMINI_API_KEY"))

    metas = load_metas(args.meta_jsonl)
    gt = load_gt_soft_labels()
    asins = sorted(metas.keys())
    if args.limit is not None:
        asins = asins[: args.limit]

    preds = PREDICATES
    if args.predicates:
        want = [s.strip() for s in args.predicates.split(",") if s.strip()]
        known = {p["slug"] for p in PREDICATES}
        unknown = [s for s in want if s not in known]
        if unknown:
            raise SystemExit(f"unknown predicate slug(s): {unknown}; known: {sorted(known)}")
        preds = [p for p in PREDICATES if p["slug"] in want]

    if args.batch < 1:
        raise SystemExit("--batch must be >= 1")
    if args.batch > 1 and args.modality != "text":
        # Multiple images in one prompt makes the product<->answer mapping much
        # riskier; not worth it since batching only pays off on text-side input.
        raise SystemExit("--batch > 1 is only supported with --modality text")

    # A task is (asins, predicate): batch=1 -> one asin per task (identical to the
    # original behaviour); batch=N -> N asins per task, same predicate.
    tasks = [(asins[i:i + args.batch], pred)
             for pred in preds
             for i in range(0, len(asins), args.batch)]

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    answers_path = out_dir / "subq_answers.csv"
    calls_path = out_dir / "subq_calls.csv"
    _ensure_header(answers_path, ANSWERS_COLUMNS)
    _ensure_header(calls_path, CALLS_COLUMNS)

    print(f"[subq] model      : {args.model}")
    print(f"[subq] products   : {len(asins)}")
    print(f"[subq] predicates : {len(preds)} ({', '.join(p['slug'] for p in preds)})")
    print(f"[subq] batch      : {args.batch} product(s)/prompt  -> {len(tasks)} calls "
          f"for {len(asins)*len(preds)} (product,predicate) units")
    print(f"[subq] workers    : {args.workers}")
    print(f"[subq] out dir    : {out_dir}")
    print()

    lock = threading.Lock()
    latencies: list[float] = []
    total_cost = 0.0
    n_parse_err = 0
    done = 0
    total = len(tasks)

    def work(batch_asins: list[str], pred: dict):
        """Returns (answer_rows, call_rows). Retries transient server errors
        (503 overload) / timeouts with backoff; run_* swallow them into
        parse_errors rather than raising."""
        for attempt in range(5):
            if len(batch_asins) == 1:
                arows, crow = run_one(
                    client, batch_asins[0], metas[batch_asins[0]], pred, gt,
                    model=args.model, temperature=args.temperature,
                    timeout_s=args.timeout_s, max_retries=args.max_retries,
                    modality=args.modality,
                )
                crow.setdefault("batch_id", "")
                crow.setdefault("batch_size", 1)
                crows = [crow]
            else:
                arows, crows = run_batch(
                    client, batch_asins, metas, pred, gt,
                    model=args.model, temperature=args.temperature,
                    timeout_s=args.timeout_s, max_retries=args.max_retries,
                )
            if not crows[0]["parse_errors"].startswith("call_error"):
                return arows, crows
            time.sleep(1.5 * (attempt + 1))
        return arows, crows

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(work, bs, p): (bs, p) for bs, p in tasks}
        for fut in as_completed(futures):
            bs, p = futures[fut]
            done += 1
            label = bs[0] if len(bs) == 1 else f"{bs[0]}+{len(bs)-1}"
            try:
                answer_rows, call_rows = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  [{done}/{total}] {label} [{p['slug']}] WORKER ERROR: {exc}")
                continue
            c0 = call_rows[0]
            with lock:
                _append_rows(answers_path, ANSWERS_COLUMNS, answer_rows)
                _append_rows(calls_path, CALLS_COLUMNS, call_rows)
                # per-call latency (rows carry the attributed share)
                latencies.append(float(c0["latency_s"]) * len(bs))
                total_cost += sum(float(r["cost_usd"]) for r in call_rows)
                if c0["parse_errors"]:
                    n_parse_err += 1
            flag = "  ERR" if c0["parse_errors"] else ""
            print(f"  [{done}/{total}] {label} [{p['slug']}] "
                  f"lat={float(c0['latency_s'])*len(bs):.3f}s "
                  f"tok={int(c0['prompt_tokens'])*len(bs)}/{int(c0['output_tokens'])*len(bs)} "
                  f"cost=${sum(float(r['cost_usd']) for r in call_rows):.6f} "
                  f"n={c0['n_answers']}/5{flag}")

    print()
    print("=" * 60)
    print(f"[subq] calls           : {total}")
    print(f"[subq] total cost      : ${total_cost:.4f}")
    if latencies:
        latencies.sort()
        p95 = latencies[int(0.95 * (len(latencies) - 1))]
        print(f"[subq] latency mean/p95: {mean(latencies):.2f}s / {p95:.2f}s")
    print(f"[subq] calls w/ errors : {n_parse_err}")
    print(f"[subq] answers -> {answers_path}")
    print(f"[subq] calls   -> {calls_path}")


if __name__ == "__main__":
    main()
