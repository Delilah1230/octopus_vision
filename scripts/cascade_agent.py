#!/usr/bin/env python3
"""
cascade_agent — the module that turns a PREDICATE into its 5 sub-questions.

This is the missing front stage of the cascade pipeline (see notes/todo.md
"子问题设计 —— 最大杠杆,但没有 owner" and subquestion_design_risk.md). Today the
5 sub-questions per predicate are hand-written directly in subquestion_probe.py's
PREDICATES table; the goal is for THIS module to generate them automatically:

    predicate  ─┐
                ├─> build_generation_prompt() ─> LM ─> 5 sub-questions
    generation ─┘        (English meta-prompt)         (3 content + 2 meta)
    instruction

INTENDED interface (the seam this module defines):
    input  : a predicate (slug + natural-language description) + a generation
             instruction (the English meta-prompt below).
    action : fill the predicate into the instruction, send to an LM.
    output : list[SubQuestion] — exactly 3 content + 2 meta, in the SAME shape
             (sub_id, sub_type, text) that subquestion_probe.PREDICATES uses, so
             the rest of the cascade is unchanged whether the sub-questions were
             hand-written or generated.

STEP 1 — FAKE AUTOMATION (this commit). The default mode is "canned": it does NOT
call any model, it just returns the existing hand-written sub-questions for that
predicate. This lets us stand the module up and slot it into place WITHOUT yet
trusting an auto-generated decomposition. The real path (mode="llm") is written
and wired to connect_api, but is off by default and NOT yet plugged into the
cascade — flip the mode only once its output is validated against the held-out
predicates.

    from cascade_agent import CascadeAgent
    agent = CascadeAgent()                      # canned (fake) by default
    subqs = agent.generate("vegan")             # -> the 5 hand-written ones
    # later, once validated:
    agent = CascadeAgent(mode="llm")            # real auto-generation
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

sys.path.insert(0, str(Path(__file__).resolve().parent))
from subquestion_probe import PREDICATES  # noqa: E402

# ── the sub-question shape, identical to subquestion_probe.PREDICATES entries ──
# Each existing entry is a tuple (sub_id, sub_type, text); we keep that tuple
# contract (SubQuestion.as_tuple()) so a generated list is drop-in compatible.


@dataclass(frozen=True)
class SubQuestion:
    sub_id: str          # e.g. "veg_C1"
    sub_type: str        # "content" | "meta"
    text: str            # the English question

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.sub_id, self.sub_type, self.text)


# Leading tokens that carry no identity — dropping them is what makes
# "is_a_positive_review" produce the same "pos" prefix a human picked.
_SLUG_STOPWORDS = frozenset({"is", "a", "an", "the", "are", "was", "were", "be"})


# A predicate the agent decomposes: a slug (stable id) + a natural-language
# description of what the predicate means. This is the module's INPUT.
@dataclass(frozen=True)
class PredicateSpec:
    slug: str            # e.g. "vegan"
    key: str             # e.g. "is vegan" — the NL description handed to the LM
    # What KIND of record this predicate is evaluated over, in one sentence, e.g.
    # "a critic's written review of a movie". Optional, but supply it whenever the
    # schema knows: without it the generator has only the predicate string and
    # invents a domain. "is a positive review" alone reliably produces questions
    # about "the product or service" — the exact Amazon leakage that made the
    # 2026-07-29 auto-generated decomposition score F1 0.60 on movie reviews.
    # This is schema context the system genuinely has (the parser already reads
    # the table's column descriptions), NOT knowledge of the test set.
    context: Optional[str] = None

    @property
    def id_prefix(self) -> str:
        """Slug -> sub_id prefix, matching the hand-written ids: "vegan" -> "veg",
        "kids_suitable" -> "kid", "is_a_positive_review" -> "pos" (the leading
        "is"/"a" carry no identity, and slug[:3] would give a useless "is_" that
        collides across every predicate phrased as "is ...")."""
        tokens = [t for t in self.slug.split("_") if t]
        meaningful = [t for t in tokens if t not in _SLUG_STOPWORDS] or tokens
        return (meaningful[0][:3] or "prd").lower()


# Every predicate the system already knows, keyed by slug, sourced from the one
# place they are currently defined. This is what the CANNED (fake) path returns.
_PREDICATES_BY_SLUG: dict[str, dict] = {p["slug"]: p for p in PREDICATES}


# The counts are a hard contract of the decomposition (see subquestion_probe.py
# POLARITY CONVENTION): 3 content routes toward the predicate + 2 meta questions
# (sufficiency, clarity). The LLM path must return exactly this shape.
N_CONTENT = 3
N_META = 2


# ── the English generation instruction (meta-prompt) ──────────────────────────
# This is the second INPUT to the agent. It is intentionally English (all prompts
# sent to an LM are English; Chinese is only for our own review). It encodes the
# same decomposition design that produced the hand-written sub-questions:
#   * 3 CONTENT sub-questions = three DISTINCT evidence routes toward the
#     predicate holding: (1) an explicit statement in the listing, (2) inference
#     from the product's attributes/ingredients, (3) a category/type prior.
#     Polarity: answer "T" == evidence the predicate HOLDS.
#   * 2 META sub-questions about the EVIDENCE itself, not the product:
#     (M1) sufficiency — is there enough information to decide at all?
#     (M2) clarity     — is the evidence clear/consistent vs ambiguous?
#     Polarity: answer "T" == the favorable state (sufficient / clear).
GENERATION_INSTRUCTION = """\
You design evaluation sub-questions for a product-filtering system.

You are given ONE predicate: a natural-language property that a product may or
may not satisfy. Decompose it into exactly FIVE independent sub-questions that a
downstream model will answer (each with "T", "F", or "cannot_determine") using
only a product's listing text.

Produce exactly {n_content} CONTENT sub-questions and exactly {n_meta} META
sub-questions.

CONTENT sub-questions probe the product itself. Make them three DISTINCT
evidence routes toward the predicate HOLDING, ideally:
  1. an EXPLICIT signal — does the listing directly state the property?
  2. an INFERENCE from attributes — do the ingredients/materials/features imply it?
  3. a CATEGORY PRIOR — is this a type of product that tends to satisfy it?
POLARITY: for every content sub-question, answer "T" must mean evidence that the
predicate HOLDS. Phrase each one positively toward the predicate.

META sub-questions probe the EVIDENCE, not the product:
  M1 SUFFICIENCY — does the listing contain enough information to decide the
     predicate at all?
  M2 CLARITY — is the available evidence clear and internally consistent, as
     opposed to ambiguous, missing, or self-contradictory?
POLARITY: for every meta sub-question, answer "T" must mean the FAVORABLE state
(sufficient / clear).

Keep each sub-question a single self-contained sentence. Do not reference the
other sub-questions. Do not restate the predicate verbatim.

Output ONLY schema-conformant JSON: an object with a "subquestions" array of
{n_total} objects, each {{"sub_type": "content"|"meta", "text": "..."}}, ordered
as the {n_content} content questions first, then the {n_meta} meta questions.
"""

# ── RELATED-QUESTION generation (paper §1.3) ──────────────────────────────────
# The other design: N same-direction RELATED questions, no content/meta split and
# no claim that they decompose the predicate. Three things are load-bearing:
#
#   * SAME DIRECTION. Every question is phrased so that "T" means evidence the
#     predicate HOLDS. That is what makes r (the fraction answering T) a
#     meaningful vote -- a mixed-polarity set would have T mean opposite things
#     on different questions and the count would be noise.
#   * "RELATED", not "SUB". A sub-question implies a containment relation, which
#     immediately raises "why are they weighted equally?". Related questions
#     carry no such claim, so equal weight needs no defence.
#   * EVEN N. With an odd N a majority always exists and UNDETERMINED could never
#     arise -- but the data model depends on U existing (§3.1 Filter, provenance
#     NULL overloading, Table 2). An even N lets U emerge from an exact tie
#     instead of requiring a third answer option.
N_RELATED = 6

RELATED_GENERATION_INSTRUCTION = """\
You design evaluation questions for a data system that filters records with a
natural-language predicate.

You are given ONE predicate: a property that a record may or may not satisfy,
and a description of WHAT KIND OF RECORD it is evaluated over. Write exactly
{n_related} independent questions about a record, which a downstream model will
answer with "T", "F", or "cannot_determine" using only that record's text.

Write the questions for THE RECORD KIND YOU ARE GIVEN. Do not assume a domain
that was not stated, and do not name entities the record kind does not mention.

REQUIREMENTS:
1. SAME DIRECTION. For EVERY question, the answer "T" must mean evidence that
   the predicate HOLDS, and "F" must mean evidence that it does NOT hold. Never
   phrase a question so that "T" argues against the predicate.
2. INDEPENDENT ANGLES. The questions should approach the predicate from
   different angles -- an explicit statement, an inference from attributes, a
   category or type prior, a characteristic consequence, a typical phrasing, an
   absence of disqualifiers -- so that they are not paraphrases of each other
   and do not all fail together on the same kind of record.
3. EQUAL STANDING. Each question is answered on its own and counted equally. Do
   not write a question that only makes sense given another one's answer, and do
   not write one that is merely a restatement of the predicate.
4. SELF-CONTAINED. One sentence each, answerable from the record's text alone,
   with no reference to the other questions.

Output ONLY schema-conformant JSON: an object with a "subquestions" array of
exactly {n_related} objects, each {{"sub_type": "related", "text": "..."}}.
"""

_RELATED_GENERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "subquestions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sub_type": {"type": "string", "enum": ["related"]},
                    "text": {"type": "string"},
                },
                "required": ["sub_type", "text"],
            },
        }
    },
    "required": ["subquestions"],
}


def build_related_generation_prompt(spec: PredicateSpec,
                                    instruction: str = RELATED_GENERATION_INSTRUCTION,
                                    n_related: int = N_RELATED) -> str:
    """Fill the predicate + its record kind into the instruction -> prompt."""
    context = spec.context or "a record described by free text"
    return (f"{instruction.format(n_related=n_related)}\n"
            f"RECORD KIND: {context}\n"
            f'PREDICATE: "{spec.key}"\n')


def assign_related_ids(spec: PredicateSpec, raw: list[dict],
                       n_related: int = N_RELATED) -> list["SubQuestion"]:
    """[{sub_type, text}, ...] -> SubQuestions with deterministic R-prefixed ids."""
    if len(raw) != n_related:
        raise ValueError(f"expected {n_related} related questions, got {len(raw)}")
    return [SubQuestion(f"{spec.id_prefix}_R{i}", "related", r["text"].strip())
            for i, r in enumerate(raw, start=1)]


# JSON schema for the LLM path's structured output.
_GENERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "subquestions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sub_type": {"type": "string", "enum": ["content", "meta"]},
                    "text": {"type": "string"},
                },
                "required": ["sub_type", "text"],
            },
        }
    },
    "required": ["subquestions"],
}


def build_generation_prompt(spec: PredicateSpec,
                            instruction: str = GENERATION_INSTRUCTION) -> str:
    """Fill the predicate into the generation instruction -> prompt for the LM."""
    header = instruction.format(
        n_content=N_CONTENT, n_meta=N_META, n_total=N_CONTENT + N_META,
    )
    return f'{header}\nPREDICATE: "{spec.key}"\n'


def _assign_sub_ids(spec: PredicateSpec,
                    raw: list[dict]) -> list[SubQuestion]:
    """Turn the LM's [{sub_type, text}] into SubQuestions with stable sub_ids.

    Ids follow the hand-written convention: <prefix>_C1..C3 then <prefix>_M1..M2.
    Ordering is by type (content first) so the ids are deterministic regardless
    of the order the model happened to emit them in.
    """
    content = [r for r in raw if r.get("sub_type") == "content"]
    meta = [r for r in raw if r.get("sub_type") == "meta"]
    if len(content) != N_CONTENT or len(meta) != N_META:
        raise ValueError(
            f"expected {N_CONTENT} content + {N_META} meta sub-questions, got "
            f"{len(content)} + {len(meta)}"
        )
    out: list[SubQuestion] = []
    for i, r in enumerate(content, start=1):
        out.append(SubQuestion(f"{spec.id_prefix}_C{i}", "content", r["text"].strip()))
    for i, r in enumerate(meta, start=1):
        out.append(SubQuestion(f"{spec.id_prefix}_M{i}", "meta", r["text"].strip()))
    return out


def _spec_of(predicate: Union[str, dict, PredicateSpec]) -> PredicateSpec:
    """Accept a slug, a PREDICATES-style dict, or a PredicateSpec."""
    if isinstance(predicate, PredicateSpec):
        return predicate
    if isinstance(predicate, dict):
        return PredicateSpec(slug=predicate["slug"], key=predicate["key"])
    if isinstance(predicate, str):
        p = _PREDICATES_BY_SLUG.get(predicate)
        if p is None:
            raise KeyError(
                f"unknown predicate slug {predicate!r}; known: "
                f"{sorted(_PREDICATES_BY_SLUG)}"
            )
        return PredicateSpec(slug=p["slug"], key=p["key"])
    raise TypeError(f"predicate must be str | dict | PredicateSpec, got {type(predicate)}")


class CascadeAgent:
    """Generates the 5 sub-questions for a predicate.

    mode="canned" (default, STEP 1): fake automation — returns the existing
        hand-written sub-questions. No model call. Deterministic.
    mode="llm": real automation — builds the generation prompt and calls an LM.
        Written and wired, but OFF by default and not yet plugged into the
        cascade; validate its output before trusting it.
    """

    def __init__(self, mode: str = "canned", *,
                 model: str = "gemini-2.5-pro",
                 instruction: str | None = None,
                 client=None):
        if mode not in ("canned", "llm", "related"):
            raise ValueError(f"mode must be 'canned', 'llm' or 'related', got {mode!r}")
        self.mode = mode
        self.model = model
        self.instruction = instruction or (
            RELATED_GENERATION_INSTRUCTION if mode == "related" else GENERATION_INSTRUCTION
        )
        self._client = client

    def generate(self, predicate: Union[str, dict, PredicateSpec]) -> list[SubQuestion]:
        spec = _spec_of(predicate)
        if self.mode == "canned":
            return self._canned(spec)
        if self.mode == "related":
            return self._related(spec)
        return self._llm(spec)

    # ── RELATED-QUESTION generation (paper §1.3) ─────────────────────────────
    def _related(self, spec: PredicateSpec) -> list[SubQuestion]:
        """N same-direction related questions from a STRONG model.

        Meant to be run ONCE PER PREDICATE, OFFLINE, and the result stored. That
        is what makes an expensive model affordable here: generation is
        per-predicate, while answering is per-argument. Generating at query time
        is a different (and measurably worse) system -- see
        notes/2026-08-10.md §1.4."""
        prompt = build_related_generation_prompt(spec, self.instruction)
        raw = self._call_model(prompt, schema=_RELATED_GENERATION_SCHEMA)
        return assign_related_ids(spec, raw)

    # ── STEP 1: fake automation ──────────────────────────────────────────────
    def _canned(self, spec: PredicateSpec) -> list[SubQuestion]:
        """Return the hand-written sub-questions, as if we had generated them."""
        p = _PREDICATES_BY_SLUG.get(spec.slug)
        if p is None:
            raise KeyError(
                f"canned mode has no hand-written sub-questions for {spec.slug!r}; "
                f"known: {sorted(_PREDICATES_BY_SLUG)}"
            )
        return [SubQuestion(sid, stype, text) for sid, stype, text in p["subquestions"]]

    # ── the real automation (wired, not yet trusted / used) ──────────────────
    def _llm(self, spec: PredicateSpec) -> list[SubQuestion]:
        prompt = build_generation_prompt(spec, self.instruction)
        raw = self._call_model(prompt)
        return _assign_sub_ids(spec, raw)

    def _call_model(self, prompt: str, schema: dict | None = None) -> list[dict]:
        """Send the generation prompt to Gemini, return [{sub_type, text}, ...]."""
        from google import genai
        from google.genai import types

        from connect_api import _require_key, load_env

        load_env()
        client = self._client or genai.Client(api_key=_require_key("GEMINI_API_KEY"))
        resp = client.models.generate_content(
            model=self.model,
            contents=types.Part.from_text(text=prompt),
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=schema or _GENERATION_SCHEMA,
            ),
        )
        data = json.loads(resp.text)
        return data["subquestions"]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Generate a predicate's 5 sub-questions.")
    ap.add_argument("slug", nargs="?", default="vegan",
                    help=f"predicate slug (known: {sorted(_PREDICATES_BY_SLUG)})")
    ap.add_argument("--mode", choices=["canned", "llm", "related"], default="canned",
                    help="canned = hand-written; llm = generated 3-content+2-meta; "
                         "related = generated N same-direction related questions (paper §1.3)")
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--show-prompt", action="store_true",
                    help="also print the generation prompt that would be sent")
    args = ap.parse_args()

    spec = _spec_of(args.slug)
    if args.show_prompt:
        print("─" * 68)
        print(build_generation_prompt(spec))
        print("─" * 68)

    agent = CascadeAgent(mode=args.mode, model=args.model)
    subqs = agent.generate(args.slug)
    print(f"predicate: {spec.slug}  ({spec.key!r})   mode={args.mode}")
    for sq in subqs:
        print(f"  {sq.sub_id:10s} [{sq.sub_type:7s}] {sq.text}")


if __name__ == "__main__":
    main()
