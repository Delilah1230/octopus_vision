#!/usr/bin/env python3
"""
Octopus query parser pipeline.

Accepts a plain-English query and produces a structured ParsedQuery:
  - Structural conditions: mapped to schema columns, translated to SQL, verified against sample data
  - Semantic predicates: classified as objective/subjective, assigned modality hints,
    and formatted as ready-to-use agent prompts

Pipeline steps:
  1. Schema context  — table names + column list, known at module load
  2. LLM decompose   — classify predicates, identify target table, map columns,
                       generate SQL + agent prompts
  3. SQL verify      — run SQL conditions against sample JSONL, confirm syntax + sensible results
  4. Return ParsedQuery

Usage::

  # Single query
  python scripts/query_parser.py --query "Find handmade products suitable as wedding gifts under $30"

  # Single query with explicit table override
  python scripts/query_parser.py --query "..." --table handmade_products

  # Run all built-in test queries
  python scripts/query_parser.py --run-tests

  # Run tests and write JSON output
  python scripts/query_parser.py --run-tests --json-out results/parser_test.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────

def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


SAMPLE_ROOT = _repo_root() / "dataset/AmazonReviews2023/sample_3x40"

# ── schema definition ──────────────────────────────────────────────────────────

# Maps DB table name → JSONL folder name (for sample verification)
CATEGORY_TABLES: dict[str, str] = {
    "handmade_products":        "Handmade_Products",
    "all_beauty":               "All_Beauty",
    "gift_cards":               "Gift_Cards",
    "grocery_and_gourmet_food": "Grocery_and_Gourmet_Food",
    "video_games":              "Video_Games",
}

# Columns shared across all five category tables
TABLE_COLUMNS: dict[str, dict] = {
    "price": {
        "type": "FLOAT",
        "description": "Product price in USD",
        "nl_aliases": "price, cost, budget, cheap, expensive, under $X, over $X, less than, more than",
    },
    "average_rating": {
        "type": "FLOAT (1.0–5.0)",
        "description": "Average customer rating",
        "nl_aliases": "rating, stars, highly rated, well reviewed, top rated, above X stars",
    },
    "rating_number": {
        "type": "INT",
        "description": "Number of customer ratings",
        "nl_aliases": "popular, many reviews, widely reviewed, well known",
    },
}

# ── output types ───────────────────────────────────────────────────────────────

@dataclass
class StructuralCondition:
    nl_expression: str   # original NL fragment:  "under $50"
    column: str          # schema column name:     "price"
    operator: str        # SQL operator:           "<"
    value: Any           # typed value:            50
    sql: str             # SQL condition string:   "price < 50"


@dataclass
class SemanticPredicate:
    nl: str              # NL predicate:           "suitable as a wedding gift"
    predicate_type: str  # "objective_semantic" | "subjective_semantic"
    agent_prompt: str    # ready-to-use agent prompt (positive assertion, T/F answerable)
    tag_column: str | None = None   # filled by parser: the tag_* column to READ
    tag_negated: bool = False        # True => this predicate is the NEGATION of tag_column,
                                     # so a cached TRUE means FALSE here and vice versa
    tag_relation: str | None = None  # EXACT | EQUIVALENT | NEGATION | BROADER | NARROWER
    # Which cached value carries over, for the PARTIAL relations. Under BROADER the
    # cached predicate implies this one, so a cached TRUE settles this row TRUE;
    # under NARROWER this one implies the cached, so a cached FALSE settles it
    # FALSE. The other rows are not settled and must still be evaluated -- which is
    # what makes these relations partial rather than full reuse.
    tag_inherit: str | None = None   # None | "TRUE" | "FALSE"


@dataclass
class SQLVerification:
    verified: bool
    sample_total: int
    sample_matches: int
    error: str | None = None


@dataclass
class LogicNode:
    """A boolean tree over semantic predicate indices.
    Leaves are int (pred_idx). Internal nodes are LogicNode with op + children.
    """
    op:       str                            # "AND" | "OR"
    children: list["LogicNode | int"]

    def to_dict(self) -> dict:
        return {
            "op": self.op,
            "children": [
                c.to_dict() if isinstance(c, LogicNode) else c
                for c in self.children
            ],
        }


def _build_logic(raw: Any) -> "LogicNode | int | None":
    """Recursively turn an LLM-emitted logic JSON value into LogicNode | int | None."""
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, dict) and "op" in raw and "children" in raw:
        return LogicNode(
            op=raw["op"].upper(),
            children=[_build_logic(c) for c in raw["children"]],
        )
    raise ValueError(f"malformed logic node: {raw!r}")


def _logic_to_str(node: "LogicNode | int | None") -> str:
    """Pretty-print: AND(0, OR(1, 2))."""
    if node is None:
        return "(none)"
    if isinstance(node, int):
        return str(node)
    return f"{node.op}(" + ", ".join(_logic_to_str(c) for c in node.children) + ")"


@dataclass
class ParsedQuery:
    query: str
    table_name: str | None          # DB table to query, e.g. "handmade_products"
    structural: list[StructuralCondition]
    semantic: list[SemanticPredicate]
    logic: "LogicNode | int | None" # boolean tree over semantic predicate indices
    sql_select: str                 # full SELECT statement ready for PostgreSQL
    verification: SQLVerification | None
    raw_llm_output: dict            # full LLM response for debugging

    def to_dict(self) -> dict:
        d = asdict(self)
        if isinstance(self.logic, LogicNode):
            d["logic"] = self.logic.to_dict()
        return d


# ── prompt ─────────────────────────────────────────────────────────────────────

def _table_schema_description() -> str:
    col_lines = "\n".join(
        f"    - {col} ({info['type']}): {info['description']}  [phrases: {info['nl_aliases']}]"
        for col, info in TABLE_COLUMNS.items()
    )
    table_lines = "\n".join(
        f"  - {table}" for table in CATEGORY_TABLES
    )
    return f"Available tables (all share the same columns):\n{table_lines}\n\nFilterable columns:\n{col_lines}"


_DECOMPOSE_PROMPT = """\
You are a query parser for a product search system over an Amazon product dataset.

Given a natural language query, decompose it into:
  1. Structural conditions — exact filters on existing schema columns → translate to SQL
  2. Semantic predicates  — conditions that require LLM judgment → preserve as NL strings

──────────────────────────────────────────────────
DATABASE SCHEMA:

{schema}

Non-filterable fields (require LLM to evaluate):
  Text fields : title, features (bullet list), customer review text
  Image fields: product images

Each table holds one product category. Choose the single most relevant table
based on the query subject matter. If the query spans multiple categories or
is unclear, set table_name to null.
──────────────────────────────────────────────────

PREDICATE TYPE DEFINITIONS:

"objective_semantic"
  A factual, discrete, verifiable product attribute that is NOT in the schema.
  A domain expert would give the same answer regardless of personal taste.
  Examples: material composition, hardware compatibility, skin type target,
            preparation method, dietary category.

"subjective_semantic"
  An aesthetic, emotional, or contextual judgment. Reasonable people can disagree.
  No schema column could fully capture it.
  Examples: looks luxury, suitable as a gift, attractive packaging, premium feel.

  When uncertain between the two, use "subjective_semantic".

PREDICATE SEPARATION RULE (updated 2026-07-29 — no merging):
  Keep EVERY semantic predicate separate and atomic — NEVER merge, regardless of
  objective/subjective type. Each predicate (objective OR subjective) becomes an
  INDEPENDENTLY REUSABLE attribute tag; tag creation/reuse no longer depends on
  predicate type. So every predicate must stay atomic and be connected to the
  others only through the logic tree (never by cramming "and"/"or" into one NL
  string).

NEGATION:
  Do not introduce a NOT operator in the logic tree. If a predicate is negated,
  write the negation directly into the predicate's NL string and agent_prompt
  (e.g., nl = "is not vegan", agent_prompt = "Is this product NOT vegan?").

AGENT PROMPT FORMAT:
  Each semantic predicate must become a positive assertion answerable as True/False
  for a single product.
  Example: "contains metal as the main part"
        → "Does this product contain metal as the main part?"

LOGIC TREE:
  After listing predicates, emit a boolean tree connecting them by index.
    - Leaves are integers = 0-based index into "semantic_predicates"
    - Internal nodes are objects: {{"op": "AND"|"OR", "children": [...]}}
    - The tree may be deeply nested
    - If there is only one predicate, emit just its index (e.g., 0)
    - If there are zero predicates, emit null

  Examples:
    AND of all 2 preds         → {{"op": "AND", "children": [0, 1]}}
    (0 OR 1) AND 2             → {{"op": "AND", "children": [{{"op": "OR", "children": [0, 1]}}, 2]}}
    single predicate           → 0
    no semantic predicates     → null

──────────────────────────────────────────────────
WORKED EXAMPLES (study these carefully — they encode the merging/splitting rules):

Example A — OR between objectives (KEEP THEM SEPARATE, encode OR in logic tree):
  Input: "Find products that are chocolate or cookie, are vegan, and look indulgent"
  Output:
  {{
    "table_name": "grocery_and_gourmet_food",
    "structural_conditions": [],
    "semantic_predicates": [
      {{"nl": "is chocolate",    "predicate_type": "objective_semantic",
        "agent_prompt": "Is this product chocolate?"}},
      {{"nl": "is a cookie",     "predicate_type": "objective_semantic",
        "agent_prompt": "Is this product a cookie?"}},
      {{"nl": "is vegan",        "predicate_type": "objective_semantic",
        "agent_prompt": "Is this product vegan?"}},
      {{"nl": "looks indulgent", "predicate_type": "subjective_semantic",
        "agent_prompt": "Does this product look indulgent?"}}
    ],
    "logic": {{"op": "AND", "children": [
      {{"op": "OR", "children": [0, 1]}}, 2, 3
    ]}}
  }}

Example B — multiple subjective preds ANDed (KEEP SEPARATE, connect via logic):
  Input: "Find products that look high-quality and look gift-worthy"
  Output:
  {{
    "table_name": null,
    "structural_conditions": [],
    "semantic_predicates": [
      {{"nl": "looks high-quality", "predicate_type": "subjective_semantic",
        "agent_prompt": "Does this product look high-quality?"}},
      {{"nl": "looks gift-worthy", "predicate_type": "subjective_semantic",
        "agent_prompt": "Does this product look gift-worthy?"}}
    ],
    "logic": {{"op": "AND", "children": [0, 1]}}
  }}

Example C — simple AND mixing one obj and one subj (the common case):
  Input: "Find handmade products that are made of wood and suitable as a birthday gift"
  Output:
  {{
    "table_name": "handmade_products",
    "structural_conditions": [],
    "semantic_predicates": [
      {{"nl": "is made of wood",
        "predicate_type": "objective_semantic",
        "agent_prompt": "Is this product made of wood?"}},
      {{"nl": "is suitable as a birthday gift",
        "predicate_type": "subjective_semantic",
        "agent_prompt": "Is this product suitable as a birthday gift?"}}
    ],
    "logic": {{"op": "AND", "children": [0, 1]}}
  }}

──────────────────────────────────────────────────
OUTPUT: valid JSON only, no prose, no markdown fences.

{{
  "table_name": "<one of the available table names above, or null>",
  "structural_conditions": [
    {{
      "nl_expression": "<original NL fragment from query>",
      "column":        "<exact column name from filterable columns>",
      "operator":      "<one of: <, <=, >, >=, =, !=, LIKE>",
      "value":         <typed value — number for numeric columns, string for str columns>,
      "sql":           "<complete SQL condition, e.g. price < 50>"
    }}
  ],
  "semantic_predicates": [
    {{
      "nl":             "<positive assertion NL string>",
      "predicate_type": "objective_semantic" | "subjective_semantic",
      "agent_prompt":   "<ready-to-use T/F question for the LLM agent>"
    }}
  ],
  "logic": <tree per LOGIC TREE rules above>
}}
──────────────────────────────────────────────────
Query: {query}
"""


# ── step 2: LLM decomposition ──────────────────────────────────────────────────

# ── parser cost accounting ────────────────────────────────────────────────────
# The per-ROW cascade writes subq_calls.csv, and every cost number we report was
# read from that file -- which means the PARSER and the predicate matcher, both
# of which call an LLM, were never counted. That understates us specifically:
# LOTUS and Palimpzest have no parser at all (their queries are hand-written
# operators), so this is an extra component of ours whose cost was invisible.
# It is per query, not per row, but "small" is not the same as "unreported".
PARSE_USAGE = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "cost": 0.0}


def reset_parse_usage() -> None:
    PARSE_USAGE.update(calls=0, prompt_tokens=0, output_tokens=0, cost=0.0)


def _record_parse_usage(resp: dict, model: str) -> None:
    from subquestion_probe import PRICING
    u = resp.get("usage") or {}
    pt = u.get("prompt_token_count") or u.get("prompt_tokens") or 0
    ot = ((u.get("candidates_token_count") or u.get("output_tokens") or 0)
          + (u.get("thoughts_token_count") or 0))     # thinking bills at output rate
    price = PRICING.get(model, {"input": 0.0, "output": 0.0})
    PARSE_USAGE["calls"] += 1
    PARSE_USAGE["prompt_tokens"] += pt
    PARSE_USAGE["output_tokens"] += ot
    PARSE_USAGE["cost"] += pt / 1e6 * price["input"] + ot / 1e6 * price["output"]


# Parser model. Per QUERY, never per row, so a top model is affordable here --
# this is the "spend model quality where cost does not grow with the data"
# principle, not an indulgence. Runners override it; nothing changes by default.
DEFAULT_MODEL = "gemini-2.5-flash"

# Thinking for the PARSER's own calls. gemini-3.x rejects thinking_budget=0 (that
# knob is 2.5-era); the 3.x control is thinking_level, and "minimal" is what
# actually zeroes thoughts_token_count. Thinking bills at the OUTPUT rate, and
# output is where the parser's cost lives -- 857-1921 output tokens per parse,
# most of it thought. None = provider default (thinking on).
PARSER_THINKING_LEVEL: str | None = None


def _llm_decompose(query: str, model: str) -> dict:
    """Call Gemini to decompose the query. Returns parsed JSON dict."""
    from connect_api import call_gemini_with_usage, load_env  # type: ignore

    load_env()
    prompt = _DECOMPOSE_PROMPT.format(
        schema=_table_schema_description(),
        query=query,
    )
    resp = call_gemini_with_usage(prompt, model=model, temperature=0.0,
                                  thinking_level=PARSER_THINKING_LEVEL)
    _record_parse_usage(resp, model)
    raw_text = resp.get("text", "")

    # Strip markdown fences if present
    text = raw_text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"No JSON object in LLM response:\n{raw_text[:600]}")
    return json.loads(text[start : end + 1])


# ── step 3: SQL verification ───────────────────────────────────────────────────

def _load_meta_df(table_name: str) -> pd.DataFrame:
    """Load product metadata JSONL for a table into a DataFrame.

    Handles records that span multiple lines by accumulating lines until a
    complete JSON object is parseable.
    """
    category = CATEGORY_TABLES.get(table_name)
    if not category:
        return pd.DataFrame()
    meta_file = SAMPLE_ROOT / category / f"meta_{category}_sample_40.jsonl"
    if not meta_file.exists():
        return pd.DataFrame()
    records: list[dict] = []
    buffer = ""
    for line in meta_file.read_text(encoding="utf-8").splitlines():
        buffer = (buffer + " " + line).strip() if buffer else line.strip()
        if not buffer:
            continue
        try:
            records.append(json.loads(buffer))
            buffer = ""
        except json.JSONDecodeError:
            pass  # incomplete record — accumulate next line
    return pd.DataFrame(records)


def _verify_sql(
    conditions: list[StructuralCondition],
    table_name: str | None,
) -> SQLVerification:
    """
    Run structural conditions against sample metadata and verify:
      - All column names exist in the data
      - Operators are valid
      - At least one record is evaluated (data loaded successfully)
    Returns match count and total so the caller can sanity-check selectivity.
    """
    target = table_name or next(iter(CATEGORY_TABLES))
    df = _load_meta_df(target)
    if df.empty:
        return SQLVerification(
            verified=False, sample_total=0, sample_matches=0,
            error=f"No sample data found for table '{target}'",
        )

    n_total = len(df)
    mask = pd.Series([True] * n_total, index=df.index)

    for cond in conditions:
        col = cond.column
        if col not in df.columns:
            return SQLVerification(
                verified=False, sample_total=n_total, sample_matches=0,
                error=f"Column '{col}' not in sample data. Available: {sorted(df.columns.tolist())}",
            )
        try:
            series = df[col]
            op, val = cond.operator, cond.value
            if op == "<":
                col_mask = series < val
            elif op == "<=":
                col_mask = series <= val
            elif op == ">":
                col_mask = series > val
            elif op == ">=":
                col_mask = series >= val
            elif op == "=":
                col_mask = series == val
            elif op == "!=":
                col_mask = series != val
            elif op.upper() == "LIKE":
                pattern = str(val).replace("%", ".*").replace("_", ".")
                col_mask = series.astype(str).str.contains(pattern, case=False, na=False)
            else:
                return SQLVerification(
                    verified=False, sample_total=n_total, sample_matches=0,
                    error=f"Unsupported operator '{op}' in condition: {cond.sql}",
                )
            mask = mask & col_mask.fillna(False)
        except Exception as exc:
            return SQLVerification(
                verified=False, sample_total=n_total, sample_matches=0,
                error=f"Error evaluating '{cond.sql}': {exc}",
            )

    return SQLVerification(
        verified=True,
        sample_total=n_total,
        sample_matches=int(mask.sum()),
    )


# ── build SELECT statement ─────────────────────────────────────────────────────

def _build_select(
    conditions: list[StructuralCondition],
    table_name: str | None,
) -> str:
    from_clause = f"FROM {table_name}" if table_name else "FROM <table>"
    where_parts = [c.sql for c in conditions]
    where_clause = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""
    parts = ["SELECT *", from_clause]
    if where_clause:
        parts.append(where_clause)
    return " ".join(parts)


# ── tag-model helpers ─────────────────────────────────────────────────────────

_TAG_PREFIX = "tag_"

# Semantic predicate matching (§3.2). OFF would keep the historical exact-string
# behaviour; ON adds ONE parser-model call per query that has cached predicates.
# Per query, never per row -- the same budget the rest of the parser runs on.
ENABLE_SEMANTIC_TAG_MATCH: bool = True

# Domain facts the matcher cannot infer from two predicate strings alone, but the
# SCHEMA does know. "is negative" is the strict complement of "is positive" only
# where the sentiment is binary; without being told, the model correctly refuses
# (a review could be neither). Set this from the schema, never from the answers.
PREDICATE_DOMAIN_NOTE: str = ""
_MAX_PG_IDENT_LEN = 63


def canonicalize_predicate_to_column(nl: str) -> str:
    """
    Map an objective-semantic predicate NL string to a deterministic
    Postgres-safe column name with the `tag_` prefix.

    Examples:
        "is vegan"            → "tag_is_vegan"
        "is made of wood"     → "tag_is_made_of_wood"
        "is gluten-free"      → "tag_is_gluten_free"
    """
    s = nl.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    col = _TAG_PREFIX + s
    return col[:_MAX_PG_IDENT_LEN]


def _decanonicalize(col: str) -> str:
    """"tag_is_a_positive_review" -> "is a positive review". canonicalize() only
    lowercases and squashes non-alphanumerics, so this recovers a readable phrase
    (not the original string byte-for-byte, but enough for a model to judge)."""
    return col[len(_TAG_PREFIX):].replace("_", " ").strip() if col.startswith(_TAG_PREFIX) else col


# The parser is an agent, so predicate matching is an agent task too. Until now
# the ONLY matching was `canonicalize(nl) in existing_columns` -- exact string
# equality after normalisation. That is far narrower than what §3.2 claims: it
# misses "is clearly positive" vs "is a positive review" (same property, different
# words) and it misses "is negative" vs NOT "is positive" (complementary property).
#
# ASYMMETRY IS THE WHOLE DESIGN (§3.2): wrongly claiming a relation silently
# corrupts answers, while missing one only costs money. So the instruction pushes
# hard toward NONE and demands the relation hold for EVERY record, not typically.
PREDICATE_RELATION_INSTRUCTION = """You maintain a cache of computed predicates for a database. Each cached predicate
is a yes/no property that has already been evaluated and stored for every row.

You are given ONE NEW predicate and the list of CACHED predicates. Decide whether
the new predicate can be answered from a cached one INSTEAD of being recomputed.

Answer with exactly one relation:
  "EQUIVALENT" - the new predicate is TRUE on exactly the same rows as the cached
                 one. Wording may differ; the truth condition must not.
  "NEGATION"   - the new predicate is TRUE on exactly the rows where the cached
                 one is FALSE, and FALSE where it is TRUE. They must be strict
                 complements, together covering every row with no middle ground.
  "BROADER"    - the new predicate is strictly WEAKER: every row satisfying the
                 cached predicate also satisfies the new one, but not the reverse.
                 (cached implies new.) Example: cached "is a luxury sports car",
                 new "is a car".
  "NARROWER"   - the new predicate is strictly STRONGER: every row satisfying the
                 new predicate also satisfies the cached one, but not the reverse.
                 (new implies cached.) Example: cached "is a car", new "is a
                 luxury sports car".
  "NONE"       - anything else. This is the default and the safe answer.

BROADER and NARROWER are PARTIAL: only one side of the cached answer carries over
(a TRUE under BROADER, a FALSE under NARROWER); the rest of the rows still have to
be evaluated. Claim them only when the implication is strict and holds for EVERY
row -- if it merely holds usually, answer NONE.

BEFORE answering BROADER or NARROWER, state the implication to yourself as a
sentence and check it:
  NARROWER requires: "if NEW is true of a row, then CACHED is necessarily true".
  BROADER  requires: "if CACHED is true of a row, then NEW is necessarily true".
If the sentence is false for even one imaginable row, the answer is NONE.

A related-sounding predicate is NOT automatically ordered. "is a neutral review"
next to a cached "is a positive review" is NONE, not NARROWER: a neutral review
is not a positive one, so "neutral implies positive" is false -- the two are
disjoint, not nested. Being on the same scale is not the same as being nested.

BE CONSERVATIVE. A wrong EQUIVALENT or NEGATION silently returns wrong data,
because the cached value is trusted without re-checking. A missed match only
costs a recomputation. When the two predicates merely overlap, correlate, or are
usually-but-not-always the same, answer NONE.

Specifically answer NONE when:
  - they could both be false for the same row AND neither implies the other;
  - the two merely overlap or correlate without either implying the other;
  - the match depends on a domain assumption that is not stated in the predicates.

Judge the truth conditions, not the surface words.

Output ONLY schema-conformant JSON:
  {"relation": "...", "cached_predicate": "<exactly one of the cached predicates, or empty string when relation is NONE>", "why": "<one short sentence>"}
"""

_RELATION_SCHEMA = {
    "type": "object",
    "properties": {
        "relation": {"type": "string",
                     "enum": ["EQUIVALENT", "NEGATION", "BROADER", "NARROWER", "NONE"]},
        "cached_predicate": {"type": "string"},
        "why": {"type": "string"},
    },
    "required": ["relation", "cached_predicate", "why"],
}


def match_predicate_to_cached(pred_nl: str, cached_cols: list[str],
                              model: str | None = None) -> tuple[str | None, str, str]:
    """(column_to_reuse, relation, why). relation in {EXACT, EQUIVALENT, NEGATION, NONE}.

    Exact string match short-circuits without a model call -- it is free and it
    cannot be wrong. Everything else costs ONE call PER QUERY (not per row), which
    is the same budget the rest of the parser runs on."""
    cand = canonicalize_predicate_to_column(pred_nl)
    if cand in cached_cols:
        return cand, "EXACT", "identical after normalisation"
    if not cached_cols or not ENABLE_SEMANTIC_TAG_MATCH:
        return None, "NONE", "no cached predicates" if not cached_cols else "disabled"

    by_phrase = {_decanonicalize(c): c for c in sorted(cached_cols)}
    listing = "\n".join(f"  - {p}" for p in by_phrase)
    domain = (f"\nDOMAIN FACTS (from the schema — you may rely on these):\n"
              f"{PREDICATE_DOMAIN_NOTE}\n" if PREDICATE_DOMAIN_NOTE else "")
    prompt = (f"{PREDICATE_RELATION_INSTRUCTION}{domain}\n"
              f"NEW PREDICATE: \"{pred_nl}\"\n\nCACHED PREDICATES:\n{listing}\n")

    from connect_api import call_gemini_with_usage, load_env
    try:
        load_env()
        _m = model or DEFAULT_MODEL
        resp = call_gemini_with_usage(prompt, model=_m, temperature=0.0,
                                      thinking_level=PARSER_THINKING_LEVEL)
        _record_parse_usage(resp, _m)
        text = (resp.get("text") or "").strip()
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fence:
            text = fence.group(1).strip()
        a, b = text.find("{"), text.rfind("}")
        data = json.loads(text[a:b + 1])
    except Exception as exc:                                    # noqa: BLE001
        print(f"[parser] predicate-relation match failed ({type(exc).__name__}: {exc}); "
              f"treating as NONE", file=sys.stderr)
        return None, "NONE", f"match error: {exc}"

    rel = data.get("relation", "NONE")
    phrase = (data.get("cached_predicate") or "").strip()
    why = (data.get("why") or "").strip()
    if rel == "NONE" or phrase not in by_phrase:
        if rel != "NONE":
            print(f"[parser] relation {rel} named an unknown cached predicate "
                  f"{phrase!r}; treating as NONE", file=sys.stderr)
        return None, "NONE", why
    return by_phrase[phrase], rel, why


def tag_write_column(pred) -> str:
    """The column a predicate's OWN results belong in.

    EQUIVALENT reuse ADOPTS the cached column: same truth condition, so writing a
    second column would duplicate the data and leave the two able to drift apart.
    NEGATION does NOT adopt -- the values are inverted, so they must live in this
    predicate's own column or the cached one would be corrupted."""
    col = getattr(pred, "tag_column", None)
    partial = getattr(pred, "tag_inherit", None) is not None
    if col and not getattr(pred, "tag_negated", False) and not partial:
        return col
    return canonicalize_predicate_to_column(pred.nl)


def _existing_tag_columns(table_name: str) -> set[str]:
    """
    Read information_schema and return the set of columns on `table_name`
    whose name starts with `tag_` AND HOLD AT LEAST ONE VALUE. Returns an empty
    set if the table doesn't exist or has no tag columns.

    The non-NULL requirement matters: an all-NULL tag column is a leftover from a
    previous run or a predicate that was registered but never evaluated. It is not
    a cache -- there is nothing in it to reuse -- and offering it to the predicate
    matcher invites binding a new predicate to an empty column, which would look
    like reuse and deliver nothing.
    """
    if not table_name:
        return set()
    import psycopg2
    try:
        conn = psycopg2.connect(
            host="127.0.0.1", port=5432,
            dbname="octopus", user="octopus_user", password="octopus",
        )
    except Exception as exc:
        print(f"[parser] could not inspect tag columns: {exc}", file=sys.stderr)
        return set()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = %s AND column_name LIKE %s",
                (table_name, f"{_TAG_PREFIX}%"),
            )
            cols = [row[0] for row in cur.fetchall()]
            if not cols:
                return set()
            # Keep only columns that actually HOLD something. An all-NULL tag column
            # is a leftover, not a cache: binding to it looks like reuse and delivers
            # nothing, and worse, it lets an EXACT name match shadow a real semantic
            # match against a populated column (measured: Q8's "is a negative review"
            # bound EXACT to an empty tag_is_a_negative_review instead of NEGATION
            # against the populated tag_is_a_positive_review).
            counts = ", ".join(f"count({c}) AS {c}" for c in cols)
            cur.execute(f"SELECT {counts} FROM {table_name}")
            row = cur.fetchone() or ()
            return {c for c, n in zip(cols, row) if n}
    finally:
        conn.close()


# Confidence-recheck policy (todo ④a, defined 2026-07-21). Single source of truth —
# state_manager.prepopulate imports these so the SQL "keep for re-run" gate and the
# prepopulate "don't copy → resume" gate agree on the SAME rows. See
# notes/cache_confidence_recheck.md.
RECHECK_TAU = {1: 0.9, 2: 0.8, 3: 0.0}      # per-settled_tier trust bar; below → re-run
TERMINAL_STEP = (3, "text_image")            # route's oracle rung; force-committed, never rechecked


def _tau_case_sql(alias: str) -> str:
    """SQL CASE mapping settled_tier → its recheck τ (default 0.0 = trust)."""
    whens = " ".join(f"WHEN {t} THEN {v}" for t, v in sorted(RECHECK_TAU.items()))
    return f"CASE {alias}.settled_tier {whens} ELSE 0.0 END"


def _rewrite_sql_with_tags(sql_select: str, tag_cols: list) -> str:
    """
    Wrap the base candidate SELECT so each cached tag column acts as a prefilter
    WITH a per-tier confidence recheck (todo ④a). A row is KEPT in the candidate
    set iff, for every cached predicate:

        tag = TRUE            (a match — must stay in the result set), OR
        tag IS NULL           (never evaluated), OR
        LOW-CONFIDENCE & NON-TERMINAL  (m.confidence < τ[settled_tier]  AND
                                        the settling step was not the oracle rung
                                        (3, text_image)) — re-enters so the cascade
                                        can resume-escalate it.

    Only DROPPED: FALSE that is high-confidence OR settled at the oracle rung.
    This is what stops a wrong FALSE from being filtered out forever (the
    unfalsifiable-FALSE problem). Rows with no tag_meta row (legacy, no
    provenance) have m.confidence = NULL → recheck clause is false → the bool is
    trusted as-is (TRUE/NULL kept, FALSE dropped), matching prior behaviour.

    Implemented as an outer wrapper (`SELECT _base.* FROM (<base>) _base LEFT JOIN
    tag_meta ...`) so downstream (initialize_state re-wraps this) still sees
    exactly the category-table columns and never the meta columns.
    """
    if not tag_cols:
        return sql_select

    # Accept ("col", negated) pairs; a bare string means not negated.
    tag_cols = [(c, False, None) if isinstance(c, str) else
                (c + (None,) if len(c) == 2 else c) for c in tag_cols]
    joins, clauses = [], []
    for i, (c, negated, inherit) in enumerate(tag_cols):
        m = f"_m{i}"
        joins.append(
            f"LEFT JOIN tag_meta {m} "
            f"ON {m}.parent_asin = _base.parent_asin AND {m}.predicate_canon = '{c}'"
        )
        recheck = (
            f"({m}.confidence < {_tau_case_sql(m)} "
            f"AND NOT ({m}.settled_tier = {TERMINAL_STEP[0]} "
            f"AND {m}.settled_modality = '{TERMINAL_STEP[1]}'))"
        )
        if inherit == "FALSE":
            # NARROWER: the new predicate implies the cached one, so a cached FALSE
            # settles this row FALSE and it can never match -- drop it. A cached
            # TRUE says nothing (the row may or may not clear the higher bar), so
            # those rows stay and are evaluated.
            clauses.append(f"(_base.{c} IS NOT FALSE OR {recheck})")
        elif inherit == "TRUE":
            # BROADER: a cached TRUE settles this row TRUE, but that makes it an
            # ANSWER, not something to drop. Everything else still needs judging.
            # So nothing is filtered out here; prepopulate does the settling.
            pass
        else:
            # A NEGATION reuse flips which cached value means "this row can match":
            # the new predicate holds exactly where the cached one is FALSE.
            keep = "FALSE" if negated else "TRUE"
            clauses.append(f"(_base.{c} = {keep} OR _base.{c} IS NULL OR {recheck})")
    if not clauses:                    # BROADER only: nothing to filter out
        return sql_select
    return (
        f"SELECT _base.* FROM ({sql_select.rstrip(';')}) _base "
        + " ".join(joins)
        + " WHERE " + " AND ".join(clauses)
    )


# ── main pipeline ──────────────────────────────────────────────────────────────

def parse_query(
    query: str,
    *,
    table: str | None = None,
    model: str | None = None,
    verify: bool = True,
) -> ParsedQuery:
    """
    Full query parser pipeline.

    Step 1 — Schema context: table names + shared columns, loaded at module level.
    Step 2 — LLM decomposition: classify predicates, identify table, map columns,
              generate SQL + agent prompts.
    Step 3 — SQL verification: run conditions against sample data (if verify=True).
    Step 4 — Build and return ParsedQuery.
    """
    # ── step 2: LLM decomposition ──────────────────────────────────────────────
    print(f"[parser] decomposing: {query!r}", file=sys.stderr)
    model = model or DEFAULT_MODEL
    raw = _llm_decompose(query, model=model)

    table_name = table or raw.get("table_name") or None

    structural: list[StructuralCondition] = []
    for item in raw.get("structural_conditions", []):
        try:
            structural.append(StructuralCondition(
                nl_expression=item["nl_expression"],
                column=item["column"],
                operator=item["operator"],
                value=item["value"],
                sql=item["sql"],
            ))
        except KeyError as exc:
            print(f"[parser] warning: skipping malformed structural condition {item}: {exc}", file=sys.stderr)

    semantic: list[SemanticPredicate] = []
    for item in raw.get("semantic_predicates", []):
        try:
            semantic.append(SemanticPredicate(
                nl=item["nl"],
                predicate_type=item["predicate_type"],
                agent_prompt=item["agent_prompt"],
            ))
        except KeyError as exc:
            print(f"[parser] warning: skipping malformed semantic predicate {item}: {exc}", file=sys.stderr)

    sql_select = _build_select(structural, table_name)

    # ── step 2b: tag-model integration ────────────────────────────────────────
    # For EVERY semantic predicate (any type). Changed 2026-07-29: reuse is no
    # longer gated by objective/subjective — subjective predicates are cached and
    # reused as tags just like objective ones. If a matching tag column exists on
    # the target table:
    #   • mark pred.tag_column so downstream code knows it's tag-backed
    #   • rewrite the SQL to filter on the tag (drop False, keep TRUE+NULL)
    tag_columns_to_use: list[tuple[str, bool]] = []
    if table_name:
        existing_tags = sorted(_existing_tag_columns(table_name))
        for pred in semantic:
            col, rel, why = match_predicate_to_cached(pred.nl, existing_tags, model=model)
            if col is None:
                continue
            pred.tag_column, pred.tag_relation = col, rel
            pred.tag_negated = (rel == "NEGATION")
            pred.tag_inherit = {"BROADER": "TRUE", "NARROWER": "FALSE"}.get(rel)
            tag_columns_to_use.append((col, pred.tag_negated, pred.tag_inherit))
            arrow = "NOT " if pred.tag_negated else ""
            extra = (f", inheriting only the cached {pred.tag_inherit} rows"
                     if pred.tag_inherit else "")
            print(f"[parser] tag-cached [{rel}]: pred {pred.nl!r} → {arrow}{col}{extra}"
                  + (f"  ({why})" if rel != "EXACT" else ""), file=sys.stderr)
        if tag_columns_to_use:
            sql_select = _rewrite_sql_with_tags(sql_select, tag_columns_to_use)
            print(f"[parser] SQL rewritten with {len(tag_columns_to_use)} tag clause(s):\n        {sql_select}",
                  file=sys.stderr)

    # ── step 3: SQL verification ───────────────────────────────────────────────
    verification: SQLVerification | None = None
    if verify and structural:
        print("[parser] verifying SQL conditions against sample data...", file=sys.stderr)
        verification = _verify_sql(structural, table_name)
        if not verification.verified:
            print(f"[parser] SQL verification FAILED: {verification.error}", file=sys.stderr)
        else:
            print(
                f"[parser] SQL verified — {verification.sample_matches}/{verification.sample_total}"
                " records match on sample",
                file=sys.stderr,
            )

    try:
        logic = _build_logic(raw.get("logic"))
    except ValueError as exc:
        print(f"[parser] warning: malformed logic tree ({exc}); falling back to None", file=sys.stderr)
        logic = None

    return ParsedQuery(
        query=query,
        table_name=table_name,
        structural=structural,
        semantic=semantic,
        logic=logic,
        sql_select=sql_select,
        verification=verification,
        raw_llm_output=raw,
    )


# ── display ────────────────────────────────────────────────────────────────────

def _print_plan(plan: ParsedQuery) -> None:
    print(f"\n  Query    : {plan.query}")
    print(f"  Table    : {plan.table_name or '(none)'}")
    print(f"  SQL      : {plan.sql_select}")
    if plan.verification:
        v = plan.verification
        status = "OK" if v.verified else f"FAILED — {v.error}"
        print(f"  Verify   : {status}  ({v.sample_matches}/{v.sample_total} sample records match)")

    print(f"\n  Structural conditions ({len(plan.structural)}):")
    if plan.structural:
        for c in plan.structural:
            print(f"    [{c.column}]  {c.nl_expression!r}  →  {c.sql}")
    else:
        print("    (none)")

    print(f"\n  Semantic predicates ({len(plan.semantic)}):")
    if plan.semantic:
        for i, p in enumerate(plan.semantic):
            print(f"    [{i}] [{p.predicate_type}]")
            print(f"        nl           : {p.nl!r}")
            print(f"        agent_prompt : {p.agent_prompt!r}")
    else:
        print("    (none)")

    print(f"\n  Logic tree:")
    print(f"    {_logic_to_str(plan.logic)}")


# ── test queries ───────────────────────────────────────────────────────────────

# Covers: pure semantic, mixed structural+semantic, pure structural
TEST_QUERIES: list[tuple[str, str | None]] = [
    # -- pure semantic (maps to existing Q1-Q7 soft-label queries) --
    ("Find all products that contain metal as the main part.", "handmade_products"),
    ("Find products designed for customers with sensitive skin.", "all_beauty"),
    ("Find all beauty products whose packaging visually suggests a premium or luxury product.", "all_beauty"),
    ("Find games, devices, or accessories that are compatible with Nintendo Switch.", "video_games"),
    ("Find food that is required to be cooked or processed before eating.", "grocery_and_gourmet_food"),
    # -- mixed: structural + semantic --
    ("Find handmade products suitable as wedding gifts under $30.", "handmade_products"),
    ("Find beauty products for sensitive skin with an average rating above 4.", "all_beauty"),
    ("Find food products that need cooking, priced under $15.", "grocery_and_gourmet_food"),
    ("Find luxury beauty products under $50 with more than 100 reviews.", "all_beauty"),
    ("Find highly rated handmade products that look suitable as anniversary gifts.", "handmade_products"),
    # -- pure structural --
    ("Find products under $10 with a rating above 4.5.", None),
    ("Find products with more than 500 reviews.", None),
]


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Octopus query parser")
    ap.add_argument("--query", type=str, default=None, help="NL query to parse")
    ap.add_argument("--table", type=str, default=None, help="Override target table name")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--no-verify", action="store_true", help="Skip SQL verification step")
    ap.add_argument("--run-tests", action="store_true", help="Run all built-in test queries")
    ap.add_argument("--json-out", type=Path, default=None, help="Write results to JSON file")
    args = ap.parse_args()

    if args.run_tests:
        results = []
        for i, (query, tbl) in enumerate(TEST_QUERIES, 1):
            print(f"\n{'='*72}")
            print(f"[{i}/{len(TEST_QUERIES)}] {query}")
            print(f"table: {tbl or '(auto-detect)'}")
            try:
                plan = parse_query(query, table=tbl, model=args.model, verify=not args.no_verify)
                _print_plan(plan)
                results.append(plan.to_dict())
            except Exception as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                results.append({"query": query, "error": str(exc)})
        print(f"\n{'='*72}")
        print(f"Completed {len(results)}/{len(TEST_QUERIES)} queries.")
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"Wrote {args.json_out}")
        return

    if not args.query:
        ap.error("Provide --query TEXT or --run-tests")

    plan = parse_query(
        args.query,
        table=args.table,
        model=args.model,
        verify=not args.no_verify,
    )
    _print_plan(plan)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
