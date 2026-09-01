#!/usr/bin/env python3
"""
Fetch one product + its reviews from PostgreSQL and return a formatted text block
for LLM consumption. Shared by sampled_profiling and full_evaluation modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import psycopg2
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).resolve().parent))
from query_record_filter import format_amazon_product_block

DB_CONFIG = {
    "host": "127.0.0.1", "port": 5432,
    "dbname": "octopus", "user": "octopus_user", "password": "octopus",
}


def fetch_product_content(parent_asin: str, table_name: str) -> str:
    """
    Returns a formatted text block (title + features + reviews) for the given product.
    Image modality is not included here — see fetch_product_image_url for that.
    """
    # Movie scenario (SemBench): the "content" is a single review's text. There is
    # no Amazon-style features/reviews join. Added 2026-07-29 for CIDR eval.
    if table_name == "movie_reviews":
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT review_text FROM movie_reviews WHERE parent_asin = %s",
                    (parent_asin,),
                )
                row = cur.fetchone()
                if row is None:
                    raise ValueError(f"parent_asin {parent_asin!r} not found in movie_reviews")
                return f"Review: {row[0]}"

    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"SELECT * FROM {table_name} WHERE parent_asin = %s", (parent_asin,))
            product_row = cur.fetchone()
            if product_row is None:
                raise ValueError(f"parent_asin {parent_asin!r} not found in {table_name}")

            cur.execute(
                "SELECT title, text, rating, helpful_vote FROM reviews WHERE parent_asin = %s",
                (parent_asin,),
            )
            review_rows = cur.fetchall()

    meta = {
        "parent_asin": product_row["parent_asin"],
        "title":       product_row.get("title"),
        "features":    product_row.get("features") or [],
        "images":      [],
    }
    reviews = [
        {"title": r["title"], "text": r["text"], "images": []}
        for r in review_rows
    ]
    text_block, _ = format_amazon_product_block(meta, reviews)
    return text_block


def fetch_product_image_url(parent_asin: str, table_name: str) -> str | None:
    """
    Returns the main product image URL (or None if the product has no image).
    """
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT image_url FROM {table_name} WHERE parent_asin = %s",
                (parent_asin,),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"parent_asin {parent_asin!r} not found in {table_name}")
            return row[0]


def fetch_candidates_with_image(table_name: str, candidates: list[str]) -> list[str]:
    """Filter candidates down to those that have a non-null image_url."""
    if not candidates:
        return []
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT parent_asin FROM {table_name} "
                f"WHERE parent_asin = ANY(%s) AND image_url IS NOT NULL",
                (candidates,),
            )
            return [row[0] for row in cur.fetchall()]


# ── SourceSpec-aware fetch (used by the new pipeline) ─────────────────────────

from dataclasses import dataclass


@dataclass
class FetchedContent:
    """Result of a SourceSpec-driven fetch.
    Multiple image URLs are supported (one per image-modality source); the
    caller picks one or sends all to a multimodal model.
    """
    text_block:  str | None
    image_urls:  list[str]


# Operators allowed in SourceTable.filters (the value side). Each filter is
# a {column: "<op> <value>"} string; we only parse a small, safe vocabulary.
_FILTER_OPS = (">=", "<=", "!=", ">", "<", "=")


def _build_where(join_on: str | None,
                 join_value: str,
                 filters: dict | None) -> tuple[str, list]:
    """Build a parameterized WHERE clause for a single source's SELECT."""
    parts: list[str] = []
    params: list = []
    if join_on:
        parts.append(f"{join_on} = %s")
        params.append(join_value)
    if filters:
        for col, expr in filters.items():
            expr = (expr or "").strip()
            for op in _FILTER_OPS:
                if expr.startswith(op):
                    val = expr[len(op):].strip()
                    parts.append(f"{col} {op} %s")
                    params.append(_coerce_filter_value(val))
                    break
            else:
                # No operator → treat as equality.
                parts.append(f"{col} = %s")
                params.append(_coerce_filter_value(expr))
    return (" AND ".join(parts) if parts else "TRUE", params)


def _coerce_filter_value(val: str):
    """Best-effort: try int → float → strip quotes → leave as string."""
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val.strip("\"'")


def _format_value(v) -> str:
    """Stringify a SELECT result for inclusion in the text block."""
    if v is None:
        return ""
    if isinstance(v, list):
        # Postgres TEXT[] arrays come through as Python lists
        return "\n".join(f"- {str(x).strip()}" for x in v if x)
    return str(v).strip()


def fetch_with_source_spec(
    join_value:   str,
    source_spec,                   # profiling_agent.SourceSpec
    primary_table: str | None = None,
) -> FetchedContent:
    """
    Pull data from each SourceTable in `source_spec` and assemble:
      - text_block: a formatted plain-text block, one section per text source
      - image_urls: list of image URLs from image-modality sources

    `join_value` is the primary-key value (e.g., parent_asin).
    `primary_table` (optional) tells us which source is the primary table —
    primary-table sources are queried with `WHERE parent_asin = %s`; aux
    tables are queried with `WHERE <join_on> = %s`. If primary_table is None,
    we infer it as "the source whose join_on is None".
    """
    text_sections: list[str] = []
    image_urls:    list[str] = []

    with psycopg2.connect(**DB_CONFIG) as conn, \
         conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:

        for src in source_spec.tables:
            cols_sql = ", ".join(src.columns)
            join_on = src.join_on
            if join_on is None:
                # Primary table: filter on parent_asin
                where_sql, params = _build_where("parent_asin", join_value, src.filters)
            else:
                where_sql, params = _build_where(join_on, join_value, src.filters)
            sql = f"SELECT {cols_sql} FROM {src.table} WHERE {where_sql}"
            cur.execute(sql, params)
            rows = cur.fetchall()

            if src.modality == "image":
                for row in rows:
                    for col in src.columns:
                        v = row.get(col)
                        if v:
                            image_urls.append(str(v))
            else:  # text
                section_lines = [f"## SOURCE: {src.table}"]
                if not rows:
                    section_lines.append("(no rows matched)")
                else:
                    for i, row in enumerate(rows):
                        if len(rows) > 1:
                            section_lines.append(f"### row {i+1}")
                        for col in src.columns:
                            val = _format_value(row.get(col))
                            if not val:
                                continue
                            if "\n" in val:
                                section_lines.append(f"{col}:")
                                section_lines.append(val)
                            else:
                                section_lines.append(f"{col}: {val}")
                text_sections.append("\n".join(section_lines))

    text_block = ("\n\n".join(text_sections)) if text_sections else None
    return FetchedContent(text_block=text_block, image_urls=image_urls)


def estimate_tokens(text: str | None) -> int:
    """Rough char→token estimate (4 chars/token), consistent with cost scripts."""
    if not text:
        return 0
    return len(text) // 4
