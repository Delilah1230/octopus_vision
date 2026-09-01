#!/usr/bin/env python3
"""
Octopimizer — Octopus execution engine.

Executes a ParsedQuery end-to-end in two phases:
  Phase 1: Structural push-down
    - Execute the structural SQL against PostgreSQL
    - Initialize execution_state table (one row per product × predicate)
  Phase 2: Semantic Query Planner  [STUB — not yet implemented]
    - Read execution_state to find pending (product, predicate) pairs
    - Call LLM per pair, write results to execution_result
    - Return products where the predicate logic expression evaluates to True

Usage::
    python scripts/octopimizer.py --query "..." --table handmade_products
"""

from __future__ import annotations

import sys
from pathlib import Path

import psycopg2
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).resolve().parent))
from query_parser import ParsedQuery, parse_query
from state_manager import initialize_state, cleanup
from product_fetcher import fetch_product_content

DB_CONFIG = {
    "host": "127.0.0.1", "port": 5432,
    "dbname": "octopus", "user": "octopus_user", "password": "octopus",
}


# ── main entry point ───────────────────────────────────────────────────────────

def execute(parsed_query: ParsedQuery, **kwargs) -> list[str]:
    """
    Execute a ParsedQuery end-to-end. Returns matching parent_asin values.

    Phase 1 (structural push-down) always runs. Phase 2 is the cascade semantic
    planner. Extra kwargs (limit, chunk_size, measure_size, out_dir, ...) are
    forwarded to the planner.
    """
    if not parsed_query.table_name:
        raise ValueError("ParsedQuery has no table_name — cannot execute")

    # ── pure structural query (no semantic predicates) ─────────────────────────
    if not parsed_query.semantic:
        print("[octopimizer] No semantic predicates — returning SQL candidates directly")
        return _fetch_parent_asins(parsed_query.sql_select)

    # ── Phase 1: structural push-down → initialize execution_state ─────────────
    print(f"[octopimizer] Phase 1: SQL filter on table '{parsed_query.table_name}'")
    initialize_state(parsed_query)

    # ── Phase 2: semantic query planner (cascade) ──────────────────────────────
    try:
        results = _semantic_planner(parsed_query, **kwargs)
    finally:
        cleanup()

    return results



def _fetch_parent_asins(sql_select: str) -> list[str]:
    """Execute SQL and return all parent_asin values — used when there are no semantic predicates."""
    wrapped = f"SELECT parent_asin FROM ({sql_select}) AS _r"
    with psycopg2.connect(**DB_CONFIG) as conn, conn.cursor() as cur:
        cur.execute(wrapped)
        return [row[0] for row in cur.fetchall()]


def _semantic_planner(parsed_query: ParsedQuery, **kwargs) -> list[str]:
    """
    Phase 2: Semantic Query Planner — the CASCADE.

    Reads candidates + product text from the DB, runs the tiered confidence-gated
    cascade over the AND/OR logic tree (LLM calls, escalation route, oracle
    force-commit), writes settled answers to execution_result and flushes objective
    tags to the category table + tag_meta, and returns products where the
    logic tree evaluates TRUE. See scripts/cascade_planner.py.
    """
    from cascade_planner import run_semantic_cascade
    return run_semantic_cascade(parsed_query, **kwargs)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Octopimizer — execute a natural language query")
    ap.add_argument("--query",     required=True,            help="Natural language query")
    ap.add_argument("--table",     default=None,             help="Override target table")
    ap.add_argument("--model",     default="gemini-2.5-flash")
    ap.add_argument("--no-verify", action="store_true",      help="Skip SQL verification")
    args = ap.parse_args()

    pq = parse_query(
        args.query,
        table=args.table,
        model=args.model,
        verify=not args.no_verify,
    )

    print(f"\nQuery    : {pq.query}")
    print(f"Table    : {pq.table_name}")
    print(f"SQL      : {pq.sql_select}")
    print(f"Semantic : {len(pq.semantic)} predicate(s)")
    for i, p in enumerate(pq.semantic):
        print(f"  [{i}] [{p.predicate_type}] {p.nl!r}")

    print()
    try:
        results = execute(pq)
        print(f"\nResults ({len(results)} products):")
        for asin in results:
            print(f"  {asin}")
    except NotImplementedError as e:
        print(f"\n[Phase 2 stub] {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
