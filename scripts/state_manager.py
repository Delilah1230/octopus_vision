#!/usr/bin/env python3
"""
State table operations for the Octopimizer.

Three temporal tables (all cleared after each query run):
  predicate_store  — one row per semantic predicate (keyed by predicate_idx)
  execution_state  — one row per (product × predicate), tracks modality progress
  execution_result — one row per (product × predicate), written when LLM answers

Primary key for execution_state and execution_result: (parent_asin, predicate_idx).
No run_id needed — one run at a time, tables always cleaned up after each run.

Public API:
  initialize_state(parsed_query)                  -> None
  get_pending(modality)                           -> list[dict]
  write_result(parent_asin, predicate_idx, ...)
  cleanup()
"""

from __future__ import annotations

import psycopg2
import psycopg2.extras
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from query_parser import ParsedQuery

DB_CONFIG = {
    "host": "127.0.0.1", "port": 5432,
    "dbname": "octopus", "user": "octopus_user", "password": "octopus",
}


def _conn() -> psycopg2.extensions.connection:
    return psycopg2.connect(**DB_CONFIG)


# ── Phase 1: initialize ────────────────────────────────────────────────────────

def initialize_state(parsed_query: ParsedQuery) -> None:
    """
    Populate the three temporal tables for a new query run:
      1. predicate_store  — one row per semantic predicate
      2. execution_state  — one row per (product × predicate) that passed SQL filter

    Modality initialization per product:
      modality_text  = 0 if product has a title, else -1
      modality_image = 0 if product has_image = True, else -1
    """
    predicates = parsed_query.semantic
    if not predicates:
        return

    with _conn() as conn, conn.cursor() as cur:
        # 1. Store predicate definitions
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO predicate_store (predicate_idx, predicate_type, predicate_nl, agent_prompt)
            VALUES %s
            """,
            [(i, p.predicate_type, p.nl, p.agent_prompt) for i, p in enumerate(predicates)],
        )

        # 2. Populate execution_state: SQL result × predicates
        modality_sql = f"""
            SELECT parent_asin,
                   CASE WHEN title IS NOT NULL AND title <> '' THEN 0 ELSE -1 END AS modality_text,
                   CASE WHEN has_image = TRUE THEN 0 ELSE -1 END                  AS modality_image
            FROM ({parsed_query.sql_select}) AS _candidates
        """
        cur.execute(modality_sql)
        products = cur.fetchall()

        if products:
            rows = [
                (parent_asin, idx, mod_text, mod_image)
                for parent_asin, mod_text, mod_image in products
                for idx in range(len(predicates))
            ]
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO execution_state (parent_asin, predicate_idx, modality_text, modality_image)
                VALUES %s
                """,
                rows,
            )

        conn.commit()

    n_products = len(products) if products else 0
    print(f"[state_manager] {n_products} products × {len(predicates)} predicates "
          f"= {n_products * len(predicates)} state rows initialized")


# ── Phase 2 support ────────────────────────────────────────────────────────────

def get_candidate_asins() -> list[str]:
    """Return distinct parent_asins currently in execution_state (any modality state)."""
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT DISTINCT parent_asin FROM execution_state")
        return [row[0] for row in cur.fetchall()]


def get_pending_pairs() -> list[dict]:
    """
    Return execution_state rows whose (parent_asin, predicate_idx) has no answer yet
    in execution_result, regardless of which modality eventually answers it.
    Each row includes predicate metadata and the current modality state values.
    """
    with _conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT es.parent_asin, es.predicate_idx,
                   es.modality_text, es.modality_image,
                   ps.predicate_type, ps.predicate_nl, ps.agent_prompt
            FROM execution_state es
            JOIN predicate_store ps USING (predicate_idx)
            LEFT JOIN execution_result er
              ON er.parent_asin = es.parent_asin
             AND er.predicate_idx = es.predicate_idx
            WHERE er.parent_asin IS NULL
        """)
        return [dict(row) for row in cur.fetchall()]


def get_known_results() -> dict[str, dict[int, bool]]:
    """
    Snapshot all answers currently in execution_result, keyed by (asin, pred_idx) → bool.
    If both modalities have answers, prefer the text answer (consistent with profiling's
    soft-label preference). Used by the planner to walk the query's logic tree
    and short-circuit asins whose value is already determined.
    """
    out: dict[str, dict[int, bool]] = {}
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT parent_asin, predicate_idx, answer_text, answer_image
            FROM execution_result
        """)
        for asin, pred_idx, ans_text, ans_image in cur.fetchall():
            chosen = ans_text if ans_text is not None else ans_image
            if chosen is None:
                continue
            out.setdefault(asin, {})[pred_idx] = bool(chosen)
    return out


def get_pending(modality: str) -> list[dict]:
    """
    Return execution_state rows where the given modality is pending (= 0),
    joined with predicate_store for the predicate details.
    modality: 'text' | 'image'
    """
    col = "modality_text" if modality == "text" else "modality_image"
    with _conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(f"""
            SELECT es.parent_asin, es.predicate_idx,
                   es.modality_text, es.modality_image,
                   ps.predicate_type, ps.predicate_nl, ps.agent_prompt
            FROM execution_state es
            JOIN predicate_store ps USING (predicate_idx)
            WHERE es.{col} = 0
        """)
        return [dict(row) for row in cur.fetchall()]


def write_result(
    parent_asin: str,
    predicate_idx: int,
    answer_text: bool | None,
    answer_image: bool | None,
    confidence: float | None = None,
) -> None:
    """
    Write LLM answer to execution_result and mark the modality as processed
    in execution_state. Called by the semantic query planner after each LLM response.
    """
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO execution_result (parent_asin, predicate_idx, answer_text, answer_image, confidence)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (parent_asin, predicate_idx) DO UPDATE SET
                answer_text  = EXCLUDED.answer_text,
                answer_image = EXCLUDED.answer_image,
                confidence   = EXCLUDED.confidence,
                processed_at = NOW()
            """,
            (parent_asin, predicate_idx, answer_text, answer_image, confidence),
        )
        if answer_text is not None:
            cur.execute(
                "UPDATE execution_state SET modality_text = 1 WHERE parent_asin = %s AND predicate_idx = %s",
                (parent_asin, predicate_idx),
            )
        if answer_image is not None:
            cur.execute(
                "UPDATE execution_state SET modality_image = 1 WHERE parent_asin = %s AND predicate_idx = %s",
                (parent_asin, predicate_idx),
            )
        conn.commit()


# ── Tag-model helpers ─────────────────────────────────────────────────────────

def ensure_tag_column(table_name: str, tag_col: str) -> None:
    """Idempotent: add BOOLEAN tag column if it doesn't already exist.

    MUST NOT be called while another connection holds an open transaction that
    has touched `table_name` -- ALTER TABLE takes ACCESS EXCLUSIVE even when
    ADD COLUMN IF NOT EXISTS is a no-op, so it would wait on that transaction
    while that transaction waits on this call to return. See ensure_tag_columns().
    """
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {tag_col} BOOLEAN"
        )
        conn.commit()


def ensure_tag_columns(table_name: str, predicates) -> None:
    """Create every predicate's tag column up front, before the write transaction.

    Fixes a self-deadlock (2026-07-30) that fires whenever a query has >= 2
    semantic predicates: the tag write-back held one connection's transaction
    open across the loop and called ensure_tag_column() -- a SECOND connection --
    from inside it. On the first predicate that is harmless (the outer
    transaction has not touched the table yet), but by the second the outer
    transaction holds ROW EXCLUSIVE from writing predicate 1's tags, so the
    inner ALTER blocks on it forever. Postgres cannot break it: the cycle runs
    through the client. Single-predicate queries never hit it, which is why it
    only showed up on the multi-predicate grocery workload.
    """
    from query_parser import canonicalize_predicate_to_column   # deferred: circular import

    for pred in predicates:
        from query_parser import tag_write_column   # deferred: circular import
        ensure_tag_column(table_name, tag_write_column(pred))


def prepopulate_results_from_tags(parsed_query) -> tuple[int, dict[tuple[str, int], tuple[int, str]]]:
    """
    For every semantic predicate flagged as tag-cached (`pred.tag_column`), read
    the cached tag value AND its provenance (tag_meta) for each candidate
    asin still in execution_state, then split each non-NULL cached answer into:

      • TRUSTED (high-confidence, OR settled at the oracle rung, OR legacy with no
        provenance) → COPY into execution_result and mark the modality done, so the
        cascade short-circuits it for free (pure cross-query reuse). Same as before.

      • RE-RUN (low-confidence AND non-terminal, per RECHECK_TAU / TERMINAL_STEP —
        todo ④a) → do NOT copy. The row stays pending in execution_state so the
        cascade re-evaluates it; its (settled_tier, settled_modality) is returned so
        the planner can RESUME the escalation route at the next step instead of
        restarting from tier 1.

      • ABANDONED (2026-08-03 — tag_meta has a row but tag_value IS NULL: a
        prior query evaluated this at settled_tier and got U, cut short by a
        sibling's short-circuit before it could settle) → same RE-RUN treatment,
        unconditionally (there's no T/F to weigh a confidence threshold against).
        Distinguished from "never evaluated" by settled_tier being non-NULL.

    Note the SQL prefilter (_rewrite_sql_with_tags) already dropped high-confidence
    / oracle-terminal FALSE rows before they entered execution_state, using the SAME
    RECHECK_TAU/TERMINAL_STEP — so the two gates agree on which rows are re-run.

    Returns (n_trusted_copied, resume) where
        resume = {(parent_asin, predicate_idx): (settled_tier, settled_modality)}.
    """
    from query_parser import RECHECK_TAU, TERMINAL_STEP

    table = parsed_query.table_name
    resume: dict[tuple[str, int], tuple[int, str]] = {}
    if not table:
        return 0, resume
    tag_preds = [
        (i, p) for i, p in enumerate(parsed_query.semantic) if p.tag_column
    ]
    if not tag_preds:
        return 0, resume

    written = 0
    with _conn() as conn, conn.cursor() as cur:
        for pred_idx, pred in tag_preds:
            canon = pred.tag_column
            # tag value (category table) + provenance (side-table) for every
            # candidate still in execution_state, in one roundtrip per predicate.
            cur.execute(
                f"""
                SELECT es.parent_asin, t.{canon} AS val,
                       m.confidence, m.settled_tier, m.settled_modality
                  FROM (SELECT DISTINCT parent_asin FROM execution_state) es
                  JOIN {table} t USING (parent_asin)
                  LEFT JOIN tag_meta m
                    ON m.parent_asin = es.parent_asin AND m.predicate_canon = %s
                """,
                (canon,),
            )
            for asin, value, conf, s_tier, s_mod in cur.fetchall():
                if s_tier is None:
                    continue                       # never evaluated at all — leave pending
                if value is None:
                    # 2026-08-03: evaluated at s_tier but never settled to T/F
                    # (abandon_U, see write_tag_cache) -- there's no answer to trust
                    # outright, but we know tier s_tier already couldn't resolve it,
                    # so always resume-escalate at the NEXT step rather than paying
                    # for s_tier again from scratch.
                    resume[(asin, pred_idx)] = (s_tier, s_mod)
                    continue
                low_conf = conf is not None and conf < RECHECK_TAU.get(s_tier, 0.0)
                non_terminal = not (s_tier == TERMINAL_STEP[0] and s_mod == TERMINAL_STEP[1])
                if low_conf and non_terminal:
                    # RE-RUN: do NOT copy → stays pending; hand the planner the
                    # settling step so it can resume-escalate at the next rung.
                    resume[(asin, pred_idx)] = (s_tier, s_mod)
                    continue
                # TRUSTED: copy as settled (short-circuit reuse). A NEGATION
                # reuse inverts the cached boolean -- the cached predicate is the
                # strict complement of this one (query_parser.match_predicate_to_cached).
                if getattr(pred, "tag_negated", False):
                    value = not bool(value)
                # BROADER / NARROWER carry only ONE side of the cached answer:
                #   BROADER  -- cached implies this predicate, so a cached TRUE
                #               settles this row TRUE; a cached FALSE says nothing.
                #   NARROWER -- this predicate implies the cached one, so a cached
                #               FALSE settles it FALSE; a cached TRUE says nothing.
                # The uninformative side is left pending so the cascade evaluates
                # it. Copying it would be asserting an implication that does not
                # hold in that direction.
                inherit = getattr(pred, "tag_inherit", None)
                if inherit is not None and bool(value) != (inherit == "TRUE"):
                    continue
                cur.execute(
                    """
                    INSERT INTO execution_result
                          (parent_asin, predicate_idx, answer_text, answer_image, confidence)
                    VALUES (%s, %s, %s, NULL, NULL)
                    ON CONFLICT (parent_asin, predicate_idx) DO NOTHING
                    """,
                    (asin, pred_idx, bool(value)),
                )
                cur.execute(
                    "UPDATE execution_state SET modality_text = 1 "
                    "WHERE parent_asin = %s AND predicate_idx = %s",
                    (asin, pred_idx),
                )
                written += 1
        conn.commit()
    return written, resume


def write_tags_from_results(parsed_query) -> dict[str, int]:
    """
    Copy EVERY semantic predicate's answers from `execution_result` back to the
    category table's `tag_<canonical>` column. Idempotently creates columns
    that don't exist yet.

    Changed 2026-07-29: tag write-back is no longer gated by predicate_type.
    Previously only `objective_semantic` predicates were persisted; now every
    semantic predicate (objective OR subjective) creates/updates a reusable tag.

    Multi-modality reconciliation: when both answer_text and answer_image
    are present, take the AND (conservative). This is a v1 placeholder —
    see TODO in notes/2026-05-26.md for confidence-based reconciliation.

    Must be called BEFORE cleanup() so the data is still in execution_result.

    Returns a dict {tag_column → number of rows written}.
    """
    from query_parser import canonicalize_predicate_to_column

    table = parsed_query.table_name
    if not table:
        return {}

    # Changed 2026-07-29: was filtered to objective_semantic only; now every
    # semantic predicate gets a tag written back (reuse decoupled from type).
    tag_preds = list(enumerate(parsed_query.semantic))
    if not tag_preds:
        return {}

    # Outside the write transaction on purpose -- see ensure_tag_columns().
    ensure_tag_columns(table, [p for _, p in tag_preds])

    written: dict[str, int] = {}
    with _conn() as conn, conn.cursor() as cur:
        for pred_idx, pred in tag_preds:
            from query_parser import tag_write_column   # deferred: circular import
            tag_col = tag_write_column(pred)

            # Collect (asin, reconciled_answer) pairs from execution_result
            cur.execute(
                """
                SELECT parent_asin, answer_text, answer_image
                  FROM execution_result
                 WHERE predicate_idx = %s
                """,
                (pred_idx,),
            )
            updates = []
            for asin, ans_text, ans_image in cur.fetchall():
                # AND reconciliation when both present; else take whichever exists
                if ans_text is None and ans_image is None:
                    continue
                if ans_text is None:
                    final = bool(ans_image)
                elif ans_image is None:
                    final = bool(ans_text)
                else:
                    final = bool(ans_text) and bool(ans_image)
                updates.append((final, asin))

            if not updates:
                continue
            # Batched UPDATE
            for final_val, asin in updates:
                cur.execute(
                    f"UPDATE {table} SET {tag_col} = %s WHERE parent_asin = %s",
                    (final_val, asin),
                )
            written[tag_col] = len(updates)
        conn.commit()
    return written


def write_tag_cache(parsed_query) -> dict[str, int]:
    """Cascade tag flush (2026-07-17). Supersedes write_tags_from_results for the
    cascade path.

    Reads each semantic predicate's SETTLED answers from execution_result and writes
    to BOTH:
      * the category table's tag_<canonical> BOOLEAN column (the SQL prefilter needs
        the value there), and
      * the persistent `tag_meta` side-table (confidence / settled_tier /
        settled_modality — the provenance cross-query reuse needs but the prefilter
        does not).

    - Changed 2026-07-29: every semantic predicate is cached (subjective reuse too;
      reuse is no longer gated by objective/subjective).
    - Changed 2026-08-03: `inconclusive` (abandon_U — evaluated at some tier, cut off
      before settling by a sibling's short-circuit) rows used to be dropped entirely,
      so a later query needing the same (product,predicate) paid for that tier again
      from scratch every time. Now the category-table bool column still stays NULL
      (never fabricate a bool from a non-answer — the SQL prefilter's "don't know"
      semantics are unchanged), but `tag_meta` records the provenance
      (tag_value=NULL, confidence, settled_tier, settled_modality) so
      prepopulate_results_from_tags can resume-escalate the NEXT query at the step
      AFTER the one that gave up, instead of re-running it. Genuinely
      never-evaluated (product,predicate) pairs still get no tag_meta row at
      all (distinguished by settled_tier IS NULL on the read side).
    - NO AND-reconciliation: a cascade answer is one terminal direction, not a
      text/image pair. `tag_meta` is persistent; cleanup() does not touch it.

    Returns {tag_column -> rows written}. Call BEFORE cleanup().
    """
    from query_parser import canonicalize_predicate_to_column

    table = parsed_query.table_name
    if not table:
        return {}
    # Changed 2026-07-29: cache every semantic predicate, not just objective
    # (subjective predicates now reuse via tags too).
    tag_preds = list(enumerate(parsed_query.semantic))
    if not tag_preds:
        return {}

    # Outside the write transaction on purpose -- see ensure_tag_columns().
    ensure_tag_columns(table, [p for _, p in tag_preds])

    written: dict[str, int] = {}
    with _conn() as conn, conn.cursor() as cur:
        for pred_idx, pred in tag_preds:
            tag_col = canonicalize_predicate_to_column(pred.nl)

            cur.execute(
                """
                SELECT parent_asin, answer_text, confidence, status,
                       settled_tier, settled_modality
                  FROM execution_result
                 WHERE predicate_idx = %s
                """,
                (pred_idx,),
            )
            n = 0
            for asin, ans, conf, status, s_tier, s_mod in cur.fetchall():
                if status is None:
                    continue                       # reused-from-cache (prepopulate,
                                                   # status unset) — already in
                                                   # tag_meta; don't re-flush /
                                                   # overwrite provenance with NULLs
                if status == "inconclusive" or ans is None:
                    # 2026-08-03: U (abandon_U — evaluated at s_tier but cut off by a
                    # sibling's short-circuit before settling) used to be dropped here
                    # entirely, meaning the NEXT query needing this (product,predicate)
                    # paid for tier1 again from scratch, every time, even though we
                    # already know tier1 doesn't resolve it. There is no T/F to trust,
                    # so the category-table bool column stays untouched (still NULL —
                    # "don't know", same as never-evaluated, which is correct for the
                    # SQL prefilter). But tag_meta CAN record the provenance
                    # (settled_tier/settled_modality) with tag_value=NULL, so a later
                    # query's prepopulate_results_from_tags can resume-escalate at the
                    # NEXT step instead of re-running tier1 for nothing. Requires
                    # s_tier/s_mod to be present — an abandon_U with no tier recorded
                    # was never actually called, nothing to cache.
                    if s_tier is None:
                        continue
                    cur.execute(
                        """
                        INSERT INTO tag_meta (parent_asin, predicate_canon,
                            tag_value, confidence, settled_tier, settled_modality,
                            source_updated_at)
                        VALUES (%s, %s, NULL, %s, %s, %s, NOW())
                        ON CONFLICT (parent_asin, predicate_canon) DO UPDATE SET
                            tag_value         = NULL,
                            confidence        = EXCLUDED.confidence,
                            settled_tier      = EXCLUDED.settled_tier,
                            settled_modality  = EXCLUDED.settled_modality,
                            source_updated_at = NOW()
                        """,
                        (asin, tag_col, conf, s_tier, s_mod),
                    )
                    continue
                # 1. category-table bool (for the SQL prefilter)
                cur.execute(
                    f"UPDATE {table} SET {tag_col} = %s WHERE parent_asin = %s",
                    (bool(ans), asin),
                )
                # 2. persistent provenance (for confidence-aware reuse)
                cur.execute(
                    """
                    INSERT INTO tag_meta (parent_asin, predicate_canon,
                        tag_value, confidence, settled_tier, settled_modality,
                        source_updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (parent_asin, predicate_canon) DO UPDATE SET
                        tag_value         = EXCLUDED.tag_value,
                        confidence        = EXCLUDED.confidence,
                        settled_tier      = EXCLUDED.settled_tier,
                        settled_modality  = EXCLUDED.settled_modality,
                        source_updated_at = NOW()
                    """,
                    (asin, tag_col, bool(ans), conf, s_tier, s_mod),
                )
                n += 1
            if n:
                written[tag_col] = n
        conn.commit()
    return written


def get_tag_meta(predicate_canon: str,
                       min_confidence: float | None = None) -> dict[str, dict]:
    """Read persistent tag-cache provenance for one predicate — the confidence-aware
    REUSE primitive. With `min_confidence`, returns only cells at or above that bar
    (a stricter future query trusts high-confidence cached answers and re-derives the
    rest — including the oracle-forced low-confidence subset). Returns
    {parent_asin -> {tag_value, confidence, settled_tier, settled_modality}}."""
    sql = ("SELECT parent_asin, tag_value, confidence, settled_tier, settled_modality "
           "FROM tag_meta WHERE predicate_canon = %s")
    params: list = [predicate_canon]
    if min_confidence is not None:
        sql += " AND confidence >= %s"
        params.append(min_confidence)
    with _conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        return {r["parent_asin"]: {k: r[k] for k in
                ("tag_value", "confidence", "settled_tier", "settled_modality")}
                for r in cur.fetchall()}


# ── Cascade step-state substrate (2026-07-17) ──────────────────────────────────
#
# The cascade escalates each (product, predicate) along a route of (tier, modality)
# STEPS (notes/cascade_escalation_route.md §2). `step_state` records those steps as
# a SPARSE LONG table — one row per step, inserted only when a step is scheduled or
# run (most of the 6 possible steps never run; that is the saving). This REPLACES,
# for the cascade path, execution_state's (modality_text, modality_image) grid.
#
# Why a long table and not the old wide grid: the schedulers this enables query by
# step — "give me every (product,predicate) whose tier-1 step is pending" — which is
# a trivial `WHERE tier=1 AND state=0` here, but awkward across wide columns. It also
# grows to any #tiers/#modalities without a schema change.
#
# state: -1 N/A | 0 pending | 1 done. No 'running': a single orchestrator fetches a
# batch, runs it in memory, writes back (pending->done is atomic). 'running' would
# only matter for DISTRIBUTED workers claiming rows off a shared queue.
#
# NOTE: the cascade engine (scripts/cascade.py) currently drives its own escalation
# in memory and only needs the RESULT flushed (write_final_result). These step-state
# primitives are the substrate for FUTURE non-chunk schedulers (exhaustive-tier1,
# adaptive skip, etc. — see design §2.5); they are not on the naive cascade's hot path.

def schedule_step(parent_asin: str, predicate_idx: int, tier: int,
                  modality: str, applicable: bool = True) -> None:
    """Mark a (product, predicate, tier, modality) step as pending (0), or N/A (-1)
    when it cannot run (e.g. text_image on a product with no image). Idempotent."""
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO step_state (parent_asin, predicate_idx, tier, modality, state)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (parent_asin, predicate_idx, tier, modality) DO NOTHING
            """,
            (parent_asin, predicate_idx, tier, modality, 0 if applicable else -1),
        )
        conn.commit()


def write_step_result(parent_asin: str, predicate_idx: int, tier: int,
                      modality: str, direction: bool, confidence: float | None,
                      cost_usd: float | None = None,
                      latency_s: float | None = None) -> None:
    """Record a step's outcome and mark it done (1). Upserts, so a step scheduled
    (pending) or never scheduled both resolve to the same done row."""
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO step_state (parent_asin, predicate_idx, tier, modality,
                                    state, direction, confidence, cost_usd,
                                    latency_s, processed_at)
            VALUES (%s, %s, %s, %s, 1, %s, %s, %s, %s, NOW())
            ON CONFLICT (parent_asin, predicate_idx, tier, modality) DO UPDATE SET
                state        = 1,
                direction    = EXCLUDED.direction,
                confidence   = EXCLUDED.confidence,
                cost_usd     = EXCLUDED.cost_usd,
                latency_s    = EXCLUDED.latency_s,
                processed_at = NOW()
            """,
            (parent_asin, predicate_idx, tier, modality, direction, confidence,
             cost_usd, latency_s),
        )
        conn.commit()


def get_pending_steps(tier: int | None = None,
                      modality: str | None = None) -> list[dict]:
    """Pending (state=0) steps, optionally filtered by tier and/or modality, joined
    with predicate metadata. This is the scheduler primitive: e.g. get_pending_steps(
    tier=1) returns every (product, predicate) whose tier-1 step is waiting — the
    'run all tier-1 first' (exhaustive-cheap) execution reads exactly this."""
    clauses = ["ss.state = 0"]
    params: list = []
    if tier is not None:
        clauses.append("ss.tier = %s"); params.append(tier)
    if modality is not None:
        clauses.append("ss.modality = %s"); params.append(modality)
    where = " AND ".join(clauses)
    with _conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(f"""
            SELECT ss.parent_asin, ss.predicate_idx, ss.tier, ss.modality,
                   ps.predicate_type, ps.predicate_nl, ps.agent_prompt
            FROM step_state ss
            JOIN predicate_store ps USING (predicate_idx)
            WHERE {where}
        """, params)
        return [dict(row) for row in cur.fetchall()]


def get_step_states(parent_asin: str | None = None) -> list[dict]:
    """Snapshot of step_state rows (all, or for one product). For debugging /
    resumability inspection and for schedulers that plan over current progress."""
    with _conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        if parent_asin is None:
            cur.execute("SELECT * FROM step_state")
        else:
            cur.execute("SELECT * FROM step_state WHERE parent_asin = %s", (parent_asin,))
        return [dict(row) for row in cur.fetchall()]


def write_final_result(parent_asin: str, predicate_idx: int, status: str,
                       direction: bool | None, confidence: float | None,
                       settled_tier: int | None = None,
                       settled_modality: str | None = None) -> None:
    """Flush a (product, predicate)'s SETTLED cascade answer into execution_result.

    status ∈ {'matched', 'not_matched', 'inconclusive'} — the 3-valued Kleene
    outcome. inconclusive (U) lands here as a real status, NOT as a NULL row
    (absence of a row still means 'not processed'). direction is the T/F for a
    settled answer, or NULL for inconclusive. settled_tier/modality record which
    escalation rung produced it.
    """
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO execution_result (parent_asin, predicate_idx, answer_text,
                                          confidence, status, settled_tier,
                                          settled_modality, processed_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (parent_asin, predicate_idx) DO UPDATE SET
                answer_text      = EXCLUDED.answer_text,
                confidence       = EXCLUDED.confidence,
                status           = EXCLUDED.status,
                settled_tier     = EXCLUDED.settled_tier,
                settled_modality = EXCLUDED.settled_modality,
                processed_at     = NOW()
            """,
            (parent_asin, predicate_idx, direction, confidence, status,
             settled_tier, settled_modality),
        )
        conn.commit()


# ── Reuse-frequency ledger (2026-08-02) ──────────────────────────────────────────
# PERSISTENT (not touched by cleanup()) -- see db/schema.sql predicate_query_history
# for the full design rationale. Feeds leaf_order()'s reuse-aware AND ordering:
# a predicate many past queries have needed is worth evaluating fully (not
# short-circuited away) even at some local cost, because it builds cross-query
# cache inventory. See notes/grocery_reuse_ordering.md §9.

def _ensure_reuse_table() -> None:
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS predicate_query_history (
                predicate     TEXT NOT NULL,
                query_label   TEXT NOT NULL,
                first_seen_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (predicate, query_label)
            )
            """
        )
        conn.commit()


def get_reuse_scores(predicates: list[str], exclude_label: str) -> dict[str, float]:
    """predicate -> fraction of PAST distinct queries (excluding `exclude_label`,
    the query about to run) that also needed it. 0.0 for a predicate never seen
    before (including the very first query ever run -- there is no lookahead into
    the future, only history). Bounded [0, 1], the same scale as kill_rate, so it
    can be combined with it directly in leaf_order()."""
    if not predicates:
        return {}
    _ensure_reuse_table()
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(DISTINCT query_label) FROM predicate_query_history "
            "WHERE query_label != %s",
            (exclude_label,),
        )
        total = cur.fetchone()[0] or 0
        if total == 0:
            return {p: 0.0 for p in predicates}
        cur.execute(
            "SELECT predicate, COUNT(DISTINCT query_label) FROM predicate_query_history "
            "WHERE predicate = ANY(%s) AND query_label != %s GROUP BY predicate",
            (predicates, exclude_label),
        )
        counts = dict(cur.fetchall())
    return {p: counts.get(p, 0) / total for p in predicates}


def record_predicate_usage(predicates: list[str], query_label: str) -> None:
    """Record that `query_label` wanted each of `predicates`, for future queries'
    get_reuse_scores() to see. Idempotent (ON CONFLICT DO NOTHING) -- a query that
    runs the same predicate twice, or gets retried, does not inflate its own count."""
    if not predicates:
        return
    _ensure_reuse_table()
    with _conn() as conn, conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO predicate_query_history (predicate, query_label) VALUES %s "
            "ON CONFLICT DO NOTHING",
            [(p, query_label) for p in predicates],
        )
        conn.commit()


# ── Cleanup ────────────────────────────────────────────────────────────────────

def cleanup() -> None:
    """
    Delete all rows from the temporal tables.
    Called after a query run completes.
    """
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM execution_result")
        cur.execute("DELETE FROM step_state")
        cur.execute("DELETE FROM execution_state")
        cur.execute("DELETE FROM predicate_store")
        conn.commit()
    print("[state_manager] temporal tables cleared")
