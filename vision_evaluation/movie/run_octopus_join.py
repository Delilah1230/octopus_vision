#!/usr/bin/env python3
"""
Octopus on SemBench movie Q5/Q6/Q7 — the honest two-step, SINGLE-STAGE cascade.

Our parser cannot express a row-pair predicate: it flattens "pairs with the same
sentiment" into OR(expresses positive, expresses negative), a tautology matching
every row, and octopimizer.execute() returns a flat list of row keys with no
place for a pair to live. So the SEMANTIC work (sentiment per ant_man review)
goes through the real cascade exactly as in Q2, and the PAIRING is a plain SQL
self-join over the resulting tag.

That IS the result -- the tag collapses a semantic join into a relational one --
but the plan step is manual and the paper must say so.

Pairs are emitted as distinct UNORDERED pairs (r1.review_id < r2.review_id); the
evaluator norms both sides to unordered sets, so this matches GT without wasting
Q5/Q6's 10-row budget on (a,b)/(b,a) twins.

    python vision_evaluation/movie/run_octopus_join.py --clear-tags
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import protocol as P

P.configure_movie_schema()
P.configure_octopus()          # MUST precede the octopimizer/planner imports below

import psycopg2                                                    # noqa: E402
import octopimizer                                                 # noqa: E402
import query_parser as qp                                          # noqa: E402
import state_manager as sm                                         # noqa: E402

TAG_COL = "tag_is_a_positive_review"

# ORDER BY is not cosmetic here. Q5/Q6 ask for TEN pairs out of thousands, so the
# score is decided entirely by WHICH ten come back -- and without an ORDER BY that
# is Postgres's physical row order, i.e. not reproducible and not meaningful. The
# same system could score 1.000 or 0.600 on Q6 with no change to a single answer.
#
# Two orderings, and the difference between them is itself the measurement:
#   "id"    -- lexicographic by review id. Neutral and deterministic. Nothing about
#              the answer influences which pairs are chosen, so this is the honest
#              default: it reports what the system knows, not what it knows best.
#   "conf"  -- most-confident pairs first, where a pair is only as confident as its
#              weaker member (min of the two). This is a real capability the tag
#              store enables and arguably what a system SHOULD do when asked for k
#              out of many. But it RAISES a limit-k score without improving a single
#              underlying judgement, so it is opt-in and must be declared when used.
PAIR_SQL = f"""
WITH pairs AS (
  -- DISTINCT because the published sf_2000 ships duplicate review rows: two
  -- copies each of two reviews produce FOUR identical (rid1, rid2) rows. The
  -- evaluator scores with set semantics, so those copies cannot earn anything --
  -- they can only burn slots in a LIMIT 10 answer. Measured: Q6 submitted 10 rows
  -- that were only 3 distinct pairs, capping recall at 0.3 before a single
  -- judgement was considered.
  SELECT DISTINCT r1.id, r1.review_id AS rid1, r2.review_id AS rid2,
         LEAST(COALESCE(m1.confidence, 0), COALESCE(m2.confidence, 0)) AS pair_conf
    FROM movie_reviews r1
    JOIN movie_reviews r2
      ON r1.id = r2.id
     AND r1.review_id < r2.review_id
    LEFT JOIN tag_meta m1 ON m1.parent_asin = r1.parent_asin
                         AND m1.predicate_canon = '{TAG_COL}'
    LEFT JOIN tag_meta m2 ON m2.parent_asin = r2.parent_asin
                         AND m2.predicate_canon = '{TAG_COL}'
   WHERE r1.id = %s
     AND r1.{TAG_COL} IS NOT NULL
     AND r2.{TAG_COL} IS NOT NULL
     AND r1.{TAG_COL} {{op}} r2.{TAG_COL}
),
ranked AS (
  -- Rank within each left-hand review, so ordering by rn takes ONE pair from each
  -- review before taking a second from any. Plain lexicographic ordering put all
  -- ten of Q6's pairs on a SINGLE left review: a degenerate answer set, and one
  -- where a single misjudged row would have taken all ten down with it.
  SELECT *, row_number() OVER (PARTITION BY rid1 ORDER BY rid2) AS rn FROM pairs
)
SELECT id, rid1, rid2 FROM ranked
 ORDER BY {{order}}
"""
ORDERINGS = {
    "id":   "rn, rid1, rid2",
    "conf": "pair_conf DESC, rn, rid1, rid2",
}


def db(sql, params=(), commit=False):
    with psycopg2.connect(**P.DB) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        if commit:
            conn.commit()
            return cur.rowcount
        return cur.fetchall()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clear-tags", action="store_true")
    ap.add_argument("--reset-tags", action="store_true",
                    help="wipe the ENTIRE movie tag store first")
    ap.add_argument("--workers", type=int, default=P.WORKERS)
    ap.add_argument("--batch", type=int, default=P.BATCH)
    ap.add_argument("--chunk-size", type=int, default=P.CHUNK_SIZE)
    ap.add_argument("--order", choices=sorted(ORDERINGS), default="id",
                    help="which pairs the limit-10 queries return. 'id' (default) is "
                         "neutral and deterministic; 'conf' returns the most confident "
                         "pairs first, which raises Q5/Q6 without improving any "
                         "underlying judgement -- declare it if you use it.")
    ap.add_argument("--share-over", type=int, default=3,
                    help="how many queries this one semantic pass serves. 3 is the "
                         "normal case: one invocation answers Q5, Q6 and Q7, so its "
                         "cost is split three ways. Pass 1 when the invocation serves "
                         "ONE query -- e.g. the no-tag ablation runs the join three "
                         "separate times, and dividing there would under-report each "
                         "by 3x.")
    ap.add_argument("--out-suffix", default="")
    a = ap.parse_args()

    out = P.out_dir("octopus", a.out_suffix)
    P.verify_row_parity()

    if a.reset_tags:
        P.reset_tag_store()
    elif a.clear_tags:
        db(f"UPDATE movie_reviews SET {TAG_COL}=NULL WHERE id=%s", (P.ANTMAN,), commit=True)
        db("DELETE FROM tag_meta WHERE parent_asin IN "
           "(SELECT parent_asin FROM movie_reviews WHERE id=%s)", (P.ANTMAN,), commit=True)
        print(f"[join] cleared tags for {P.ANTMAN}")

    P.warn_on_stale_tags()

    n_total = db("SELECT COUNT(*) FROM movie_reviews WHERE id=%s", (P.ANTMAN,))[0][0]
    print(f"[join] {P.ANTMAN}: {n_total} reviews")

    # ── step 1: the semantic op, through the real cascade (same as Q2) ─────────
    nl = f'Positive reviews for movie "{P.ANTMAN}". Return reviewId.'
    pq = qp.parse_query(nl, table="movie_reviews", verify=False)
    print(f"[join] parsed: structural={[c.sql for c in pq.structural]} "
          f"semantic={[(p.nl, p.predicate_type) for p in pq.semantic]}")
    sm.cleanup()
    P.reset_calls_log()
    t0 = time.time()
    matches = octopimizer.execute(pq, batch=a.batch, chunk_size=a.chunk_size,
                                  workers=a.workers)
    t_sem = time.time() - t0
    sem_metrics = P.read_calls_log()
    n_tagged = db(f"SELECT COUNT(*) FROM movie_reviews WHERE id=%s AND "
                  f"{TAG_COL} IS NOT NULL", (P.ANTMAN,))[0][0]
    print(f"[join] semantic step: {len(matches)} positive / {n_total}, "
          f"{n_tagged} tagged, {t_sem:.1f}s")

    # ── step 2: pairing = plain SQL over the tag ───────────────────────────────
    for qid, op, limit in [(5, "=", 10), (6, "<>", 10), (7, "<>", None)]:
        t0 = time.time()
        sql = (PAIR_SQL.format(op=op, order=ORDERINGS[a.order])
               + (f" LIMIT {limit}" if limit else ""))
        rows = [r[:3] for r in db(sql, (P.ANTMAN,))]      # drop pair_conf
        t_sql = time.time() - t0
        with (out / f"Q{qid}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "rid1", "rid2"])
            w.writerows(rows)
        print(f"[join] Q{qid}: {len(rows)} pairs in {t_sql*1000:.1f}ms  "
              f"(0 LLM calls — pure SQL on the tag, order={a.order})")
        # The semantic step is paid ONCE and shared by Q5/Q6/Q7. Charging it to
        # whichever query happens to be first would make the other two look free
        # for the wrong reason, so split it three ways and say so.
        # Split the shared semantic step across Q5-Q7 in BOTH dimensions. It used
        # to divide latency by 3 while read_calls_log() handed each query the FULL
        # cost, so the one payment was counted three times in any sum.
        P.record_metrics(qid, t_sem / a.share_over + t_sql,
                         {"n_pairs": len(rows), "semantic_step_shared_over": a.share_over,
                          "pair_order": a.order,
                          "note": f"semantic cost AND latency split {a.share_over} "
                                  f"way(s); the pairing itself is 0 LLM calls"},
                         a.out_suffix, cost_divisor=a.share_over)

    print(f"\n[join] semantic step totals: ${sem_metrics['cost']:.4f}, "
          f"{sem_metrics['calls']} calls, {t_sem:.1f}s (shared by Q5/Q6/Q7)")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
