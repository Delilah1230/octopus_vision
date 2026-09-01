#!/usr/bin/env python3
"""
Q9/Q10 (1-to-5 SCORE queries) — the honest two-step, like the join runner.

WHAT THE PARSER CAN AND CANNOT DO HERE (measured, not assumed):

  Q9  -> 1 structural condition + 1 semantic predicate, but the predicate comes
         back as the BOOLEAN "allows for a sentiment score from 1 to 5 to be
         assigned". The 1-5 grading is dropped on the floor and execute() returns
         a flat list of row keys with nowhere to put a score. Silent degradation:
         nothing errors, the answer is just not the one that was asked for.
  Q10 -> ZERO semantic predicates and a bare `SELECT * FROM movie_reviews`. The
         semantics are lost entirely.

So the PLAN below is written by hand, exactly as the pairing step in
run_octopus_join.py is, and the paper must say so.

WHERE THE SCORE COMES FROM — and why this is not a workaround:

The related-question scheme already computes an ORDINAL quantity per row:

    r = fraction of the 6 related questions answered TRUE   in {0, 1/6, ..., 1}

The system stores the two things r is made of -- the committed direction (the
tag column) and c = max(r, 1-r) (tag_meta.confidence) -- so r is recoverable
exactly:

    r = c            if the tag is TRUE
    r = 1 - c        if the tag is FALSE
    r = 0.5          if the tag is NULL (U: an exact 3-3 tie)

and a 1-5 score is an affine read of it:  score = 1 + 4r.

That means Q9 and Q10 cost ZERO additional LLM calls once the sentiment tags
exist. The graded answer was already paid for by the boolean query; only the
boolean was ever exposed. This is a property of counting same-direction
questions, not a trick -- notes/2026-08-10.md §1.3 already argues the score is
ORDINAL rather than a probability, and an ordinal score is exactly what a
"rank these 1-5" query wants.

CAVEAT to report with the numbers: r has only 7 levels (0, 1/6 ... 1), so the
score is coarse, and Spearman against a continuous ground truth will be dragged
down by ties, not only by ranking errors.

    python vision_evaluation/movie/run_octopus_score.py --q 9
    python vision_evaluation/movie/run_octopus_score.py --q 10
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

# Q9 grades ant_man's reviews; Q10 grades every review in the table. Both are
# graded with the SAME predicate, so whichever runs first pays and the other
# reuses. Q1 uses this predicate too -- run Q1 before Q10 and Q10 is free.
PRED_NL = "is clearly positive"
# Resolved from the parse at runtime, never hardcoded -- the parser rewords
# predicates between runs and each wording gets its own column (see run_octopus.py).
TAG_COL = "tag_is_clearly_positive"      # fallback only


def db(sql, params=(), commit=False):
    with psycopg2.connect(**P.DB) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        if commit:
            conn.commit()
            return cur.rowcount
        return cur.fetchall()


def resolve_tag_col(movie_id: str | None) -> str:
    """Parse the grading query once and take the column the parser bound."""
    nl = (f'Reviews that are clearly positive for movie "{movie_id}". Return reviewId.'
          if movie_id else "Reviews that are clearly positive. Return reviewId.")
    pq = qp.parse_query(nl, table="movie_reviews", verify=False)
    col = qp.tag_write_column(pq.semantic[0]) if pq.semantic else TAG_COL
    print(f"[score] predicate {pq.semantic[0].nl!r} -> column {col}" if pq.semantic
          else f"[score] no semantic predicate parsed; falling back to {col}")
    return col, pq


def score_rows(movie_id: str | None, TAG_COL: str = TAG_COL):
    """(review_id, movie_id, score_1_to_5) for every row that has a settled tag.

    r is rebuilt from the two stored pieces; see the module docstring. Rows with
    no tag_meta row have no r and are skipped rather than guessed at."""
    where = "WHERE r.id = %s" if movie_id else ""
    params = (movie_id,) if movie_id else ()
    rows = db(f"""
        SELECT r.review_id, r.id, r.{TAG_COL}, m.confidence
          FROM movie_reviews r
          LEFT JOIN tag_meta m
            ON m.parent_asin = r.parent_asin
           AND m.predicate_canon = '{TAG_COL}'
          {where}
    """, params)
    out = []
    for review_id, mid, tag, conf in rows:
        if conf is None and tag is None:
            continue                       # never evaluated
        if tag is None:
            r = 0.5                        # U: exact tie
        else:
            c = float(conf) if conf is not None else 1.0
            r = c if tag else 1.0 - c
        out.append((review_id, mid, round(1.0 + 4.0 * r, 4)))
    return out


def ensure_tagged(movie_id, workers: int, batch: int, chunk: int,
                  TAG_COL: str = TAG_COL, pq=None) -> float:
    """Run the semantic step if the tags are not already there. Returns seconds."""
    scope = "WHERE id = %s" if movie_id else ""
    params = (movie_id,) if movie_id else ()
    total = db(f"SELECT COUNT(*) FROM movie_reviews {scope}", params)[0][0]
    tagged = db(f"SELECT COUNT(*) FROM movie_reviews r JOIN tag_meta m "
                f"ON m.parent_asin=r.parent_asin AND m.predicate_canon='{TAG_COL}' "
                f"{scope}", params)[0][0]
    print(f"[score] {tagged}/{total} rows already graded by '{PRED_NL}'")
    if tagged >= total:
        print("[score] reusing the tag store — 0 LLM calls for the semantic step")
        return 0.0

    print(f"[score] parsed: structural={[c.sql for c in pq.structural]} "
          f"semantic={[(p.nl, p.predicate_type) for p in pq.semantic]}")
    sm.cleanup()
    t0 = time.time()
    octopimizer.execute(pq, batch=batch, chunk_size=chunk, workers=workers)
    return time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, required=True, choices=[9, 10])
    ap.add_argument("--workers", type=int, default=P.WORKERS)
    ap.add_argument("--batch", type=int, default=P.BATCH)
    ap.add_argument("--chunk-size", type=int, default=P.CHUNK_SIZE)
    ap.add_argument("--out-suffix", default="")
    a = ap.parse_args()

    out = P.out_dir("octopus", a.out_suffix)
    P.verify_row_parity()

    movie = P.ANTMAN if a.q == 9 else None      # Q10 covers every movie
    P.reset_calls_log()
    _t_all = time.time()
    tag_col, pq = resolve_tag_col(movie)
    t_sem = ensure_tagged(movie, a.workers, a.batch, a.chunk_size, tag_col, pq)

    t0 = time.time()
    rows = score_rows(movie, tag_col)
    res = out / f"Q{a.q}.csv"
    if a.q == 9:
        with res.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["reviewId", "reviewScore"])
            for review_id, _mid, s in rows:
                w.writerow([review_id, s])
        n = len(rows)
    else:
        agg: dict[str, list[float]] = {}
        for _rid, mid, s in rows:
            agg.setdefault(mid, []).append(s)
        with res.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["movieId", "movieScore"])
            for mid, vals in sorted(agg.items()):
                w.writerow([mid, round(sum(vals) / len(vals), 4)])
        n = len(agg)
    t_plan = time.time() - t0

    levels = sorted({s for _r, _m, s in rows})
    print(f"\n[Q{a.q}] {n} row(s) scored  semantic={t_sem:.1f}s  "
          f"score-and-aggregate={t_plan*1000:.1f}ms "
          f"({'0 LLM calls' if t_sem == 0 else 'semantic step paid once'})")
    print(f"[Q{a.q}] distinct score levels: {levels}  "
          f"(r has only 7 levels — the score is coarse by construction)")
    P.record_metrics(a.q, time.time() - _t_all,
                     {"n_rows_out": n, "score_levels": len(levels)}, a.out_suffix)
    print(f"saved -> {res}")


if __name__ == "__main__":
    main()
