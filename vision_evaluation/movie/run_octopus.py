#!/usr/bin/env python3
"""
Octopus on SemBench movie Q2/Q3/Q4/Q8 — SINGLE-STAGE cascade.

All four queries carry the SAME semantic op ("is a positive review" over
taken_3), so after the first one tags the rows the rest reuse the tag and make
zero LLM calls. That reuse -- not the cascade -- is what these numbers are for.
The cascade is collapsed to one rung (gemini-2.5-flash, text) on purpose; see
protocol.py::configure_octopus for what that changes and why.

    python vision_evaluation/movie/run_octopus.py --q 2
    python vision_evaluation/movie/run_octopus.py --q 2 --clear-tags   # cold cache
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

# NL text comes from queries.json (nl_octopus), not a dict here -- see
# protocol.load_queries(). Q5/Q6/Q7/Q9/Q10 have nl_octopus=null because their
# plan is hand-written in the join/score runners.
_Q = P.load_queries()
NL = {qid: q["nl_octopus"] for qid, q in _Q.items() if q["nl_octopus"]}
TAG_COL = "tag_is_a_positive_review"


def db(sql, params=(), commit=False):
    with psycopg2.connect(**P.DB) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        if commit:
            conn.commit()
            return cur.rowcount
        return cur.fetchall()


def clear_tags(movie_id: str) -> None:
    db(f"UPDATE movie_reviews SET {TAG_COL}=NULL WHERE id=%s", (movie_id,), commit=True)
    db("DELETE FROM tag_meta WHERE parent_asin IN "
       "(SELECT parent_asin FROM movie_reviews WHERE id=%s)", (movie_id,), commit=True)
    print(f"[octopus] cleared tags for {movie_id}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, required=True, choices=[1, 2, 3, 4, 8])
    ap.add_argument("--clear-tags", action="store_true",
                    help="cold-start this query (drop the sentiment tag for taken_3 "
                         "first). Without it the run reuses whatever the tag store "
                         "already holds -- which is the point of the workload curve, "
                         "but makes a single-query number meaningless.")
    ap.add_argument("--reset-tags", action="store_true",
                    help="wipe the ENTIRE movie tag store first. Use this to start a "
                         "workload from a genuinely cold cache.")
    ap.add_argument("--batch", type=int, default=P.BATCH,
                    help="CAP on reviews per LLM prompt (protocol default; see "
                         "protocol.BATCH). 1 = one row per call, matching LOTUS/PZ.")
    ap.add_argument("--chunk-size", type=int, default=P.CHUNK_SIZE)
    ap.add_argument("--workers", type=int, default=P.WORKERS)
    ap.add_argument("--out-suffix", default="")
    a = ap.parse_args()

    out = P.out_dir("octopus", a.out_suffix)
    P.verify_row_parity()
    if a.reset_tags:
        P.reset_tag_store()
    elif a.clear_tags:
        clear_tags(P.TAKEN3)
    P.warn_on_stale_tags()

    P.reset_calls_log()
    t0 = time.time()
    pq = qp.parse_query(NL[a.q], table="movie_reviews", verify=False)
    sm.cleanup()
    matches = octopimizer.execute(pq, batch=a.batch, chunk_size=a.chunk_size,
                                  workers=a.workers)
    elapsed = time.time() - t0                       # parse + state build + cascade

    # Read the three-valued outcome from the tag column rather than inferring
    # negatives by subtraction. Under the §1.3 scorer an exact tie settles as U
    # and is stored as a NULL tag, so `total - positive` would silently count
    # every undetermined review as NEGATIVE. Q8 asks for the two counts, and
    # inventing negatives is exactly the kind of error the three-valued logic
    # exists to prevent.
    # Derive the column from what the parser ACTUALLY bound, never a hardcoded
    # name. The parser is an LLM: the same NL produced "is clearly positive" on one
    # run and "is a clearly positive review" on the next, each canonicalising to a
    # different column. A hardcoded name then reads an empty column and reports
    # 1865 undetermined for a run that in fact settled 1479 rows.
    tag_col = (qp.tag_write_column(pq.semantic[0]) if pq.semantic else TAG_COL)
    scope, params = ("", ()) if a.q == 1 else ("WHERE id=%s", (P.TAKEN3,))
    print(f"[Q{a.q}] counting from column {tag_col}")
    n_pos, n_neg, n_undet = db(
        f"SELECT COUNT(*) FILTER (WHERE {tag_col} IS TRUE), "
        f"       COUNT(*) FILTER (WHERE {tag_col} IS FALSE), "
        f"       COUNT(*) FILTER (WHERE {tag_col} IS NULL) "
        f"FROM movie_reviews {scope}", params)[0]
    n_total = n_pos + n_neg + n_undet
    if n_pos != len(matches):
        print(f"[warn] tag column says {n_pos} positive but the planner returned "
              f"{len(matches)} matches")
    print(f"\n[Q{a.q}] positive={n_pos}  negative={n_neg}  undetermined={n_undet}  "
          f"total={n_total}  elapsed={elapsed:.1f}s")
    if n_undet:
        print(f"[Q{a.q}] {n_undet} review(s) settled as U (exact tie among the "
              f"related questions) — stored as NULL, counted as neither")

    res = out / f"Q{a.q}.csv"
    with res.open("w", newline="") as f:
        w = csv.writer(f)
        if a.q in (1, 2):
            rid = dict(db("SELECT parent_asin, review_id FROM movie_reviews "
                          "WHERE parent_asin = ANY(%s)", (list(matches),))) if matches else {}
            w.writerow(["reviewId"])
            for key in list(matches)[:5]:
                w.writerow([rid.get(key, key)])
        elif a.q == 3:
            w.writerow(["positive_review_cnt"])
            w.writerow([n_pos])
        elif a.q == 4:
            w.writerow(["positivity_ratio"])
            w.writerow([round(n_pos / n_total, 6) if n_total else 0.0])
        elif a.q == 8:
            w.writerow(["scoreSentiment", "count"])
            w.writerow(["POSITIVE", n_pos])
            w.writerow(["NEGATIVE", n_neg])
    P.record_metrics(a.q, elapsed, {"n_matched": n_pos, "n_undetermined": n_undet},
                     a.out_suffix)
    print(f"saved -> {res}")


if __name__ == "__main__":
    main()
