#!/usr/bin/env python3
"""
Build the ONE input file every system reads: sf_2000 with exact-duplicate review
rows dropped.

SemBench's sf_2000 sample contains 135 rows that are literally identical on
(id, reviewId, criticName, reviewText, scoreSentiment) -- upstream, not our
loader. ant_man is the worst case: 256 rows = 128 distinct reviews x 2 copies,
which inflated the Q7 self-join by exactly 4x.

Dropping them is safe for scoring (the evaluator uses set semantics on both the
system and GT side) and it removes a real caliber mismatch: previously Octopus
ran on the deduped Postgres table while the baselines' numbers were on 2000 rows.

    python vision_evaluation/movie/prepare_data.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import protocol as P

KEY = ["id", "reviewId", "criticName", "reviewText", "scoreSentiment"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing prepared directory")
    a = ap.parse_args()

    import pandas as pd

    if P.DATA_DIR.exists() and not a.force:
        print(f"{P.DATA_DIR} already exists (use --force to rebuild)")
        return
    P.DATA_DIR.mkdir(parents=True, exist_ok=True)

    reviews = pd.read_csv(P.RAW_DATA_DIR / "Reviews.csv")
    n_before = len(reviews)
    key = [c for c in KEY if c in reviews.columns]
    if key != KEY:
        print(f"WARNING: expected columns {KEY}, deduping on {key}")
    deduped = reviews.drop_duplicates(subset=key, keep="first")

    per_movie = (reviews.groupby("id").size() - deduped.groupby("id").size()).fillna(0)
    worst = per_movie[per_movie > 0].sort_values(ascending=False)

    deduped.to_csv(P.DATA_DIR / "Reviews.csv", index=False)
    shutil.copy(P.RAW_DATA_DIR / "Movies.csv", P.DATA_DIR / "Movies.csv")

    print(f"Reviews: {n_before} -> {len(deduped)}  ({n_before - len(deduped)} exact duplicates dropped)")
    if len(worst):
        print("dropped per movie (top 5):")
        for mid, n in worst.head(5).items():
            print(f"  {mid}: -{int(n)}")
    for mid in (P.TAKEN3, P.ANTMAN):
        print(f"  {mid}: {int((deduped['id'] == mid).sum())} rows kept")
    print(f"\nsaved -> {P.DATA_DIR}")


if __name__ == "__main__":
    main()
