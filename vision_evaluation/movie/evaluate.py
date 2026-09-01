#!/usr/bin/env python3
"""
Scorer for SemBench movie Q1-Q8. Metric formulas copied 1:1 from
src/scenario/movie/evaluation/evaluate.py; ground truth is the gold SQL run over
the CSV via DuckDB (the CSV keeps the hidden scoreSentiment column, the Postgres
load does not).

IMPORTANT: GT is computed from the SAME file the systems read (the deduped
sf_2000 by default). Scoring a deduped run against 2000-row GT silently punishes
the count/ratio queries -- that mismatch is exactly what this folder exists to
remove. Pass --raw only together with runners that were also given --raw.

    python vision_evaluation/movie/evaluate.py --system octopus
    python vision_evaluation/movie/evaluate.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import protocol as P

QUERIES = P.load_queries()          # queries.json is the source of truth
GOLD_SQL = {qid: q["gold_sql"] for qid, q in QUERIES.items()}
LIMIT = {qid: q["limit"] for qid, q in QUERIES.items()}


# ── metric fns, copied from SemBench MovieEvaluator ──────────────────────────

def retrieval_limit(sys_df, gt, limit):
    if len(sys_df) == 0:
        return dict(precision=1.0 if len(gt) == 0 else 0.0, recall=0.0, f1=0.0)
    if len(gt) == 0:
        return dict(precision=0.0, recall=0.0, f1=0.0)
    sys_df = sys_df.head(limit)
    sys_ids = set(sys_df[sys_df.columns[0]].dropna())
    gt_ids = set(gt[gt.columns[0]].dropna())
    inter = sys_ids & gt_ids
    p = len(inter) / len(sys_ids) if sys_ids else 0.0
    r = min(len(inter), limit) / min(limit, len(gt_ids)) if gt_ids else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return dict(precision=p, recall=r, f1=f1)


def aggregation(sys_df, gt):
    if len(sys_df) == 0 or len(gt) == 0:
        return dict(relative_error=1.0)
    try:
        sv, gv = float(sys_df.iloc[0, 0]), float(gt.iloc[0, 0])
    except Exception:                                 # noqa: BLE001
        return dict(relative_error=1.0)
    if gv == 0:
        return dict(relative_error=0.0 if sv == 0 else 1.0)
    return dict(relative_error=abs(sv - gv) / abs(gv))


def pairs(sys_df, gt, limit=None):
    import pandas as pd
    if len(sys_df) == 0:
        return dict(precision=1.0 if len(gt) == 0 else 0.0, recall=0.0, f1=0.0)
    if len(gt) == 0 or len(sys_df.columns) < 3 or len(gt.columns) < 3:
        return dict(precision=0.0, recall=0.0, f1=0.0)
    if limit:
        sys_df = sys_df.head(limit)

    def norm(df):
        s = set()
        for _, row in df.iterrows():
            mid, a, b = row.iloc[0], row.iloc[1], row.iloc[2]
            if pd.notna(mid) and pd.notna(a) and pd.notna(b):
                s.add((mid, tuple(sorted([a, b]))))
        return s

    sp, gp = norm(sys_df), norm(gt)
    correct = sp & gp
    p = len(correct) / len(sp) if sp else 0.0
    if limit:
        r = min(len(correct), limit) / min(limit, len(gp)) if gp else 0.0
    else:
        r = len(correct) / len(gp) if gp else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return dict(precision=p, recall=r, f1=f1)


def sentiment_counts(sys_df, gt):
    import pandas as pd
    if len(sys_df) == 0 or len(gt) == 0 or len(sys_df.columns) < 2 or len(gt.columns) < 2:
        return dict(relative_error=1.0)

    def to_map(df):
        d = {}
        for _, row in df.iterrows():
            k, v = row.iloc[0], row.iloc[1]
            if pd.notna(k) and pd.notna(v):
                try:
                    d[str(k).strip().upper()] = float(v)
                except Exception:                     # noqa: BLE001, S110
                    pass
        return d

    sm, gm = to_map(sys_df), to_map(gt)
    tot, n = 0.0, 0
    for k in set(sm) | set(gm):
        s, g = sm.get(k, 0.0), gm.get(k, 0.0)
        if g != 0:
            tot += abs(s - g) / abs(g)
            n += 1
        elif s != 0:
            tot += 1.0
            n += 1
    return dict(relative_error=(tot / n) if n else 0.0)


def ranking(sys_df, gt):
    """Spearman + Kendall tau over ids present in BOTH sides (upstream
    _generic_ranking_evaluation). First column is the id, second the score."""
    import pandas as pd
    from scipy.stats import kendalltau, spearmanr

    empty = dict(spearman=0.0, kendall_tau=0.0, n_common=0)
    if len(sys_df) == 0 or len(gt) == 0:
        return empty
    if len(sys_df.columns) < 2 or len(gt.columns) < 2:
        return empty

    def to_map(df):
        d = {}
        for _, row in df.iterrows():
            k, v = row.iloc[0], row.iloc[1]
            if pd.notna(k) and pd.notna(v):
                try:
                    d[k] = float(v)
                except (ValueError, TypeError):
                    continue
        return d

    sm, gm = to_map(sys_df), to_map(gt)
    common = sorted(set(sm) & set(gm))
    if len(common) < 2:
        return empty
    sv = [sm[k] for k in common]
    gv = [gm[k] for k in common]
    sp = spearmanr(sv, gv).correlation
    kt = kendalltau(sv, gv).correlation
    # A constant prediction (every row scored the same) makes the correlation
    # undefined -> nan. Report it as 0.0: no ordering information was produced.
    sp = 0.0 if sp != sp else float(sp)
    kt = 0.0 if kt != kt else float(kt)
    return dict(spearman=sp, kendall_tau=kt, n_common=len(common))


# Metric per query KIND, and the limit comes from queries.json -- so adding or
# retuning a query never needs a matching edit here.
_BY_KIND = {
    "retrieval":        lambda s, g, lim: retrieval_limit(s, g, lim),
    "aggregation":      lambda s, g, lim: aggregation(s, g),
    "pairs":            lambda s, g, lim: pairs(s, g, lim),
    "sentiment_counts": lambda s, g, lim: sentiment_counts(s, g),
    "rank":             lambda s, g, lim: ranking(s, g),
}
SCORER = {qid: (lambda s, g, _q=q: _BY_KIND[_q["kind"]](s, g, _q["limit"]))
          for qid, q in QUERIES.items()}

SYSTEMS = ["octopus", "lotus", "palimpzest"]


def score_system(con, results_dir: Path, qids: list[int]) -> dict:
    import pandas as pd
    out = {}
    for qid in qids:
        f = results_dir / f"Q{qid}.csv"
        if not f.exists():
            continue
        sys_df = pd.read_csv(f)
        gt = con.execute(GOLD_SQL[qid]).fetchdf()
        m = SCORER[qid](sys_df, gt)
        # SemBench's [0,1] quality score. F1 is already in range; relative error is
        # mapped with 1/(1+e) -- NOT 1-e (paper, "Evaluation Settings"). Table 4
        # reports this normalized value, so always compare on `quality`.
        # Rank queries report Spearman, which the paper leaves un-normalized
        # (their measured values were all > 0); a negative correlation is floored
        # at 0 rather than rewarded.
        if "f1" in m:
            m["quality"] = m["f1"]
        elif "relative_error" in m:
            m["quality"] = 1.0 / (1.0 + m["relative_error"])
        else:
            m["quality"] = max(0.0, m["spearman"])
        m["gt_rows"], m["sys_rows"] = len(gt), len(sys_df)
        out[f"Q{qid}"] = m
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=SYSTEMS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--queries", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--dedup", action="store_true",
                    help="score against the DEDUPED CSV; only correct if the runners "
                         "were also given --dedup. Default is the benchmark as published.")
    ap.add_argument("--out-suffix", default="")
    a = ap.parse_args()
    if not a.system and not a.all:
        ap.error("pass --system NAME or --all")

    import duckdb
    ddir = P.data_dir(dedup=a.dedup)
    con = duckdb.connect()
    con.execute("CREATE VIEW Reviews AS SELECT * FROM "
                f"read_csv_auto('{ddir/'Reviews.csv'}', header=true)")
    con.execute("CREATE VIEW Movies  AS SELECT * FROM "
                f"read_csv_auto('{ddir/'Movies.csv'}', header=true)")
    print(f"[eval] ground truth from {ddir}")

    targets = SYSTEMS if a.all else [a.system]
    summary = {}
    for sysname in targets:
        rdir = P.RESULTS / f"{sysname}{a.out_suffix}"
        if not rdir.exists():
            print(f"\n{sysname}: (no results at {rdir})")
            continue
        res = score_system(con, rdir, a.queries)
        summary[sysname] = res
        print(f"\n=== {sysname} ===")
        print(f"{'Q':>3}  {'metric':<15} {'value':>8}   {'quality':>7}   rows(gt/sys)")
        for q, m in res.items():
            key = ("f1" if "f1" in m else
                   "relative_error" if "relative_error" in m else "spearman")
            print(f"{q:>3}  {key:<15} {m[key]:>8.4f}   {m['quality']:>7.4f}   "
                  f"{m['gt_rows']}/{m['sys_rows']}")
        if res:
            mean_q = sum(m["quality"] for m in res.values()) / len(res)
            print(f"{'':>3}  {'MEAN quality':<15} {'':>8}   {mean_q:>7.4f}")
        (rdir / "quality_metrics.json").write_text(json.dumps(res, indent=2))
        print(f"saved -> {rdir/'quality_metrics.json'}")

    if len(summary) > 1:
        (P.RESULTS / "quality_all.json").write_text(json.dumps(summary, indent=2))
        print(f"\nsaved -> {P.RESULTS/'quality_all.json'}")


if __name__ == "__main__":
    main()
