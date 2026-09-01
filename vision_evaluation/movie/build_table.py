#!/usr/bin/env python3
"""
Freeze the five-system comparison into a versioned artifact.

Everything downstream (the note, the paper, the figures) reads THIS file rather
than recomputing from scratch, so a number in the paper can always be traced to
one snapshot instead of to whatever the results directory happened to hold.

Ours = mean over our own 5 cold repetitions (run_repeats.py).
Theirs = mean over SemBench's own 5 published runs (sembench_declared.json).
Both on sf_2000 as published; no adjustment factors anywhere.

The standard deviation follows the paper's convention: the per-query sd is
averaged over queries to give ONE number per system per metric, not a sd per
cell. The paper uses it to say cost/quality are stable and run time is not,
which is a statement about a system, not about a query.

    python vision_evaluation/movie/build_table.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import protocol as P

DECLARED = Path(__file__).resolve().parent / "sembench_declared.json"
OUT = P.RESULTS / "comparison"
SYSTEMS = ["octopus", "lotus", "palimpzest", "thalamusdb", "bigquery"]
QS = [f"Q{i}" for i in range(1, 11)]


def quality_of(d: dict):
    """SemBench's [0,1] score: F1 as-is, relative error via 1/(1+e), Spearman
    floored at 0 (their measured values were all positive)."""
    if d.get("f1_score") is not None:
        return d["f1_score"]
    if d.get("relative_error") is not None:
        return 1.0 / (1.0 + d["relative_error"])
    for k in ("spearman_correlation", "spearman"):
        if d.get(k) is not None:
            return max(0.0, d[k])
    return None


# The parser and predicate matcher are LLM calls too, and they were invisible in
# every number before 2026-08-11 because cost was read from the per-ROW call log
# while the parser runs per query. That mattered: they were 23% of our total with
# the parser's thinking on, 9% with it minimal -- and they are a component NEITHER
# baseline has, since LOTUS and Palimpzest queries are hand-written operators with
# no NL->plan step.
#
# record_metrics now folds parser_cost into money_cost at run time, so there is
# nothing left to add here. Adding parser_cost.json on top would DOUBLE-COUNT it.
# The function stays only to detect the older shape.
def parser_already_included(reps) -> bool:
    """Q9 makes ZERO per-row calls -- it answers entirely from tags Q1 wrote. So
    a non-zero Q9 cost can only be the parser, which makes it a reliable probe for
    whether these numbers already carry parser accounting. Checking for the
    breakdown FIELDS instead would give a false negative on runs made before
    run_repeats stored them, even though their money_cost was already correct."""
    vals = [r["Q9"]["money_cost"] for r in reps if "Q9" in r]
    return bool(vals) and min(vals) > 0


def series(system: str, q: str, reps) -> tuple[list, list, list] | None:
    """Per-run (quality, cost, seconds) for one system+query."""
    if system == "octopus":
        v = [r[q] for r in reps if q in r]
        if not v:
            return None
        return ([x["quality"] for x in v], [x["money_cost"] for x in v],
                [x["elapsed_s"] for x in v])
    runs = json.loads(DECLARED.read_text())[system]["runs"]
    ok = [r for r in runs if r.get(q, {}).get("status") == "success"
          and r[q].get("money_cost") is not None and quality_of(r[q]) is not None]
    if not ok:
        return None
    return ([quality_of(r[q]) for r in ok], [r[q]["money_cost"] for r in ok],
            [r[q].get("execution_time", 0) for r in ok])


QTYPE = {"Q1": "filter", "Q2": "filter", "Q3": "count", "Q4": "ratio",
         "Q5": "join", "Q6": "join", "Q7": "join", "Q8": "counts",
         "Q9": "score", "Q10": "rank"}


def write_latex(table: dict) -> None:
    """Emit the body rows of tab:sembench, to be \\input by the paper.

    This exists because the table and the cumulative figures drifted apart once:
    the figures were regenerated after the parser's thinking was set to minimal
    and the table was not, so the same run appeared as $0.415/142s in one place
    and $0.350/79s in the other. Both now come from THIS file, which comes from
    comparison.json, which comes from one frozen snapshot.

    Only the body is generated. The header, the column spec and the caption stay
    hand-written in the .tex, because they carry prose that no script should own.
    """
    def cell(c, best):
        if c is None:
            return None
        f = lambda v, s, p: (f"\\textbf{{{v:.{p}f}}}" if abs(v - s) < 10**-(p+1)
                             else f"{v:.{p}f}")
        return (f"{f(c['quality'], best[0], 3)} & {f(c['money_cost'], best[1], 3)} "
                f"& {f(c['elapsed_s'], best[2], 0)}")

    lines = []
    for q in QS:
        cs = [table[s]["per_query"][q] for s in SYSTEMS]
        live = [c for c in cs if c]
        best = (max(c["quality"] for c in live), min(c["money_cost"] for c in live),
                min(c["elapsed_s"] for c in live))
        parts = [cell(c, best) or "\\multicolumn{3}{c}{---}" for c in cs]
        lines.append(f"{q:<4}& {QTYPE[q]:<7}& " +
                     "\n             & ".join(parts) + " \\\\")

    # A system that answered 8 of the 10 queries is not eligible to win ANY of the
    # three totals -- not just quality. Its cost and run time are sums over fewer
    # queries, and the two it skips (Q9/Q10) are not cheap ones. Bolding its $0.165
    # as "best" would credit it for work it did not do; the caption already says
    # this about quality, and the same reasoning applies to a sum.
    elig = [s for s in SYSTEMS if table[s]["n_queries"] == len(QS)]
    bq = max(table[s]["mean_quality"] for s in elig)
    bc = min(table[s]["total_cost"] for s in elig)
    bs = min(table[s]["total_seconds"] for s in elig)
    tot = []
    for s in SYSTEMS:
        d = table[s]
        star = "$^*$" if d["n_queries"] != len(QS) else ""
        f = lambda v, b, p: (f"\\textbf{{{v:.{p}f}}}" if abs(v - b) < 10**-(p+1)
                             else f"{v:.{p}f}")
        qcell = (f"{d['mean_quality']:.3f}{star}" if star
                 else f(d["mean_quality"], bq, 3))
        tot.append(f"{qcell} & {f(d['total_cost'], bc, 3)} & "
                   f"{f(d['total_seconds'], bs, 0)}")
    lines.append("\\hline\n\\multicolumn{2}{@{}l}{\\textbf{mean\\,/\\,total}}\n"
                 "             & " + "\n             & ".join(tot) + " \\\\")
    sd = [f"{table[s]['sd_quality']:.4f}".lstrip("0") + " & " +
          f"{table[s]['sd_cost']:.4f}".lstrip("0") + " & " +
          f"{table[s]['sd_seconds']:.1f}" for s in SYSTEMS]
    lines.append("\\multicolumn{2}{@{}l}{\\emph{sd}}\n             & " +
                 "\n             & ".join(sd) + " \\\\")

    (OUT / "table_sembench.tex").write_text(
        "% GENERATED by vision_evaluation/movie/build_table.py -- do not edit.\n"
        "% Source: comparison.json (same snapshot as fig5/fig6).\n"
        + "\n".join(lines) + "\n")


def main() -> None:
    reps = json.loads((P.RESULTS / "repeats/repeats.json").read_text())["repetitions"]
    native = parser_already_included(reps)
    if not native:
        raise SystemExit("repeats.json predates parser accounting (Q9 cost is 0) — "
                         "re-run run_repeats.py so cost and latency come from one "
                         "protocol")
    table: dict = {"_about": {
        "scenario": "movie", "data": "sf_2000 as published (2000 rows)",
        "model": "gemini-2.5-flash for all systems", "workers": 20,
        "octopus_runs": len(reps),
        "baselines": "SemBench's own published runs (across_system_2.5flash_1..5)",
        "octopus_cost_includes": "per-row cascade + parser/matcher calls, recorded "
                                 "in the same run. Baselines have no parser step.",
        "parser_thinking": "minimal (gemini-3.x uses thinking_level, not "
                           "thinking_budget; only 'minimal' zeroes thoughts)",
        "sd_convention": "per-query sd averaged over queries — one number per "
                         "system per metric, as the paper reports it",
        "caveats": [
            "ThalamusDB does not implement Q9/Q10; its mean covers 8 queries, and "
            "the two it skips are the ones every system scores worst on (0.37-0.44). "
            "Its mean is NOT comparable to a 10-query mean.",
            "Q5/Q6 are limit-10 over thousands of gold pairs: which ten come back "
            "decides the score. Ours are deterministic (ORDER BY) so sd=0, but that "
            "is reproducible, not meaningful — SemBench's own runs show PZ Q5 "
            "sd 0.12, BigQuery Q5 sd 0.09.",
            "Q5/Q6/Q7 share one semantic step; its cost and latency are split "
            "three ways. The pairing itself is 0 LLM calls.",
            "Q9/Q10 cost 0 only because Q1 already graded those rows — a different "
            "query order removes those zeros.",
        ]}}

    for s in SYSTEMS:
        per_q, sdq, sdc, sdr = {}, [], [], []
        tq = tc = ts = 0.0
        n = 0
        for q in QS:
            r = series(s, q, reps)
            if r is None:
                per_q[q] = None
                continue
            ql, co, se = r
            total = st.mean(co)          # already cascade + parser for octopus
            per_q[q] = {"quality": st.mean(ql), "money_cost": total,
                        "elapsed_s": st.mean(se), "n_runs": len(ql)}
            if s == "octopus":
                v = [rr[q] for rr in reps if q in rr]
                if any("parser_cost" in x for x in v):
                    per_q[q]["cascade_cost"] = st.mean(x.get("cascade_cost", 0.0) for x in v)
                    per_q[q]["parser_cost"] = st.mean(x.get("parser_cost", 0.0) for x in v)
            tq += st.mean(ql); tc += total; ts += st.mean(se); n += 1
            if len(ql) > 1:
                sdq.append(st.pstdev(ql)); sdc.append(st.pstdev(co)); sdr.append(st.pstdev(se))
        table[s] = {
            "per_query": per_q,
            "mean_quality": tq / n, "total_cost": tc, "total_seconds": ts,
            "n_queries": n,
            "sd_quality": st.mean(sdq) if sdq else 0.0,
            "sd_cost": st.mean(sdc) if sdc else 0.0,
            "sd_seconds": st.mean(sdr) if sdr else 0.0,
        }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comparison.json").write_text(json.dumps(table, indent=2))

    # markdown, for pasting into the note / paper
    md = ["| Q | " + " | ".join(s.capitalize() for s in SYSTEMS) + " |",
          "|---|" + "---|" * len(SYSTEMS)]
    for q in QS:
        row = [q]
        for s in SYSTEMS:
            c = table[s]["per_query"][q]
            row.append("—" if c is None
                       else f"{c['quality']:.3f} / {c['money_cost']:.3f} / {c['elapsed_s']:.0f}")
        md.append("| " + " | ".join(row) + " |")
    md.append("| **mean** | " + " | ".join(
        f"**{table[s]['mean_quality']:.3f} / \\${table[s]['total_cost']:.3f} / "
        f"{table[s]['total_seconds']:.0f}s**" for s in SYSTEMS) + " |")
    md.append("| *sd (avg per query)* | " + " | ".join(
        f"{table[s]['sd_quality']:.4f} / {table[s]['sd_cost']:.4f} / "
        f"{table[s]['sd_seconds']:.2f}s" for s in SYSTEMS) + " |")
    (OUT / "comparison.md").write_text("\n".join(md) + "\n")
    write_latex(table)

    print("\n".join(md))
    o = table["octopus"]
    print()
    for s in SYSTEMS[1:]:
        if table[s]["n_queries"] == 10:
            print(f"  vs {s:<12} cost {table[s]['total_cost']/o['total_cost']:>5.1f}x   "
                  f"latency {table[s]['total_seconds']/o['total_seconds']:>5.1f}x   "
                  f"quality {table[s]['mean_quality']-o['mean_quality']:+.3f}")
    print(f"\nsaved -> {OUT}/comparison.json and comparison.md")


if __name__ == "__main__":
    main()
