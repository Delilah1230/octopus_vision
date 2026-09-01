#!/usr/bin/env python3
"""
ABLATION — the same system with the tag store cold before EVERY query.

The full protocol resets the tag store once per repetition; here it is reset
before each of the ten queries, so nothing is ever amortised across queries.
Everything else is unchanged: same model, same single-stage cascade, same
related questions, same array/batch-6 encoding, same relational collapse.

So this isolates ONE mechanism -- cross-query reuse -- and nothing else.

CAN THE JOIN EVEN RUN WITHOUT REUSE?  Yes, and the reason is worth stating: the
join was never a reuse trick. Q5/Q6/Q7 run the sentiment predicate as an
ordinary filter and then pair the rows with SQL. Reuse only decides whether that
filter has already been paid for. With reuse, one semantic pass serves all three;
without it, each of the three pays its own pass. The collapse survives; only the
amortisation goes away.

WHAT THIS ABLATION EXPOSES, beyond cost:

  * Q8 needs a SECOND predicate. With a warm store, "is a negative review" binds
    by NEGATION to the positive tag and costs nothing. Cold, there is nothing to
    negate against, so it is a new predicate and must actually be evaluated --
    which is why is_a_negative_review had to be added to the registry before
    this ablation could run at all.
  * Q9/Q10 stop being free. They answer entirely from Q1's tags in the full
    protocol; cold, each pays its own pass (Q10 over the whole table).

Q5/Q6/Q7 are run as THREE separate cold invocations, one per query, so each
carries a full semantic pass rather than a third of one.

    python vision_evaluation/movie/run_ablation_notag.py
    python vision_evaluation/movie/run_ablation_notag.py --compare
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol as P

PY = sys.executable
SUFFIX = "-notag"
OUT = P.RESULTS / f"octopus{SUFFIX}"
QS = [f"Q{i}" for i in range(1, 11)]

# (query id, command). Each entry is preceded by a full tag-store reset, so the
# join runner is invoked three times -- once per pair query -- rather than once
# for all three.
STEPS = [
    ("Q1",  [PY, str(HERE / "run_octopus.py"), "--q", "1"]),
    ("Q2",  [PY, str(HERE / "run_octopus.py"), "--q", "2"]),
    ("Q3",  [PY, str(HERE / "run_octopus.py"), "--q", "3"]),
    ("Q4",  [PY, str(HERE / "run_octopus.py"), "--q", "4"]),
    # --share-over 1: each of these invocations answers ONE query, so the whole
    # semantic pass belongs to it. Leaving the default 3 would divide a cost that
    # is not being shared and under-report the ablation by 3x on these three.
    ("Q5",  [PY, str(HERE / "run_octopus_join.py"), "--order", "id", "--share-over", "1"]),
    ("Q6",  [PY, str(HERE / "run_octopus_join.py"), "--order", "id", "--share-over", "1"]),
    ("Q7",  [PY, str(HERE / "run_octopus_join.py"), "--order", "id", "--share-over", "1"]),
    ("Q8",  [PY, str(HERE / "run_octopus.py"), "--q", "8"]),
    ("Q9",  [PY, str(HERE / "run_octopus_score.py"), "--q", "9"]),
    ("Q10", [PY, str(HERE / "run_octopus_score.py"), "--q", "10"]),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true",
                    help="just print the comparison against the full protocol")
    a = ap.parse_args()

    if not a.compare:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "octopus_metrics.json").unlink(missing_ok=True)
        for qid, cmd in STEPS:
            P.reset_tag_store()                      # cold before EVERY query
            t0 = time.time()
            # "--out-suffix=-notag", not two args: the value starts with "-", so
            # argparse would read it as another option.
            r = subprocess.run(cmd + [f"--out-suffix={SUFFIX}"],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(f"{qid}: FAILED\n{r.stdout[-700:]}\n{r.stderr[-700:]}")
                raise SystemExit(f"ablation aborted at {qid}")
            print(f"  {qid:<4} {time.time()-t0:>6.1f}s")
        r = subprocess.run([PY, str(HERE / "evaluate.py"), "--system", "octopus",
                            f"--out-suffix={SUFFIX}"], capture_output=True, text=True)
        print(r.stdout[-900:])

    full_m = json.loads((P.RESULTS / "octopus/octopus_metrics.json").read_text())
    full_q = json.loads((P.RESULTS / "octopus/quality_metrics.json").read_text())
    ab_m = json.loads((OUT / "octopus_metrics.json").read_text())
    ab_q = json.loads((OUT / "quality_metrics.json").read_text())

    print(f"\n{'Q':<5}{'full $':>10}{'no-tag $':>11}{'x':>7}"
          f"{'full qual':>11}{'no-tag qual':>13}")
    print("-" * 57)
    tf = ta = 0.0
    for q in QS:
        if q not in ab_m or q not in full_m:
            continue
        f, n = full_m[q]["money_cost"], ab_m[q]["money_cost"]
        tf += f; ta += n
        ratio = "—" if f <= 0 else f"{n/f:.1f}"
        print(f"{q:<5}{f:>10.4f}{n:>11.4f}{ratio:>7}"
              f"{full_q[q]['quality']:>11.3f}{ab_q[q]['quality']:>13.3f}")
    print("-" * 57)
    print(f"{'TOT':<5}{tf:>10.4f}{ta:>11.4f}{ta/tf:>7.1f}x")
    print(f"\nreuse removes {(1 - tf/ta)*100:.0f}% of the workload's cost "
          f"({ta:.3f} -> {tf:.3f})")


if __name__ == "__main__":
    main()
