#!/usr/bin/env python3
"""
Run the whole Q1-Q10 workload N times and report mean + spread, matching how
SemBench reports its own systems.

SemBench publishes FIVE independent runs per system (across_system_2.5flash_1..5)
and the paper reports the average. A single run of ours is therefore not
comparable to their table -- and their own data shows why it matters: LOTUS's Q5
f1 alternates between 0.533 and 0.667 across runs while its Q3 relative error is
identical in all five. Reporting one draw of a metric that swings by 0.13 is not
a measurement.

EACH REPETITION STARTS COLD. This is not optional for us the way it would be for
a stateless system: our tag store persists, so repetition 2 would reuse
repetition 1's answers and cost nothing. That would measure caching across
repetitions, which is not what a repetition is for. reset_tag_store() runs first
every time, so each repetition is an independent cold workload.

Query order within a repetition is fixed (Q1..Q10) because reuse makes the order
matter -- Q9/Q10 are free only because Q1 already graded those rows.

    python vision_evaluation/movie/run_repeats.py --n 5
    python vision_evaluation/movie/run_repeats.py --report
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol as P

OUT = P.RESULTS / "repeats"
PY = sys.executable

STEPS = [
    ("Q1",  [PY, str(HERE / "run_octopus.py"), "--q", "1"]),
    ("Q2",  [PY, str(HERE / "run_octopus.py"), "--q", "2"]),
    ("Q3",  [PY, str(HERE / "run_octopus.py"), "--q", "3"]),
    ("Q4",  [PY, str(HERE / "run_octopus.py"), "--q", "4"]),
    ("Q8",  [PY, str(HERE / "run_octopus.py"), "--q", "8"]),
    ("Q5-7", [PY, str(HERE / "run_octopus_join.py"), "--order", "id"]),
    ("Q9",  [PY, str(HERE / "run_octopus_score.py"), "--q", "9"]),
    ("Q10", [PY, str(HERE / "run_octopus_score.py"), "--q", "10"]),
]
QS = [f"Q{i}" for i in range(1, 11)]


def one_repetition(rep: int) -> dict:
    P.reset_tag_store()
    (P.RESULTS / "octopus" / "octopus_metrics.json").unlink(missing_ok=True)
    for label, cmd in STEPS:
        t0 = time.time()
        # One retry per step. A repetition is ~5 minutes of paid work; throwing it
        # away because a single provider call returned 503 wastes the run and
        # biases which repetitions survive. Steps are idempotent -- a re-run of a
        # query whose tags already landed just reuses them.
        for attempt in (1, 2):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                break
            if attempt == 1:
                print(f"  rep{rep} {label}: failed, retrying once")
                time.sleep(5)
        if r.returncode != 0:
            print(f"  rep{rep} {label}: FAILED twice\n{r.stdout[-600:]}\n{r.stderr[-600:]}")
            raise SystemExit(f"repetition {rep} aborted at {label}")
        print(f"  rep{rep} {label:<6} {time.time()-t0:>6.1f}s")
    subprocess.run([PY, str(HERE / "evaluate.py"), "--system", "octopus"],
                   capture_output=True, text=True)
    qual = json.loads((P.RESULTS / "octopus" / "quality_metrics.json").read_text())
    met = json.loads((P.RESULTS / "octopus" / "octopus_metrics.json").read_text())
    return {q: {"quality": qual[q]["quality"],
                "money_cost": met[q]["money_cost"],          # cascade + parser
                "cascade_cost": met[q].get("cascade_cost", met[q]["money_cost"]),
                "parser_cost": met[q].get("parser_cost", 0.0),
                "elapsed_s": met[q]["elapsed_s"]} for q in QS if q in qual and q in met}


def report() -> None:
    f = OUT / "repeats.json"
    if not f.exists():
        sys.exit(f"no repetitions yet at {f}")
    reps = json.loads(f.read_text())["repetitions"]
    n = len(reps)
    print(f"Octopus, {n} independent cold repetitions\n")
    print(f"{'Q':<5}{'quality mean':>14}{'sd':>7}{'min–max':>14}"
          f"{'$ mean':>10}{'s mean':>9}")
    print("-" * 60)
    tq = tc = ts = 0.0
    for q in QS:
        vals = [r[q]["quality"] for r in reps if q in r]
        cost = [r[q]["money_cost"] for r in reps if q in r]
        sec = [r[q]["elapsed_s"] for r in reps if q in r]
        if not vals:
            continue
        sd = st.pstdev(vals) if len(vals) > 1 else 0.0
        tq += st.mean(vals); tc += st.mean(cost); ts += st.mean(sec)
        print(f"{q:<5}{st.mean(vals):>14.3f}{sd:>7.3f}"
              f"{f'{min(vals):.2f}–{max(vals):.2f}':>14}"
              f"{st.mean(cost):>10.4f}{st.mean(sec):>9.1f}")
    print("-" * 60)
    print(f"{'MEAN':<5}{tq/len(QS):>14.3f}{'':>7}{'':>14}{tc:>10.4f}{ts:>9.1f}"
          f"   ($ and s are TOTALS)")
    unstable = [q for q in QS
                if len([r[q]["quality"] for r in reps if q in r]) > 1
                and st.pstdev([r[q]["quality"] for r in reps if q in r]) > 0.05]
    if unstable:
        print(f"\n  sd > 0.05 on {unstable} — report these as a range, not a point. "
              f"SemBench's own five runs show the same instability on the limit-k "
              f"pair queries.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        return report()

    OUT.mkdir(parents=True, exist_ok=True)
    reps = []
    for i in range(1, a.n + 1):
        print(f"\n═══ repetition {i}/{a.n} (cold) ═══")
        reps.append(one_repetition(i))
        (OUT / "repeats.json").write_text(json.dumps(
            {"n": len(reps), "repetitions": reps}, indent=2))
    print()
    report()


if __name__ == "__main__":
    main()
