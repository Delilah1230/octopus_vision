#!/usr/bin/env python3
"""
Compute a [0,1] confidence score from subq_answers.csv and fit alpha to GT.

Scheme (see design notes):
  content_score = mean over content sub-qs of {T:1, F:0, cannot_determine:0.5}
  meta_score    = mean over meta   sub-qs of {T:1, F:0, cannot_determine:0.5}
  confidence(a) = 0.5 + (a + (1-a)*meta_score) * (content_score - 0.5)

content = direction toward predicate TRUE; meta = reliability gain.
alpha in [0,1]: 1 => ignore meta (confidence == content_score);
0 => meta fully gates how far confidence may leave 0.5.

Fit: grid-search alpha to minimize mean |confidence - T_rate| (MAE) and,
separately, to maximize Pearson correlation with T_rate.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ANSWERS = REPO_ROOT / "results" / "subq_probe" / "subq_answers.csv"

CONTENT_MAP = {"T": 1.0, "F": 0.0, "cannot_determine": 0.5, "MISSING": 0.5}
META_MAP = {"T": 1.0, "F": 0.0, "cannot_determine": 0.5, "MISSING": 0.5}


def load_units(path: Path) -> list[dict]:
    """Group rows into (asin, predicate) units with content/meta scores + T_rate."""
    groups: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"content": [], "meta": [], "gt": None, "slug": None}
    )
    with path.open() as f:
        for r in csv.DictReader(f):
            key = (r["parent_asin"], r["predicate_slug"])
            g = groups[key]
            g["slug"] = r["predicate_slug"]
            g["gt"] = float(r["gt_T_rate"]) if r["gt_T_rate"] not in ("", None) else None
            if r["sub_type"] == "content":
                g["content"].append(CONTENT_MAP.get(r["answer"], 0.5))
            elif r["sub_type"] == "meta":
                g["meta"].append(META_MAP.get(r["answer"], 0.5))

    units: list[dict] = []
    for (asin, slug), g in groups.items():
        if g["gt"] is None or not g["content"] or not g["meta"]:
            continue
        cs = sum(g["content"]) / len(g["content"])
        ms = sum(g["meta"]) / len(g["meta"])
        units.append({"asin": asin, "slug": slug, "content": cs,
                      "meta": ms, "gt": g["gt"]})
    return units


def confidence(cs: float, ms: float, alpha: float) -> float:
    gain = alpha + (1.0 - alpha) * ms
    return 0.5 + gain * (cs - 0.5)


def mae(units: list[dict], alpha: float) -> float:
    return sum(abs(confidence(u["content"], u["meta"], alpha) - u["gt"])
               for u in units) / len(units)


def pearson(units: list[dict], alpha: float) -> float:
    xs = [confidence(u["content"], u["meta"], alpha) for u in units]
    ys = [u["gt"] for u in units]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (vx * vy) if vx > 0 and vy > 0 else 0.0


def grid(units: list[dict], steps: int = 101) -> tuple[dict, dict]:
    alphas = [i / (steps - 1) for i in range(steps)]
    best_mae = min(alphas, key=lambda a: mae(units, a))
    best_cor = max(alphas, key=lambda a: pearson(units, a))
    return (
        {"alpha": best_mae, "mae": mae(units, best_mae), "cor": pearson(units, best_mae)},
        {"alpha": best_cor, "mae": mae(units, best_cor), "cor": pearson(units, best_cor)},
    )


def report(label: str, units: list[dict]) -> None:
    if not units:
        print(f"\n== {label}: no units ==")
        return
    bm, bc = grid(units)
    print(f"\n== {label}  (n={len(units)}) ==")
    print(f"  baseline alpha=1 (content only): MAE={mae(units,1.0):.4f}  cor={pearson(units,1.0):.3f}")
    print(f"  baseline alpha=0 (meta-gated)  : MAE={mae(units,0.0):.4f}  cor={pearson(units,0.0):.3f}")
    print(f"  best-by-MAE  : alpha={bm['alpha']:.2f}  MAE={bm['mae']:.4f}  cor={bm['cor']:.3f}")
    print(f"  best-by-cor  : alpha={bc['alpha']:.2f}  MAE={bc['mae']:.4f}  cor={bc['cor']:.3f}")
    # small alpha sweep table
    print("  alpha sweep:")
    for a in (0.0, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
        print(f"    alpha={a:.1f}  MAE={mae(units,a):.4f}  cor={pearson(units,a):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    args = ap.parse_args()

    units = load_units(args.answers)
    report("ALL predicates", units)
    for slug in sorted({u["slug"] for u in units}):
        report(f"predicate={slug}", [u for u in units if u["slug"] == slug])


if __name__ == "__main__":
    main()
