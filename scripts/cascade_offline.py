#!/usr/bin/env python3
"""
Offline AnswerSource + driver for the cascade engine (scripts/cascade.py).

Replays the recorded per-tier probe tables — no API calls, real recorded
cost/latency. This is the same engine that would run online; only the
AnswerSource differs.

Usage:
    python scripts/cascade_offline.py                       # vegan AND indulgent
    python scripts/cascade_offline.py --query "vegan OR indulgent_valentine"
    python scripts/cascade_offline.py --chunk-size 50 --batch 5 --workers 8
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cascade import (  # noqa: E402
    Answer, CascadeConfig, Leaf, Node, ROUTES, T, F, U,
    chunks_of, eval_tree, learn_kill_rates, parse_tree, run_cascade,
)
from subq_score import load_units  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# tier -> (results dir, label).
#
# ⚠️ TIER-2 MISMATCH (2026-07-14): cascade_live.py now runs tier 2 on
# `gemini-2.5-flash`, but the only recorded tier-2 table is
# `subq_probe_flash3` = 400 calls of `gemini-3-flash-preview`. Until a
# gemini-2.5-flash probe is run, OFFLINE AND LIVE USE DIFFERENT TIER-2 MODELS,
# so their results are NOT comparable (e.g. the live 20/173/7 vs offline 18/175/7
# comparison is void). Rebuild with:
#     python scripts/subquestion_probe.py --model gemini-2.5-flash \
#         --predicates vegan,indulgent_valentine --out-dir results/subq_probe_flash25
# then point tier 2 here at subq_probe_flash25.
# (tier, modality) -> (results dir, model label). The escalation route walks
# these steps; a step with no recorded table is trimmed from the OFFLINE route
# (see offline_routes) rather than faked. text_image is recorded at tier 1 only
# so far -> the (2,txt+img)/(3,txt+img) rungs are offline-unavailable until those
# probes are run; live (cascade_live.py) calls them for real.
STEP_DIRS = {
    (1, "text"):       ("subq_probe", "gemini-2.5-flash-lite"),
    (2, "text"):       ("subq_probe_flash3", "gemini-3-flash-preview"),  # STALE vs live
    (3, "text"):       ("subq_probe_pro", "gemini-2.5-pro"),
    (1, "text_image"): ("subq_probe_textimg", "gemini-2.5-flash-lite"),
}
# christmas_dinner only has a tier-1 text probe so far.
EXTRA_TIER1_DIRS = {"christmas_dinner": "subq_probe_christmas_v2"}

PREDICATE_TYPE = {
    "vegan": "objective",
    "indulgent_valentine": "subjective",
    "christmas_dinner": "subjective",
}
THRESHOLDS = {
    "objective":  {1: 0.70, 2: 0.60, 3: 0.50},
    "subjective": {1: 0.55, 2: 0.45, 3: 0.35},
}

CERT_WC, CERT_WM = 0.7, 0.3


def _confidence(content: float, meta: float) -> float:
    return CERT_WC * (2.0 * abs(content - 0.5)) + CERT_WM * meta


def _direction(content: float) -> str:
    return T if content > 0.5 else F


class OfflineSource:
    """Recorded table as an AnswerSource. Raises on a missing cell rather than
    silently guessing — a missing (product, predicate, tier) means the table
    does not cover the query, and pretending otherwise would fake the result."""

    def __init__(self):
        # keyed by (asin, slug, tier, modality)
        self.units: dict[tuple[str, str, int, str], Answer] = {}
        for (tier, modality), (d, _model) in STEP_DIRS.items():
            self._load(REPO_ROOT / "results" / d, tier, modality)
        for slug, d in EXTRA_TIER1_DIRS.items():
            self._load(REPO_ROOT / "results" / d, 1, "text", only=slug)

    def _load(self, d: Path, tier: int, modality: str,
              only: str | None = None) -> None:
        ans_p, calls_p = d / "subq_answers.csv", d / "subq_calls.csv"
        if not ans_p.exists():
            return
        cost, lat = {}, {}
        with calls_p.open() as f:
            for r in csv.DictReader(f):
                cost[(r["parent_asin"], r["predicate_slug"])] = float(r["cost_usd"])
                lat[(r["parent_asin"], r["predicate_slug"])] = float(r["latency_s"])
        for u in load_units(ans_p):
            if only and u["slug"] != only:
                continue
            key = (u["asin"], u["slug"])
            self.units[(u["asin"], u["slug"], tier, modality)] = Answer(
                direction=_direction(u["content"]),
                confidence=_confidence(u["content"], u["meta"]),
                cost=cost.get(key, 0.0), latency=lat.get(key, 0.0),
            )

    def get_many(self, asins: list[str], predicate: str, tier: int,
                 modality: str = "text", force_commit: bool = False) -> list[Answer]:
        # force_commit is a prompt-side flag (must-decide instruction); offline
        # replays a fixed recorded answer, so it is ignored here. The terminal
        # force-commit still takes effect via the engine's gate (_settled force=True).
        out = []
        for asin in asins:
            try:
                out.append(self.units[(asin, predicate, tier, modality)])
            except KeyError:
                raise KeyError(
                    f"no recorded answer for ({asin}, {predicate}, tier{tier}, "
                    f"{modality}) — the offline table does not cover this step"
                ) from None
        return out

    def coverage(self) -> dict:
        """slug -> set of (tier, modality) steps recorded for it."""
        out: dict = {}
        for (_a, p, t, m) in self.units:
            out.setdefault(p, set()).add((t, m))
        return out


def offline_routes(tree, coverage: dict) -> dict:
    """Trim each type's canonical ROUTE to the (tier, modality) steps that are
    RECORDED for every predicate of that type in this query. Honest: a step with
    no table is dropped, not faked (matches OfflineSource's raise-don't-guess)."""
    leaves_by_type: dict[str, list[str]] = {}
    for p in dict.fromkeys(tree.leaves):
        leaves_by_type.setdefault(PREDICATE_TYPE[p], []).append(p)
    routes: dict[str, tuple] = {}
    for typ, canonical in ROUTES.items():
        preds = leaves_by_type.get(typ, [])
        if not preds:
            routes[typ] = canonical
            continue
        routes[typ] = tuple(
            s for s in canonical
            if all((s.tier, s.modality) in coverage.get(p, set()) for p in preds)
        )
    return routes


def parse_query(q: str, known: list[str]):
    """Tiny parser for 'a AND b' / 'a OR b' / nested via parentheses.
    Real queries come from query_parser's JSON; this is just a CLI convenience."""
    q = q.strip()
    if q.startswith("(") and q.endswith(")"):
        depth = 0
        for i, ch in enumerate(q):
            depth += (ch == "(") - (ch == ")")
            if depth == 0 and i < len(q) - 1:
                break
        else:
            return parse_query(q[1:-1], known)
    for op in ("OR", "AND"):                       # OR binds loosest
        depth, parts, cur = 0, [], ""
        for tok in q.split():
            depth += tok.count("(") - tok.count(")")
            if depth == 0 and tok == op:
                parts.append(cur.strip()); cur = ""
            else:
                cur += tok + " "
        parts.append(cur.strip())
        if len(parts) > 1:
            return Node(op, tuple(parse_query(p, known) for p in parts))
    if q not in known:
        raise SystemExit(f"unknown predicate {q!r}; known: {known}")
    return Leaf(q)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--query", default="vegan AND indulgent_valentine")
    ap.add_argument("--chunk-size", type=int, default=100)
    ap.add_argument("--measure-size", type=int, default=200)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = OfflineSource()
    known = sorted(src.coverage())
    tree = parse_query(args.query, known)

    routes = offline_routes(tree, src.coverage())
    cfg = CascadeConfig(
        tiers=(1, 2, 3), thresholds=THRESHOLDS, predicate_type=PREDICATE_TYPE,
        routes=routes,
        chunk_size=args.chunk_size, measure_size=args.measure_size,
        batch=args.batch, workers=args.workers, seed=args.seed,
    )

    asins = sorted({a for (a, _p, _t, _m) in src.units})
    print(f"query      : {args.query}")
    print(f"tree       : {tree}")
    print(f"leaves     : {list(dict.fromkeys(tree.leaves))}")
    print(f"products   : {len(asins)}")
    print(f"params     : chunk_size={cfg.chunk_size} measure_size={cfg.measure_size} "
          f"batch={cfg.batch} workers={cfg.workers} seed={cfg.seed}")
    for typ in sorted({PREDICATE_TYPE[p] for p in dict.fromkeys(tree.leaves)}):
        full, used = ROUTES[typ], routes[typ]
        trimmed = [s.label for s in full if s not in used]
        note = f"   (offline-trimmed: {', '.join(trimmed)})" if trimmed else ""
        print(f"route {typ:<11}: {' -> '.join(s.label for s in used)}{note}")
    print()

    # ROUND 0 — measurement (no short-circuit) -> unbiased marginal k_F and k_T
    kill, m_cost, m_calls = learn_kill_rates(tree, asins, src, cfg)
    print(f"ROUND 0  measurement: {m_calls} calls, ${m_cost:.4f}")
    for p in dict.fromkeys(tree.leaves):
        print(f"   {p:22s} k_F={kill[(p, F)]:6.1%}   k_T={kill[(p, T)]:6.1%}")
    print()

    # execution chunks (granularity only — cost-neutral, §7.0)
    chs = chunks_of(asins, cfg)
    total = {"cost": 0.0, "calls": 0}
    memb: dict = {}
    from collections import Counter
    by_step: Counter = Counter()          # (tier, modality) -> units evaluated
    for ci, ch in enumerate(chs):
        r = run_cascade(tree, ch, src, cfg, kill=kill)
        memb.update(r.membership)
        total["cost"] += r.cost
        total["calls"] += r.calls
        for e in r.events:
            by_step[(e.tier, e.modality)] += 1
        print(f"  chunk{ci}: {len(ch)} products -> "
              f"TRUE={sum(v is True for v in r.membership.values())} "
              f"FALSE={sum(v is False for v in r.membership.values())} "
              f"INCONCL={sum(v is None for v in r.membership.values())}  "
              f"calls={r.calls} cost=${r.cost:.4f}")

    print(f"\nFINAL   TRUE={sum(v is True for v in memb.values())} "
          f"FALSE={sum(v is False for v in memb.values())} "
          f"INCONCLUSIVE={sum(v is None for v in memb.values())}")
    print("BY STEP " + "  ".join(
        f"t{t}·{'txt' if m == 'text' else 'txt+img'}={by_step[(t, m)]}"
        for (t, m) in sorted(by_step)))
    print(f"TOTALS  calls={total['calls']} (+{m_calls} measurement)  "
          f"cost=${total['cost']:.4f} (+${m_cost:.4f} measurement)")


if __name__ == "__main__":
    main()
