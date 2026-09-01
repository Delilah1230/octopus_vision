#!/usr/bin/env python3
"""
LIVE cascade — real API calls, no offline table.

Same engine as cascade_offline.py (scripts/cascade.py); only the AnswerSource
differs. That is the point of the AnswerSource seam: the logic validated offline
is the logic that runs here.

Key properties:
  * CACHED — each (product, predicate, tier) is called at most ONCE per run, so
    the Round-0 measurement calls are reused by the cascade, not paid twice.
  * Only calls what the cascade asks for. Short-circuited products are never sent
    to the model at all — unlike the offline replay (where every cell was
    pre-paid), here that saving is real money.
  * batch/workers are honoured: each get_many is ONE predicate at ONE tier over
    many products — exactly the unit batching and concurrency can exploit.
  * Every call is logged in subquestion_probe.py's schema, so the run leaves
    behind a reusable table.

Usage:
    python scripts/cascade_live.py --limit 20 --dry-run    # plan + cost estimate
    python scripts/cascade_live.py --limit 200 --chunk-size 20
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cascade import (  # noqa: E402
    Answer, CascadeConfig, F, ROUTES, T, U, chunks_of, effective_batch,
    learn_kill_rates, run_cascade,
)
from cascade_offline import PREDICATE_TYPE, THRESHOLDS, parse_query  # noqa: E402
from connect_api import load_env, _require_key  # noqa: E402
from subq_score import CONTENT_MAP, META_MAP  # noqa: E402
import subquestion_probe as _sq  # noqa: E402
from subquestion_probe import (  # noqa: E402
    ANSWERS_COLUMNS, CALLS_COLUMNS, PREDICATES, _append_rows, _ensure_header,
    load_gt_soft_labels, load_metas, run_batch, run_one,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Per-tier wall-clock deadline. Sized from MEASURED latency, not guessed:
# tier1 p95=0.81s max=0.88s | tier2 p50~3.5s | tier3 p50=10.6s p95=14.8s max=18.8s.
# A call that blows past ~2-3x the observed max is pathological (runaway
# thinking — we have seen 62914 thinking tokens on one call), and retrying just
# re-rolls the same dice at full price. So: short deadline, ONE retry, then let
# it degrade to U (inconclusive) and move on. A stuck leaf must never hold the
# pipeline hostage — the cascade already has a well-defined answer for "unknown".
TIER_TIMEOUT_S = {1: 10, 2: 25, 3: 40}
MAX_ATTEMPTS = 2          # total attempts per call. Was 5 outer x 2 inner = 10
                          # attempts x 60s = up to 600s PER CALL, which is how one
                          # chunk took 2961s (49 min) while its neighbour took 41s.

# Per-tier LLM configuration: model + the two reasoning knobs, together. Built
# from TIER_MODEL below by _rebuild_tier_llm(); runners call set_reasoning() to
# override the knobs for a whole run (e.g. the baseline-matched configs, which
# need thinking off everywhere and CoT on or off uniformly).
#
# Defaults preserve historical behaviour exactly: thinking_budget=None means the
# provider default, which is off for flash-lite and on for flash/pro.
TIER_LLM: dict[int, "_sq.LLMConfig"] = {}

TIER_MODEL = {
    1: "gemini-2.5-flash-lite",     # thinking off by default -> deterministic
    2: "gemini-2.5-flash",          # thinking KEPT on (default budget) — the
                                    # reasoning is the point of tier 2. Swapped
                                    # off gemini-3-flash-preview 2026-07-14: same
                                    # median thinking (621 vs 675 tok) but a much
                                    # tighter tail (max 1459 vs 2558 tok; 7.99s vs
                                    # 12.28s), and cheaper ($0.30/$2.50 vs
                                    # $0.50/$3.00). flash-3's tail caused all 6
                                    # timeouts and ~483s of the 1245s live run.
    3: "gemini-2.5-pro",
}
# rough per-call cost incl. thinking, for --dry-run only (see thinking_cost_model.md)
TIER_EST = {1: 0.00013, 2: 0.0020, 3: 0.0133}


def _rebuild_tier_llm(thinking_budget=None, cot: bool = False,
                      use_default_thinking: bool = True) -> None:
    """(Re)build TIER_LLM from TIER_MODEL.

    use_default_thinking=True keeps each tier's provider default (the historical
    behaviour: flash-lite silent, flash/pro thinking). Pass False with an explicit
    thinking_budget to force one value on every tier -- what a baseline-matched run
    needs, since LOTUS/Palimpzest are run with reasoning_effort="disable".

    2026-08-03: tier 1 NEVER gets CoT, regardless of the `cot` arg. Tier 1 is
    the cheap yes/no/cannot_determine screen -- its output schema is answer-only
    (see subquestion_probe.INSTRUCTION); a `reasoning` field there was inflating
    the one tier whose entire job is being fast and cheap. `cot` still applies
    normally to tier 2/3, where the reasoning is the point.
    """
    TIER_LLM.clear()
    for tier, model in TIER_MODEL.items():
        TIER_LLM[tier] = _sq.LLMConfig(
            model=model,
            thinking_budget=None if use_default_thinking else thinking_budget,
            cot=(cot and tier != 1),
        )


def set_reasoning(thinking: str = "default", cot: bool = False) -> None:
    """Runner-facing switch for the two LLM-invocation knobs.

    thinking: "default" -> per-tier provider default; "off" -> budget 0 everywhere.
    cot:      write a reason before each sub-answer, on tier 2/3 (tier 1 never
              gets it -- see _rebuild_tier_llm).
    """
    if thinking not in ("default", "off"):
        raise ValueError(f"thinking must be 'default' or 'off', got {thinking!r}")
    _rebuild_tier_llm(thinking_budget=0, cot=cot,
                      use_default_thinking=(thinking == "default"))


_rebuild_tier_llm()

PRED_BY_SLUG = {p["slug"]: p for p in PREDICATES}
CERT_WC, CERT_WM = 0.7, 0.3


def _score(answer_rows: list[dict]) -> tuple[float, float]:
    """5 sub-answers -> (content_score, meta_score). Same mapping as subq_score."""
    c = [CONTENT_MAP.get(r["answer"], 0.5) for r in answer_rows if r["sub_type"] == "content"]
    m = [META_MAP.get(r["answer"], 0.5) for r in answer_rows if r["sub_type"] == "meta"]
    return (sum(c) / len(c) if c else 0.5), (sum(m) / len(m) if m else 0.5)


def is_related_scheme(pred: dict) -> bool:
    """True iff this predicate uses the RELATED-QUESTION scheme (paper §1.3):
    n same-direction questions, no content/meta split. Decided by the predicate's
    own shape, so the two schemes coexist -- grocery's 3-content+2-meta predicates
    keep their scorer, and nothing has to be migrated in lockstep."""
    subqs = pred.get("subquestions") or ()
    return bool(subqs) and all(stype == "related" for _sid, stype, _text in subqs)


def _verdict_related(answer_rows: list[dict], force: bool = False) -> tuple[str, float]:
    """RELATED-QUESTION scheme (paper §1.3) -> (direction, confidence).

        r = fraction of the n questions answered TRUE
        c = max(r, 1-r)
        r > 1/2 -> T ;  r < 1/2 -> F ;  r == 1/2 -> U (undetermined)

    Every question is phrased in the SAME direction (answer "T" == evidence the
    predicate holds), so a T and an F mean the same thing on every question and
    the vote is well defined. There is no content/meta split and no weighting:
    with nothing claiming the questions decompose the predicate, there is no
    "why are they equally weighted" question to answer.

    theta does NOT apply here. The boundary is 1/2 by construction -- c is
    symmetric about it -- so a per-predicate direction threshold would break the
    identity that makes c a confidence at all.

    U at exactly n/2 is the point of choosing an EVEN n: with an odd n a majority
    always exists and undetermined could never arise, yet the data model depends
    on it existing (§3.1 Filter, provenance NULL overloading, Table 2). A U here
    is recorded as a NULL tag with provenance, exactly like any other U.

    force=True is the TERMINAL step: there is no next stage to escalate a tie to,
    so the tie has to land somewhere. It lands on F, and the reason is the
    asymmetry the whole system is built on rather than any property of ties:

      * a tie sent to F is a possible FALSE NEGATIVE -- the row drops out of one
        answer, and that is the end of it.
      * a tie sent to T is a possible FALSE POSITIVE -- it enters the answer, it
        is written to the tag, and a later predicate bound by EQUIVALENT or by
        subsumption INHERITS it. One weak judgment is exported to every query
        that touches the predicate.

    Measured on movie (2026-08-12): sending ties to T put review 2241537 (r=1/2,
    truly NEGATIVE) into the clearly-positive tag; BROADER then handed it to Q2 as
    a positive review and Q2 fell from 1.000 to 0.800. The direct judgment of the
    same row on the broader predicate was FALSE -- so T at a tie manufactured a
    value the model itself contradicts.

    An earlier version sent ties to T on the argument that all questions are asked
    in the same direction, so a tie means half the evidence supports the predicate
    while an unsupported row sits at r=0. That reasoning is sound but it only says
    ties lean positive; it says nothing about what a wrong tie COSTS, and the two
    directions do not cost the same. (The 262-of-264 measurement that supported it
    does not generalise either: on the taken_3 inheritance set, ties were 1-1.)

    A single-stage cascade is always terminal, so it never emits U. U remains
    reachable the moment there is a stage to escalate to, which is where the data
    model needs it.

    KNOWN BLIND SPOT (documented in the paper): the questions are all positive, so
    a row with NO relevant information answers F to all of them -> r = 0 -> c = 1.0
    -> a HIGH-CONFIDENCE FALSE. That row is precisely the one that most deserves
    escalation. This is a property of the default instance, not of the framework --
    the scoring function is replaceable.
    """
    vals = [CONTENT_MAP.get(r["answer"], 0.5) for r in answer_rows]
    if not vals:
        return U, 0.5
    r = sum(vals) / len(vals)
    if r > 0.5:
        direction = T
    elif r < 0.5:
        direction = F
    else:
        direction = F if force else U
    return direction, max(r, 1.0 - r)


class LiveSource:
    """Real model calls behind the AnswerSource interface."""

    def __init__(self, client, metas, gt, cfg: CascadeConfig, out_dir: Path,
                 dry_run: bool = False, pred_by_slug: dict | None = None,
                 meta_provider=None):
        self.client, self.metas, self.gt, self.cfg = client, metas, gt, cfg
        self.dry_run = dry_run
        # Optional lazy meta source: meta_provider(asins) -> {asin: meta}. When set,
        # product text is fetched ON DEMAND for exactly the batch about to be called
        # (not pre-loaded), so "how much to fetch" follows the cascade flow and the
        # caller can evict between chunks. The standalone experiment passes a full
        # metas dict up front and leaves this None. See cascade_planner.
        self.meta_provider = meta_provider
        # Predicate definitions (slug -> {slug, key, subquestions}). Defaults to the
        # hand-written 3 (module global) for the standalone experiment; the system
        # planner injects the parsed query's predicates (subquestions from cascade_agent).
        self.pred_by_slug = pred_by_slug if pred_by_slug is not None else PRED_BY_SLUG
        self.cache: dict[tuple[str, str, int, str], Answer] = {}
        self.lock = threading.Lock()
        self.calls = 0            # real model calls (batches), not units
        self.gave_up = 0          # units abandoned after MAX_ATTEMPTS -> U
        self.units = 0            # (product, predicate, tier) resolved
        self.cost = 0.0
        self.wall = 0.0
        self.per_tier: dict[int, dict] = {t: {"units": 0, "calls": 0, "cost": 0.0}
                                          for t in cfg.tiers}
        self.out_dir = out_dir
        if not dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            _ensure_header(out_dir / "subq_answers.csv", ANSWERS_COLUMNS)
            _ensure_header(out_dir / "subq_calls.csv", CALLS_COLUMNS)

    def get_many(self, asins: list[str], predicate: str, tier: int,
                 modality: str = "text", force_commit: bool = False) -> list[Answer]:
        key = lambda a: (a, predicate, tier, modality)
        todo = [a for a in asins if key(a) not in self.cache]
        if todo:
            self._fetch(todo, predicate, tier, modality, force_commit)
        return [self.cache[key(a)] for a in asins]

    def _fetch(self, asins: list[str], predicate: str, tier: int,
               modality: str, force_commit: bool = False) -> None:
        pred = self.pred_by_slug[predicate]
        related_scheme = is_related_scheme(pred)
        llm = TIER_LLM[tier]
        model = llm.model
        # WORKERS FIRST, BATCH SECOND. cfg.batch is a cap; we use the smallest
        # batch that still fills every worker. Batching before parallelising
        # starves the workers (20 items, batch=5, workers=8 -> 4 groups -> half
        # the pool idle) and batching is the worse tool for latency anyway.
        # IMAGE modality forces batch=1: run_batch is text-only (multiple images
        # in one prompt wrecks the product<->answer mapping — see subquestion_probe).
        # force_commit (terminal oracle) used to ALSO force batch=1 (run_batch had
        # no force_commit prompt variant) -- fixed 2026-08-03 (build_batch_prompt/
        # run_batch now both take force_commit). This mattered a lot: in the
        # text-only route tier 2 is always terminal (always force_commit), so it
        # was going through the unbatched path 100% of the time -- measured at
        # 98% of a run's total cost/latency (param_search_checkpoint.md).
        B = 1 if modality != "text" else effective_batch(len(asins), self.cfg)
        groups = [asins[i:i + B] for i in range(0, len(asins), B)]

        if self.dry_run:                       # count only, no calls
            with self.lock:
                for g in groups:
                    est = TIER_EST[tier] * len(g)
                    self.calls += 1
                    self.units += len(g)
                    self.cost += est
                    self.per_tier[tier]["units"] += len(g)
                    self.per_tier[tier]["calls"] += 1
                    self.per_tier[tier]["cost"] += est
                    for a in g:
                        self.cache[(a, predicate, tier, modality)] = Answer(F, 0.0, est, 0.0)
            return

        # Lazy fetch: pull product text for exactly this batch, only now (a real
        # call is about to happen). Cache hits never reach here, so settled-early
        # products are never fetched.
        if self.meta_provider is not None:
            missing = [a for a in asins if a not in self.metas]
            if missing:
                self.metas.update(self.meta_provider(missing))

        t0 = time.time()

        timeout_s = TIER_TIMEOUT_S[tier]

        def work(group: list[str]):
            # ONE retry, then give up and let the answer degrade to U. run_one/
            # run_batch already turn a failure into MISSING sub-answers ->
            # content_score 0.5 -> confidence 0.15 -> below every tau -> U, which
            # is exactly the right semantics for "we could not find out".
            for attempt in range(MAX_ATTEMPTS):
                if len(group) == 1:
                    arows, crow = run_one(
                        self.client, group[0], self.metas[group[0]], pred, self.gt,
                        model=model, temperature=0.0, timeout_s=timeout_s,
                        max_retries=0, modality=modality, force_commit=force_commit,
                        thinking_budget=llm.thinking_budget, cot=llm.cot)
                    crow.setdefault("batch_id", ""); crow.setdefault("batch_size", 1)
                    crows = [crow]
                else:
                    # group>1 only happens for text (image forces B=1 above).
                    arows, crows = run_batch(
                        self.client, group, self.metas, pred, self.gt,
                        model=model, temperature=0.0,
                        timeout_s=timeout_s * 2, max_retries=0,
                        thinking_budget=llm.thinking_budget, cot=llm.cot,
                        force_commit=force_commit)
                if not crows[0]["parse_errors"].startswith("call_error"):
                    return group, arows, crows
                if attempt < MAX_ATTEMPTS - 1:
                    time.sleep(1.0)
            with self.lock:
                self.gave_up += len(group)
            return group, arows, crows

        with ThreadPoolExecutor(max_workers=self.cfg.workers) as ex:
            for group, arows, crows in ex.map(work, groups):
                with self.lock:
                    _append_rows(self.out_dir / "subq_answers.csv", ANSWERS_COLUMNS, arows)
                    _append_rows(self.out_dir / "subq_calls.csv", CALLS_COLUMNS, crows)
                    self.calls += 1
                    for a, crow in zip(group, crows):
                        rows = [r for r in arows if r["parent_asin"] == a]
                        if related_scheme:
                            direction, confidence = _verdict_related(rows, force=force_commit)
                        else:
                            cs, ms = _score(rows)
                            theta = self.cfg.theta(predicate, tier)
                            direction = T if cs > theta else F
                            # confidence stays anchored at 0.5, NOT theta: it measures
                            # how polarized the raw content score is, independent of
                            # where the decision boundary sits. Re-centering it on
                            # theta would also silently rescale what tau means.
                            confidence = CERT_WC * (2 * abs(cs - 0.5)) + CERT_WM * ms
                        ans = Answer(
                            direction=direction,
                            confidence=confidence,
                            cost=float(crow["cost_usd"]),
                            latency=float(crow["latency_s"]),
                        )
                        self.cache[(a, predicate, tier, modality)] = ans
                        self.units += 1
                        self.cost += ans.cost
                        self.per_tier[tier]["units"] += 1
                        self.per_tier[tier]["cost"] += ans.cost
                    self.per_tier[tier]["calls"] += 1
        self.wall += time.time() - t0


def main() -> None:
    # Unbuffered: results are streamed per chunk, so they must actually appear as
    # they happen. Python block-buffers stdout when it is not a tty, which turns
    # a streaming run into a silent wait followed by a dump.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--query", default="vegan AND indulgent_valentine")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--chunk-size", type=int, default=20)
    ap.add_argument("--measure-size", type=int, default=100)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO_ROOT / "results" / "cascade_live")
    ap.add_argument("--dry-run", action="store_true",
                    help="plan + cost estimate, zero API calls")
    args = ap.parse_args()

    load_env()
    client = None
    if not args.dry_run:
        from google import genai
        client = genai.Client(api_key=_require_key("GEMINI_API_KEY"))

    metas = load_metas()
    gt = load_gt_soft_labels()
    asins = sorted(metas.keys())[: args.limit]
    tree = parse_query(args.query, sorted(PRED_BY_SLUG))

    cfg = CascadeConfig(
        tiers=(1, 2, 3), thresholds=THRESHOLDS, predicate_type=PREDICATE_TYPE,
        chunk_size=args.chunk_size, measure_size=args.measure_size,
        batch=args.batch, workers=args.workers, seed=args.seed,
    )
    src = LiveSource(client, metas, gt, cfg, args.out_dir, dry_run=args.dry_run)

    print("=" * 68)
    print(f"{'DRY RUN — no API calls' if args.dry_run else 'LIVE — real API calls'}")
    print("=" * 68)
    print(f"query    : {args.query}")
    print(f"products : {len(asins)}")
    print(f"params   : chunk_size={cfg.chunk_size} measure_size={cfg.measure_size} "
          f"batch={cfg.batch} workers={cfg.workers}")
    print(f"tiers    : " + ", ".join(f"{t}={TIER_MODEL[t]}" for t in cfg.tiers))
    for typ in sorted({PREDICATE_TYPE[p] for p in dict.fromkeys(tree.leaves)}):
        print(f"route {typ:<11}: " + " -> ".join(s.label for s in ROUTES[typ]))
    if not args.dry_run:
        print(f"out      : {args.out_dir}")
    print()

    t0 = time.time()
    kill, m_cost, m_units = learn_kill_rates(tree, asins, src, cfg)
    print(f"ROUND 0  measurement on {cfg.measure_size} products (no short-circuit) "
          f"— {time.time() - t0:.1f}s")
    print(f"         {m_units} units, ${m_cost:.4f}")
    for p in dict.fromkeys(tree.leaves):
        print(f"         {p:22s} k_F={kill[(p, F)]:6.1%}  k_T={kill[(p, T)]:6.1%}")
    order = sorted(dict.fromkeys(tree.leaves),
                   key=lambda p: -kill[(p, F)])
    print(f"         -> learned order (AND: highest k_F first): {' , '.join(order)}",
          flush=True)

    # cascade-level trace: the decisions, not just the calls. subq_calls.csv has
    # latency/cost/tokens but knows nothing about confidence, tier or the gate.
    trace_cols = ["seq", "chunk", "asin", "predicate", "tier", "modality",
                  "direction", "confidence", "tau", "value", "cost_usd", "latency_s"]
    trace_p = args.out_dir / "cascade_trace.csv"
    if not args.dry_run:
        _ensure_header(trace_p, trace_cols)

    memb: dict = {}
    seq = 0
    chunks = chunks_of(asins, cfg)
    for ci, ch in enumerate(chunks):
        # snapshot so we can report THIS chunk's spend, not the running total
        c0 = src.cost
        t_c0 = time.time()
        tier0 = {t: dict(src.per_tier[t]) for t in cfg.tiers}

        r = run_cascade(tree, ch, src, cfg, kill=kill)
        memb.update(r.membership)

        if not args.dry_run:
            rows = []
            for e in r.events:
                rows.append({"seq": seq, "chunk": ci, "asin": e.asin,
                             "predicate": e.predicate, "tier": e.tier,
                             "modality": e.modality, "direction": e.direction,
                             "confidence": f"{e.confidence:.3f}", "tau": e.tau,
                             "value": e.value, "cost_usd": f"{e.cost:.6f}",
                             "latency_s": f"{e.latency:.3f}"})
                seq += 1
            _append_rows(trace_p, trace_cols, rows)

        # ── stream this chunk's RESULT out now; the run keeps going ──
        matched = sorted(a for a, v in r.membership.items() if v is True)
        incon = sorted(a for a, v in r.membership.items() if v is None)
        dt = time.time() - t_c0
        print(f"\n── chunk {ci} " + "─" * 24 + f" {len(ch)} products ── {dt:5.1f}s ──")
        if matched:
            for i in range(0, len(matched), 6):
                head = "   ✓ MATCHED:" if i == 0 else "              "
                print(f"{head} " + "  ".join(matched[i:i + 6]))
        else:
            print("   ✓ MATCHED: (none)")
        print(f"   · not matched {sum(v is False for v in r.membership.values()):>3}"
              f"   · inconclusive {len(incon):>2}"
              + (f"  {' '.join(incon[:4])}{' …' if len(incon) > 4 else ''}" if incon else ""))
        parts = []
        for t in cfg.tiers:
            du = src.per_tier[t]["units"] - tier0[t]["units"]
            dc = src.per_tier[t]["cost"] - tier0[t]["cost"]
            if du:
                parts.append(f"tier{t} {du:>3}u ${dc:.4f}")
        print("   " + "   ".join(parts) if parts else "   (all cached)")
        print(f"   chunk ${src.cost - c0:.4f}   |   cumulative: {src.units} units  "
              f"${src.cost:.4f}   ({ci + 1}/{len(chunks)} chunks)", flush=True)
    wall = time.time() - t0

    print(f"\nFINAL    TRUE={sum(v is True for v in memb.values())} "
          f"FALSE={sum(v is False for v in memb.values())} "
          f"INCONCLUSIVE={sum(v is None for v in memb.values())}")
    print(f"\n{'tier':<6}{'model':<26}{'units':>7}{'calls':>7}{'cost$':>10}")
    for t in cfg.tiers:
        s = src.per_tier[t]
        print(f"{t:<6}{TIER_MODEL[t]:<26}{s['units']:>7}{s['calls']:>7}{s['cost']:>10.4f}")
    print(f"{'':<32}{src.units:>7}{src.calls:>7}{src.cost:>10.4f}")
    print(f"\nwall-clock: {wall:.1f}s   (units include the measurement round; "
          f"each (product,predicate,tier) called at most once)")
    if src.gave_up:
        print(f"⚠️  {src.gave_up} unit(s) gave up after {MAX_ATTEMPTS} attempts "
              f"-> degraded to U (inconclusive) rather than blocking the run")
    if args.dry_run:
        print("\n⚠️  DRY RUN IS A WORST-CASE UPPER BOUND, not an estimate: it fakes every\n"
              "    answer as confidence=0, so nothing settles, nothing short-circuits,\n"
              "    and every product walks all 3 tiers. The real run settles most\n"
              "    products at tier 1 and costs far less — the gap IS the cascade.")
    else:
        print(f"tables written -> {args.out_dir}")
        print(f"  subq_answers.csv   raw 5 sub-answers per (product,predicate)")
        print(f"  subq_calls.csv     per-call model/tokens/cost/latency/batch")
        print(f"  cascade_trace.csv  per-decision: tier, direction, confidence, tau, "
              f"T/F/U, cost, latency")


if __name__ == "__main__":
    main()
