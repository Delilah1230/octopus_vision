#!/usr/bin/env python3
"""
Semantic query planner — runs the CASCADE inside the real query pipeline.

This is octopimizer Phase 2, previously a stub. It replaces the old
profiling+epoch_planner path with the cascade:

    ParsedQuery (from query_parser)
        │  initialize_state() already ran (predicate_store + execution_state)
        ▼
    cascade_planner.run_semantic_cascade
        │  • candidates + product text  ← DB (category table)
        │  • 5 sub-questions per predicate ← cascade_agent
        │  • cascade engine (scripts/cascade.py) over the AND/OR tree,
        │    LLM calls via cascade_live.LiveSource (routes, oracle, batching)
        │  • settled answers            → execution_result (write_final_result)
        │  • objective tags             → category table + tag_meta (write_tag_cache)
        ▼
    matching parent_asins  (root of the logic tree = TRUE)

The engine, routes, oracle force-commit and thresholds are exactly the ones
validated offline — only the AnswerSource's metas come from the DB instead of a
JSONL file, and the results are written to the DB instead of only CSV.

Scope note: predicate → 5 sub-questions goes through cascade_agent. Known
predicates (matching subquestion_probe.PREDICATES by NL) use the validated
hand-written sub-questions (canned); unknown predicates fall back to cascade_agent
mode="llm" (auto-generated, not yet validated — see notes/subquestion_design_risk.md).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import psycopg2
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cascade import (  # noqa: E402
    ABSORB, CascadeConfig, F, Leaf, Node, ROUTES, T, U,
    chunks_of, eval_tree, learn_kill_rates, run_cascade,
)
from cascade_agent import CascadeAgent, PredicateSpec  # noqa: E402
from cascade_live import LiveSource  # noqa: E402
from cascade_offline import THRESHOLDS  # noqa: E402
from connect_api import _require_key, load_env  # noqa: E402
from query_parser import canonicalize_predicate_to_column  # noqa: E402
from state_manager import (  # noqa: E402
    get_candidate_asins, get_known_results, get_reuse_scores,
    prepopulate_results_from_tags, record_predicate_usage,
    write_final_result, write_tag_cache,
)
from subquestion_probe import PREDICATES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_CONFIG = {
    "host": "127.0.0.1", "port": 5432,
    "dbname": "octopus", "user": "octopus_user", "password": "octopus",
}

# NL -> validated hand-written slug (the canned path). Everything else is auto-generated.
_KNOWN_SLUG_BY_NL = {p["key"]: p["slug"] for p in PREDICATES}


# ── predicate → slug + sub-questions ──────────────────────────────────────────

def _slug_for(nl: str) -> tuple[str, bool]:
    """(slug, is_known). Known predicates reuse the validated slug; unknown ones get
    a canonical slug derived from the NL (tag_ prefix stripped).

    2026-08-10: recognition used to be by EXACT NL string only, which is too
    brittle to be relied on -- the parser is an LLM and rephrases the same
    predicate between runs ("is clearly positive" one run, "is a clearly positive
    review" the next). One wording was recognised, the other silently fell
    through to runtime sub-question generation. So also accept a predicate whose
    CANONICAL SLUG is registered: same column, same definition, therefore the
    same predicate by the only identity the tag store actually uses."""
    if nl in _KNOWN_SLUG_BY_NL:
        return _KNOWN_SLUG_BY_NL[nl], True
    slug = canonicalize_predicate_to_column(nl)
    slug = slug[4:] if slug.startswith("tag_") else slug
    return slug, slug in {p["slug"] for p in PREDICATES}


def _predicate_defs(parsed_query, subq_mode: str = "auto"):
    """Build the cascade-side predicate machinery from ParsedQuery.semantic:
      pred_by_slug : slug -> {key, slug, subquestions}  (LiveSource needs this)
      slug_by_idx  : predicate_idx -> slug              (tree leaves + result write)
      predicate_type: slug -> 'objective' | 'subjective' (cfg.tau / thresholds)
    """
    canned = CascadeAgent(mode="canned")
    llm = None
    pred_by_slug, slug_by_idx, predicate_type = {}, {}, {}
    for idx, p in enumerate(parsed_query.semantic):
        slug, known = _slug_for(p.nl)
        # A predicate bound to a cached tag by EQUIVALENT or NEGATION INHERITS that
        # tag's questions. For EQUIVALENT the truth condition is identical, so the
        # questions are literally the right ones. For NEGATION they are the right
        # questions too -- they measure the cached predicate, and this one is its
        # complement, so a row that still needs evaluating is answered by asking
        # about the cached predicate and inverting.
        # Without this, a NEGATION reuse demands 6 questions for a predicate that is
        # never asked, and the registry guard kills a query that was about to cost
        # nothing (measured: Q8 died here after the negation match succeeded).
        rel = getattr(p, "tag_relation", None)
        if rel in ("EQUIVALENT", "NEGATION") and p.tag_column:
            ref = canonicalize_predicate_to_column(p.tag_column)
            ref = ref[4:] if ref.startswith("tag_") else ref
            ref = p.tag_column[4:] if p.tag_column.startswith("tag_") else p.tag_column
            if ref in {q["slug"] for q in PREDICATES}:
                slug, known = ref, True
                print(f"[cascade_planner] {p.nl!r} inherits sub-questions from "
                      f"{ref!r} ({rel})")
        use_canned = known if subq_mode == "auto" else (subq_mode == "canned")
        if use_canned:
            subqs = canned.generate(slug)
        else:
            llm = llm or CascadeAgent(mode="llm")
            subqs = llm.generate(PredicateSpec(slug=slug, key=p.nl))
        pred_by_slug[slug] = {
            "key": p.nl, "slug": slug,
            "subquestions": [sq.as_tuple() for sq in subqs],
        }
        slug_by_idx[idx] = slug
        # "objective_semantic" -> "objective" (thresholds/routes key)
        predicate_type[slug] = p.predicate_type.replace("_semantic", "")
    return pred_by_slug, slug_by_idx, predicate_type


def _tree_from_logic(logic, slug_by_idx: dict):
    """LogicNode | int | None  ->  cascade Leaf | Node (leaves keyed by slug)."""
    if logic is None:
        if len(slug_by_idx) != 1:
            raise ValueError("null logic tree needs exactly one predicate")
        return Leaf(slug_by_idx[0])
    if isinstance(logic, int):
        return Leaf(slug_by_idx[logic])
    # LogicNode(op, children)
    return Node(logic.op, tuple(_tree_from_logic(c, slug_by_idx) for c in logic.children))


# ── product text from the DB (→ the meta-dict shape run_one expects) ──────────

def _build_metas(asins: list[str], table: str) -> dict[str, dict]:
    """Category-table rows -> {asin: meta}. meta keys match what subquestion_probe's
    format_text_block (title/features/description) and _meta_main_image_url (images)
    read. A single image_url becomes one MAIN image entry."""
    metas: dict[str, dict] = {}
    with psycopg2.connect(**DB_CONFIG) as conn, \
            conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            f"SELECT parent_asin, title, features, description, image_url "
            f"FROM {table} WHERE parent_asin = ANY(%s)",
            (asins,),
        )
        for r in cur.fetchall():
            url = r.get("image_url")
            metas[r["parent_asin"]] = {
                "parent_asin": r["parent_asin"],
                "title": r.get("title"),
                "features": r.get("features") or [],
                "description": r.get("description") or [],
                "images": [{"large": url, "variant": "MAIN"}] if url else [],
            }
    return metas


# ── result write-back ─────────────────────────────────────────────────────────

_STATUS = {T: ("matched", True), F: ("not_matched", False), U: ("inconclusive", None)}

# per-DECISION analysis log (the LLM CALL log — tokens/cost/raw — is subq_calls.csv,
# written by LiveSource; this adds the cascade's gate decisions on top).
_TRACE_COLS = ["seq", "chunk", "asin", "predicate", "tier", "modality", "direction",
               "confidence", "tau", "value", "cost_usd", "latency_s"]


def _append_trace(path: Path, chunk_idx: int, seq0: int, events, cfg) -> int:
    rows = []
    seq = seq0
    for e in events:
        rows.append({"seq": seq, "chunk": chunk_idx, "asin": e.asin,
                     "predicate": e.predicate, "tier": e.tier, "modality": e.modality,
                     "direction": e.direction, "confidence": f"{e.confidence:.3f}",
                     "tau": e.tau, "value": e.value, "cost_usd": f"{e.cost:.6f}",
                     "latency_s": f"{e.latency:.3f}"})
        seq += 1
    with path.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=_TRACE_COLS).writerows(rows)
    return seq


def _write_chunk_results(events, idx_by_slug: dict) -> None:
    """From a chunk's LeafOutcome events, write one execution_result row per settled
    (product, predicate) — the LAST event for each pair is its terminal/settling step.
    Short-circuited pairs have no event -> no row (absence = not processed)."""
    final: dict[tuple[str, str], object] = {}
    for e in events:
        final[(e.asin, e.predicate)] = e          # later step overwrites earlier
    for (asin, slug), e in final.items():
        status, direction = _STATUS[e.value]
        write_final_result(
            parent_asin=asin, predicate_idx=idx_by_slug[slug],
            status=status, direction=direction, confidence=e.confidence,
            settled_tier=e.tier, settled_modality=e.modality,
        )


# ── short-circuit analysis (for verification / bug-hunting) ───────────────────

def _parent_ops(tree) -> dict:
    """leaf predicate -> its parent operator ('AND'|'OR')."""
    out: dict = {}

    def walk(n, op="AND"):
        if isinstance(n, Leaf):
            out[n.predicate] = op
        else:
            for c in n.children:
                walk(c, n.op)

    walk(tree)
    return out


def _short_circuit_report(label: str, tree, asins, known: dict, events, out_dir: Path):
    """Classify every (product, predicate) into reused / evaluated / short-circuited,
    attribute each short-circuit to the sibling value that absorbed it, and write it
    all out so a run can be audited for reuse/short-circuit bugs.

    reused        = value came from the tag cache (pre-seeded `known`), not evaluated.
    evaluated     = the cascade actually ran it (has events).
    short_circuited = neither — the tree settled via another predicate, so it was
                    never needed. Its cause is a sibling that took the absorbing
                    value (AND absorbs F, OR absorbs T)."""
    from collections import Counter, defaultdict

    leaves = list(dict.fromkeys(tree.leaves))
    evaluated: dict = defaultdict(dict)
    for e in events:
        evaluated[e.asin][e.predicate] = e.value        # last step per (asin,pred)

    # classes:
    #   reused          value from tag cache (pre-seeded known), not evaluated
    #   settled         evaluated AND reached T/F
    #   abandoned_U     evaluated some tier(s) but short-circuited before settling
    #                   (final U) — partial work that the sibling's absorb cut off
    #   short_circuited never evaluated at all (sibling absorbed before its turn)
    klass: dict = {}
    state: dict = {}
    for a in asins:
        st = {}
        for p in leaves:
            if a in known and p in known[a]:
                klass[(a, p)] = "reused"; st[p] = known[a][p]
            elif p in evaluated.get(a, {}):
                v = evaluated[a][p]
                klass[(a, p)] = "settled" if v in (T, F) else "abandoned_U"
                st[p] = v
            else:
                klass[(a, p)] = "short_circuited"; st[p] = U
        state[a] = st

    parent = _parent_ops(tree)
    cls = {p: Counter(klass[(a, p)] for a in asins) for p in leaves}
    reasons: Counter = Counter()
    for a in asins:
        for p in leaves:
            if klass[(a, p)] == "short_circuited":
                absorb = ABSORB[parent[p]]
                cause = ", ".join(f"{q}={state[a][q]}" for q in leaves
                                  if q != p and state[a].get(q) == absorb) or "(unknown)"
                reasons[(p, cause)] += 1

    memb = Counter()
    for a in asins:
        v = eval_tree(tree, state[a])
        memb["matched" if v == T else "not_matched" if v == F else "inconclusive"] += 1

    lines = [f"═══ short-circuit report: {label} ═══",
             f"candidates: {len(asins)}",
             f"membership: " + "  ".join(f"{k}={v}" for k, v in sorted(memb.items())),
             ""]
    lines.append(f"{'predicate':24}{'reused':>7}{'settled':>8}{'abandon_U':>10}"
                 f"{'short_circ':>11}")
    for p in leaves:
        c = cls[p]
        lines.append(f"{p:24}{c['reused']:>7}{c['settled']:>8}{c['abandoned_U']:>10}"
                     f"{c['short_circuited']:>11}")
    lines.append("")
    lines.append("short-circuit reasons (predicate never-evaluated ← because sibling absorbed):")
    for (p, cause), n in reasons.most_common():
        lines.append(f"  {p:24} skipped ×{n:<5} because {cause}")
    # invariant checks — surface likely bugs
    lines.append("")
    lines.append("invariant checks:")
    for p in leaves:
        tot = sum(cls[p].values())
        lines.append(f"  {p:24} reused+settled+abandon+short = {tot} (== {len(asins)}? "
                     f"{'OK' if tot == len(asins) else 'MISMATCH!'})")
    lines.append(f"  (objective 'settled' count should match its flushed tag count)")
    lines.append(f"  inconclusive == 0 (oracle)? {'OK' if memb['inconclusive'] == 0 else 'NO — ' + str(memb['inconclusive'])}")

    report = "\n".join(lines)
    print(report)
    (out_dir / f"short_circuit_{label}.txt").write_text(report + "\n")
    return {"class": cls, "reasons": reasons, "membership": memb}


# ── entry point ───────────────────────────────────────────────────────────────

def run_semantic_cascade(parsed_query, *, client=None, chunk_size: int = 50,
                         measure_size: int = 100, workers: int = 8, seed: int = 42,
                         limit: int | None = None, subq_mode: str = "auto",
                         out_dir: Path | None = None, label: str = "query",
                         direction_theta: dict | None = None,
                         reuse_weight: float = 0.0, batch: int = 1,
                         tau_override: dict | None = None,
                         measure_frac: float | None = None) -> list[str]:
    """Run the cascade for a ParsedQuery's semantic predicates. Assumes
    initialize_state(parsed_query) has already populated predicate_store +
    execution_state. Writes settled answers to execution_result and flushes
    objective tags; returns parent_asins where the logic tree evaluates TRUE.
    Does NOT cleanup() — the caller (octopimizer) owns the lifecycle.

    measure_frac (2026-08-03): Round-0 (learn_kill_rates) sample size as a
    FRACTION of THIS query's own post-structural-filter candidate count,
    resolved here (asins is only known after Phase 1 SQL filtering runs) --
    e.g. measure_frac=0.10 asks for roughly 10% of whatever this query's pool
    turns out to be, which differs per query (Q1~139, Q2/Q3~168). Overrides
    measure_size when set; None (default) keeps the historical absolute-count
    behavior byte-identical."""
    asins = sorted(get_candidate_asins())
    if limit is not None:
        asins = asins[:limit]
    if not asins:
        return []
    if measure_frac is not None:
        measure_size = max(1, round(measure_frac * len(asins)))

    pred_by_slug, slug_by_idx, predicate_type = _predicate_defs(parsed_query, subq_mode)
    idx_by_slug = {slug: idx for idx, slug in slug_by_idx.items()}
    tree = _tree_from_logic(parsed_query.logic, slug_by_idx)

    # reuse-aware AND ordering (2026-08-02): reuse_score is computed from PAST
    # queries only (this query's own usage is recorded AFTER, so it never sees
    # itself) -- see state_manager.get_reuse_scores / notes/grocery_reuse_ordering.md
    # §9. reuse_weight=0.0 (the default) makes reuse_score all-zeros functionally
    # irrelevant, so this is a no-op unless a caller opts in.
    reuse_score = get_reuse_scores(list(pred_by_slug), exclude_label=label)
    record_predicate_usage(list(pred_by_slug), label)

    # ── cross-query reuse: pull prior-query answers from the tag cache into the
    # freshly-rebuilt state, then hand them to the cascade as pre-seeded `known`
    # values. prepopulate_results_from_tags matches the tag column -> execution_result;
    # get_known_results reads it back; run_cascade pre-seeds the state so needed_leaves
    # skips these (predicate, product) pairs and short-circuits on them for free.
    # (Confidence-aware trust is deferred — we reuse the bool as-is; see §6.1.)
    n_pre, resume = prepopulate_results_from_tags(parsed_query)
    known = {
        a: {slug_by_idx[i]: (T if v else F) for i, v in d.items() if i in slug_by_idx}
        for a, d in get_known_results().items()
    }
    # Confidence recheck (todo ④a): low-confidence non-terminal cached answers were
    # NOT copied into `known` — instead their settling step comes back in `resume`,
    # and we translate it into a per-(product,predicate) route offset so run_cascade
    # RESUMES the escalation route at the NEXT step (k+1) rather than restarting from
    # tier 1. See notes/cache_confidence_recheck.md.
    start_step: dict[tuple[str, str], int] = {}
    for (asin, pred_idx), (s_tier, s_mod) in resume.items():
        slug = slug_by_idx.get(pred_idx)
        if slug is None:
            continue
        route = ROUTES[predicate_type[slug]]       # same lookup cfg.route() does
        pos = next((k for k, st in enumerate(route)
                    if st.tier == s_tier and st.modality == s_mod), None)
        if pos is None or pos + 1 >= len(route):
            continue                               # unknown / already terminal — let it run from step 0
        start_step[(asin, slug)] = pos + 1
    if n_pre:
        print(f"[cascade_planner] reused {n_pre} cached (product,predicate) from tags "
              f"({sum(len(v) for v in known.values())} known cells pre-seeded)")
    if resume:
        print(f"[cascade_planner] recheck: {len(resume)} low-confidence cached cell(s) "
              f"re-queued, {len(start_step)} resume-escalated at next route step")

    if client is None:
        from google import genai
        load_env()
        client = genai.Client(api_key=_require_key("GEMINI_API_KEY"))

    cfg = CascadeConfig(
        tiers=(1, 2, 3), thresholds=THRESHOLDS, predicate_type=predicate_type,
        routes=dict(ROUTES), chunk_size=chunk_size, measure_size=min(measure_size, len(asins)),
        direction_theta=direction_theta or {},
        reuse_score=reuse_score, reuse_weight=reuse_weight,
        batch=batch, workers=workers, seed=seed,
        tau_override=tau_override or {},
    )
    out_dir = out_dir or (REPO_ROOT / "results" / "cascade_system_run")
    # STREAMING input: no up-front load. The source pulls each batch's product text
    # from the DB on demand (meta_provider); we clear it per chunk so memory holds at
    # most one chunk's worth — "how much to fetch" follows the flow, robust to a
    # different chunk size / fetch strategy. The answer cache (small) persists, so
    # Round-0 tier-1 answers are still reused.
    src = LiveSource(client, {}, {}, cfg, out_dir, pred_by_slug=pred_by_slug,
                     meta_provider=lambda a: _build_metas(a, parsed_query.table_name))

    print(f"[cascade_planner] {len(asins)} candidates, "
          f"{len(pred_by_slug)} predicate(s): "
          + ", ".join(f"{s}({predicate_type[s]})" for s in slug_by_idx.values()))

    kill, m_cost, m_units = learn_kill_rates(tree, asins, src, cfg)
    print(f"[cascade_planner] Round 0: {m_units} units, ${m_cost:.4f}")

    trace_p = out_dir / "cascade_trace.csv"
    with trace_p.open("w", newline="") as f:            # header
        csv.DictWriter(f, fieldnames=_TRACE_COLS).writeheader()

    membership: dict = {}
    all_events: list = []
    seq = 0
    for ci, ch in enumerate(chunks_of(asins, cfg)):
        src.metas.clear()                       # drop last chunk's text; hold ≤1 chunk
        r = run_cascade(tree, ch, src, cfg, kill=kill, known=known, start_step=start_step)
        membership.update(r.membership)
        all_events.extend(r.events)
        _write_chunk_results(r.events, idx_by_slug)
        seq = _append_trace(trace_p, ci, seq, r.events, cfg)
        print(f"[cascade_planner] chunk {ci}: "
              f"{sum(v is True for v in r.membership.values())} matched, "
              f"${r.cost:.4f}")

    _short_circuit_report(label, tree, asins, known, all_events, out_dir)

    written = write_tag_cache(parsed_query)
    if written:
        print(f"[cascade_planner] flushed tags: {written}")
    print(f"[cascade_planner] total ${src.cost:.4f}, {src.calls} calls")

    return sorted(a for a, v in membership.items() if v is True)
