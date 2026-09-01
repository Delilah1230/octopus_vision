#!/usr/bin/env python3
"""
Cascade engine — AND/OR logic tree over tiered, confidence-gated predicates.

Implements notes/cascade_pipeline_plan.md §1.3. Replaces the flat-AND, k=2
hardcoding in cascade_sim.py (`all(x=="T")`, `preds[ci % 2]`).

Core ideas
----------
1. THREE-VALUED (Kleene) state per (product, predicate):
       T  = judged true  AND conf >= tau     (settled)
       F  = judged false AND conf >= tau     (settled)
       U  = conf < tau                       (uncertain -> escalate; U at the
                                              terminal tier => inconclusive)

2. SHORT-CIRCUIT = ABSORBING ELEMENT. One rule covers AND and OR:
       AND absorbs F,  OR absorbs T.
   A child that *confidently* takes the absorbing value settles the node
   immediately; the remaining children are never evaluated. U never absorbs.

3. ORDERING METRIC FLIPS WITH THE PARENT OPERATOR. `k` is not a property of a
   predicate, it is a property of a predicate *relative to its parent*:
       AND parent -> order children by  k_F / cost   (k_F = P(confident F))
       OR  parent -> order children by  k_T / cost   (k_T = P(confident T))
   Measured on the 200-product set: vegan k_F=17.5% / k_T=6.5%; indulgent
   k_F=50.0% / k_T=4.0%  =>  `vegan AND indulgent` runs indulgent first, but
   `vegan OR indulgent` runs vegan first. Same predicates, order reversed.

4. NEEDED LEAVES generalise short-circuit: a leaf is needed iff its value can
   still change the root's value. Any subtree settled by an absorbing value makes
   its unevaluated leaves not-needed — no special cases per operator.

5. TIER-BY-TIER over the whole tree: finish a tier across ALL needed leaves
   before going deeper, so a cheap tier on one leaf can short-circuit away an
   expensive tier on another.

The engine reads answers through an AnswerSource, so the identical logic runs
offline (replay a recorded table) or online (real API calls).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol

# ── three-valued logic ───────────────────────────────────────────────────────

T, F, U = "T", "F", "U"

ABSORB = {"AND": F, "OR": T}     # child value that settles the node outright
IDENT = {"AND": T, "OR": F}      # node value when no child absorbed and none are U


# ── query tree ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Leaf:
    predicate: str

    @property
    def leaves(self) -> tuple[str, ...]:
        return (self.predicate,)


@dataclass(frozen=True)
class Node:
    op: str                       # "AND" | "OR"
    children: tuple               # tuple[Leaf | Node, ...]

    def __post_init__(self):
        if self.op not in ABSORB:
            raise ValueError(f"op must be AND/OR, got {self.op!r}")
        if not self.children:
            raise ValueError("node needs children")

    @property
    def leaves(self) -> tuple[str, ...]:
        return tuple(p for c in self.children for p in c.leaves)


def parse_tree(spec, predicates: list[str] | None = None):
    """Build a tree from query_parser's JSON shape.

    Internal nodes: {"op": "AND"|"OR", "children": [...]}
    Leaves:         an int index into `predicates`, or a predicate name (str).
    A bare leaf spec is a valid tree (single-predicate query).
    """
    if isinstance(spec, (Leaf, Node)):
        return spec
    if isinstance(spec, int):
        if predicates is None:
            raise ValueError("integer leaf needs `predicates`")
        return Leaf(predicates[spec])
    if isinstance(spec, str):
        return Leaf(spec)
    if isinstance(spec, dict) and "op" in spec:
        return Node(spec["op"].upper(),
                    tuple(parse_tree(c, predicates) for c in spec["children"]))
    raise ValueError(f"cannot parse tree spec: {spec!r}")


# ── evaluation (Kleene + absorbing-element short-circuit) ────────────────────

def eval_tree(node, state: dict[str, str]) -> str:
    """Value of `node` given per-predicate states. Pure Kleene; no side effects."""
    if isinstance(node, Leaf):
        return state.get(node.predicate, U)
    absorb, ident = ABSORB[node.op], IDENT[node.op]
    saw_u = False
    for child in node.children:
        v = eval_tree(child, state)
        if v == absorb:
            return absorb                      # short-circuit
        if v == U:
            saw_u = True
    return U if saw_u else ident


def needed_leaves(node, state: dict[str, str]) -> set[str]:
    """Leaves whose value can still change the root's value.

    Generalises short-circuit: a subtree already settled by its absorbing value
    contributes nothing needed; and once a node is settled, none of its leaves
    are needed. This one function subsumes AND-short-circuit, OR-short-circuit
    and upward propagation.
    """
    if eval_tree(node, state) != U:
        return set()                           # settled -> nothing here matters
    if isinstance(node, Leaf):
        return {node.predicate} if state.get(node.predicate, U) == U else set()
    out: set[str] = set()
    for child in node.children:
        out |= needed_leaves(child, state)
    return out


# ── answer source ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Answer:
    direction: str        # "T" | "F"  (read off content_score ≷ 0.5)
    confidence: float
    cost: float = 0.0
    latency: float = 0.0


class AnswerSource(Protocol):
    """Where (product, predicate, tier) answers come from.

    Offline: look them up in the recorded table (free, instant).
    Online:  call the model (costs money). Identical engine either way.

    `get_many` is the primitive, not `get`: the engine always asks for a whole
    (predicate, tier) × products group at once, which is exactly the unit that
    batching (same model + same predicate, only product data differs) and
    worker-parallelism can exploit. A single get is just get_many of one.
    """

    def get_many(self, asins: list[str], predicate: str, tier: int,
                 modality: str = "text", force_commit: bool = False) -> list[Answer]: ...


# ── escalation route: (tier, modality) steps ────────────────────────────────
# The escalation unit is no longer just a tier — it is a (tier, modality) STEP.
# Adding a modality (text -> text+image) and bumping a tier are the SAME kind of
# move (spend more when uncertain) but fix different failure modes, and which one
# pays is PER-PREDICATE-TYPE (measured 2026-07-17 — see
# notes/cascade_escalation_route.md §1). So the ladder is a fixed per-type route:
#   * visual/subjective predicates (indulgent): +image is cheap AND more accurate
#     than a tier bump -> try image early, before spending on a bigger model.
#   * textual/objective predicates (vegan): the signal IS in the text -> bump the
#     model first; image is a cheap side-step that helps little.
# These routes are a hand-picked linear fallback ordered by expected value/cost;
# they intentionally do NOT obey a monotone lattice (the objective route drops
# from (2,txt) back to (1,txt+img)). The principled DAG/optimal-stopping version
# is route §3 — not built yet.

TEXT, TEXT_IMAGE = "text", "text_image"


@dataclass(frozen=True)
class Step:
    tier: int
    modality: str = TEXT

    @property
    def label(self) -> str:
        return f"t{self.tier}·{'txt' if self.modality == TEXT else 'txt+img'}"


# predicate_type -> ordered escalation route (notes/cascade_escalation_route.md §2)
ROUTES: dict[str, tuple[Step, ...]] = {
    "objective": (
        Step(1, TEXT), Step(2, TEXT), Step(1, TEXT_IMAGE),
        Step(2, TEXT_IMAGE), Step(3, TEXT), Step(3, TEXT_IMAGE),
    ),
    "subjective": (
        Step(1, TEXT), Step(1, TEXT_IMAGE), Step(2, TEXT),
        Step(2, TEXT_IMAGE), Step(3, TEXT), Step(3, TEXT_IMAGE),
    ),
}


# ── config ───────────────────────────────────────────────────────────────────

@dataclass
class CascadeConfig:
    # -- tiers & gate --
    tiers: tuple[int, ...] = (1, 2, 3)
    # thresholds[predicate_type][tier] -> tau
    thresholds: dict = field(default_factory=dict)
    predicate_type: dict = field(default_factory=dict)   # predicate -> type
    # predicate_type -> escalation route (tuple[Step]); default = ROUTES above.
    routes: dict = field(default_factory=lambda: dict(ROUTES))
    # direction_theta[predicate][tier] -> theta ("direction = T if content > theta").
    # Independent of tau: theta decides WHAT the answer is once a tier is trusted,
    # tau decides WHETHER a tier is trusted at all. Missing predicate or tier falls
    # back to 0.5, the historical hardcoded constant.
    #
    # Keyed by PREDICATE, not predicate_type -- notes/grocery_baseline_eval.md §4d-7
    # calibrated all three subjective predicates individually and found this is NOT
    # a type-level property: christmas_dinner benefits from theta=0.67 on tier1
    # (0.642->0.719), but the SAME override on indulgent_valentine collapses it
    # (0.667->0.273) and does nothing for kids_suitable. Grouping by type (as an
    # earlier version of this field did) silently drags untested predicates along
    # for the ride -- see §4d-6, where exactly that mistake cost Q1/Q3 a few points.
    direction_theta: dict = field(default_factory=dict)

    # predicate -> fraction of past distinct queries (this session) that also
    # needed it, in [0,1]. From state_manager.get_reuse_scores(). Empty dict (the
    # default) = no workload history available or reuse-awareness disabled --
    # leaf_order() falls back to pure kill_rate/cost, byte-identical to before this
    # field existed. See reuse_weight below and notes/grocery_reuse_ordering.md §9.
    reuse_score: dict = field(default_factory=dict)
    # how much to let reuse_score bend the AND/OR ordering away from this query's
    # own kill_rate/cost optimum, in order to build cross-query cache inventory for
    # commonly-needed predicates. 0.0 (default) = off, ordering is exactly the old
    # kill_rate/cost-only behavior. Same [0,1]-ish scale as kill_rate/cost, so 1.0
    # weights the two roughly equally; there is no principled default beyond that,
    # it trades this query's own cost for the workload's -- see §9 for the measured
    # trade at a few settings.
    reuse_weight: float = 0.0

    # tau_override[predicate][tier] -> confidence threshold to settle at that tier,
    # overriding the type-grouped `thresholds` default. Same rationale as
    # direction_theta above: keyed by PREDICATE not predicate_type, because
    # 2026-08-02's type-grouped theta mistake (grocery_baseline_eval.md §4d-6)
    # showed a type-level override silently drags untested siblings along. Missing
    # predicate or tier falls back to thresholds[predicate_type][tier].
    tau_override: dict = field(default_factory=dict)

    # -- granularity (NOT an optimisation: chunking is cost-neutral, plan §7.0) --
    chunk_size: int = 100          # execution granularity: streaming output, memory
    measure_size: int = 200        # bootstrap sample; statistics, not a fraction.
                                   # n≈100/predicate already gives ~5σ on ordering.

    # -- execution knobs (affect cost/latency, not the routing logic) --
    # PRIORITY: workers first, batch second — see effective_batch().
    batch: int = 1                 # CAP on products per prompt (same model +
                                   # same predicate; only product data differs).
                                   # A cap, not a fixed size: the last group may
                                   # be short, and we batch only as much as the
                                   # workers cannot already absorb.
    workers: int = 8               # concurrency; independent of #products/#predicates.
                                   # Raise this before touching `batch`: decode
                                   # parallelises across requests, never within one.
                                   # Ceiling is the provider's RPM/TPM, not CPU.

    seed: int = 42                 # chunk shuffle: chunks must be random samples

    def tau(self, predicate: str, tier: int) -> float:
        if predicate in self.tau_override and tier in self.tau_override[predicate]:
            return self.tau_override[predicate][tier]
        return self.thresholds[self.predicate_type[predicate]][tier]

    def theta(self, predicate: str, tier: int) -> float:
        return self.direction_theta.get(predicate, {}).get(tier, 0.5)

    def route(self, predicate: str) -> tuple:
        """The (tier, modality) escalation sequence for this predicate's type."""
        return self.routes[self.predicate_type[predicate]]


# ── the cascade ──────────────────────────────────────────────────────────────

def effective_batch(n_items: int, cfg: "CascadeConfig") -> int:
    """How many products to put in ONE prompt — workers first, batch second.

    `cfg.batch` is a CAP (at most N per prompt), not a fixed size.

    Why workers win: an LLM call is prefill + decode. Prefill (the prompt) is
    processed in parallel, but decode is autoregressive — token by token. Putting
    5 products in one prompt makes ONE model instance decode 5x the tokens
    serially; sending 5 requests gives 5 independent decode streams that overlap.
    Measured on tier 1: latency(B) = 0.28 + 0.42*B, so
        5 products, 1 batched call  -> 2.38s
        5 products, 5 workers       -> 0.70s      (3.4x faster)
    Batching can only ever amortise the 0.28s prefill (=40% of a call); decode
    never amortises. Workers overlap both.

    So batching only pays once the workers are already saturated — and it is not
    free: batched answers measurably drift (~27% of sub-answers shifted at B=3,
    against a tier-1 noise floor of 0). Hence: use the SMALLEST batch that still
    keeps every worker busy.
    """
    if cfg.batch <= 1 or n_items <= cfg.workers:
        return 1                       # workers can absorb it all -> never batch
    return min(cfg.batch, math.ceil(n_items / cfg.workers))


@dataclass
class LeafOutcome:
    """One evaluated (product, predicate, tier). Everything needed to audit the
    decision afterwards: the raw answer (direction/confidence), the gate it was
    judged against (tau), the resulting three-valued state, and what it cost."""
    asin: str
    predicate: str
    tier: int
    modality: str         # text / text_image — which modality this step used
    direction: str        # T / F  — the model's call, before the gate
    confidence: float
    tau: float            # the threshold it had to clear at this tier
    value: str            # T / F / U — after the gate
    cost: float
    latency: float


@dataclass
class CascadeResult:
    membership: dict          # asin -> True / False / None(inconclusive)
    kill_rate: dict           # (predicate, absorbing_value) -> learned rate
    order_by_node: dict       # id(node) -> ordered child list (for reporting)
    events: list              # LeafOutcome-ish log
    cost: float
    calls: int


def _settled(ans: Answer, tau: float, force: bool = False) -> str:
    """Gate: a direction only settles if it clears the tier's threshold — EXCEPT at
    the terminal oracle step (force=True), where there is nowhere left to escalate,
    so we commit to the direction regardless of confidence (U retires). The low
    confidence still travels with the answer as the 'this was a guess' flag.
    See notes/cascade_escalation_route.md §2.1."""
    if force:
        return ans.direction
    return ans.direction if ans.confidence >= tau else U


def learn_kill_rates(tree, asins: Iterable[str], src: AnswerSource,
                     cfg: CascadeConfig) -> tuple[dict, float, int]:
    """ROUND 0 — measurement, NO short-circuit.

    Run every leaf predicate unconditionally at tier 1 on the measurement sample,
    so each k is an unbiased marginal. No short-circuit here by construction:
    short-circuit needs an order, and the order is what we are measuring.

    Returns ({(predicate, absorb) -> k}, cost, n_calls). Both k_F and k_T are
    computed because the ordering metric depends on the parent operator (§1.3).

    2026-08-10: step 0 is the terminal rung whenever a predicate's route has only
    ONE step (the single-stage configuration). It must then be called with
    force_commit=True, exactly as run_cascade would call it. Without this the
    sample's answers are cached under a DIFFERENT prompt (no "final verdict
    required" note) and then reused by the main pass, so the measured products
    would be judged under weaker instructions than every other product in the
    same run. The cache key is (asin, predicate, tier, modality) and does not
    include force_commit, so the divergence is silent.
    """
    import random
    preds = list(dict.fromkeys(tree.leaves))
    # Round 0 exists to ORDER leaves against each other. With a single leaf there
    # is no order to learn and no sibling to short-circuit, so the measurement is
    # pure ceremony: skip it. (Its answers were cached and reused by the main
    # pass, so this does not change cost -- but it stops the trace and the
    # per-chunk cost report from attributing units to a phase that decided
    # nothing.) Any query with >= 2 predicates measures as before.
    if len(preds) < 2:
        return {(p, v): 0.0 for p in preds for v in (T, F)}, 0.0, 0
    # MUST be a random sample, not the first N: k is only an unbiased marginal if
    # the sample is unbiased, and a real scan arrives in physical (key/category)
    # order. Costs nothing to shuffle; silently biases k if you don't.
    order = list(asins)
    random.Random(cfg.seed).shuffle(order)
    sample = order[: cfg.measure_size]
    hits = {(p, v): 0 for p in preds for v in (T, F)}
    cost = 0.0
    calls = 0
    for p in preds:
        route = cfg.route(p)
        step0 = route[0]                                   # cheapest step: (1, text)
        is_terminal = len(route) == 1                      # single-stage route
        tau = cfg.tau(p, step0.tier)
        answers = src.get_many(sample, p, step0.tier, step0.modality,
                               force_commit=is_terminal)   # batched + parallel
        for ans in answers:
            cost += ans.cost
            calls += 1
            v = _settled(ans, tau, force=is_terminal)
            if v in (T, F):
                hits[(p, v)] += 1
    n = max(len(sample), 1)
    return {k: c / n for k, c in hits.items()}, cost, calls


def order_children(node: Node, kill: dict, cost_of: Callable[[str], float]):
    """Order a node's children by kill-rate-per-cost, w.r.t. THIS node's operator.

    AND wants children that confidently go F; OR wants children that confidently
    go T. A subtree's kill rate is recursive under the independence assumption
    (plan §1.2 P3): to drive `A OR B` to F you need BOTH A and B to be F.
    """
    absorb = ABSORB[node.op]

    def k_of(child) -> float:
        if isinstance(child, Leaf):
            return kill.get((child.predicate, absorb), 0.0)
        # subtree: it takes `absorb` only if all ITS children do (when its own
        # operator's identity is `absorb`), else it is a weak filter -> ~0.
        if IDENT[child.op] == absorb:
            return math.prod(k_of(c) for c in child.children)
        return 0.0

    def c_of(child) -> float:
        return sum(cost_of(p) for p in child.leaves) or 1e-9

    return sorted(node.children, key=lambda c: -k_of(c) / c_of(c))


def leaf_order(tree, kill: dict, cost_of: Callable[[str], float] | None = None,
               reuse_score: dict | None = None, reuse_weight: float = 0.0
               ) -> list[str]:
    """Leaves ordered by k/cost + reuse_weight*reuse_score, where k is taken w.r.t.
    each leaf's PARENT operator (AND -> k_F, OR -> k_T). See §1.3(3).

    reuse_score/reuse_weight (2026-08-02, see notes/grocery_reuse_ordering.md §9):
    kill_rate/cost alone optimises only THIS query's own cost, which tends to push
    a commonly-shared predicate (e.g. vegan) LATER in the order whenever it happens
    to be less selective than its sibling on this particular candidate pool -- it
    then gets short-circuited away for most candidates, building almost no
    cross-query cache inventory for the NEXT query that needs it (measured: Q1's
    vegan settled only 21/139 because indulgent_valentine went first and killed 90).
    reuse_score(p) in [0,1] (fraction of past queries that also needed p) lets a
    frequently-reused predicate outbid a higher-kill_rate sibling for the FIRST
    slot, trading some of this query's own short-circuit savings for a richer
    inventory the whole workload benefits from. reuse_weight=0.0 (default) is
    byte-identical to the pre-existing kill_rate/cost-only behavior.
    """
    cost_of = cost_of or (lambda _p: 1.0)   # per-predicate cost ~constant (§1.2 P2)
    reuse_score = reuse_score or {}
    parent_op: dict[str, str] = {}

    def index(n, op="AND"):
        if isinstance(n, Leaf):
            parent_op[n.predicate] = op
        else:
            for c in n.children:
                index(c, n.op)

    index(tree)

    def key(p: str) -> float:
        local = kill.get((p, ABSORB[parent_op[p]]), 0.0) / cost_of(p)
        return -(local + reuse_weight * reuse_score.get(p, 0.0))

    return sorted(dict.fromkeys(tree.leaves), key=key)


def run_cascade(tree, asins: list[str], src: AnswerSource, cfg: CascadeConfig,
                kill: dict | None = None,
                known: dict[str, dict[str, str]] | None = None,
                start_step: dict[tuple[str, str], int] | None = None) -> CascadeResult:
    """Evaluate `tree` for a chunk of products, STEP-BY-STEP ACROSS PRODUCTS.

    `known` = {asin: {predicate: "T"|"F"}} pre-seeds the state with values already
    known from a PRIOR query (the tag cache, via state_manager). A pre-seeded leaf
    is already settled, so needed_leaves never marks it needed -> it is never
    evaluated (no event, no cost) and it drives cross-predicate short-circuit
    exactly like a freshly-computed value. This is how cross-query reuse and
    intra-query short-circuit become the SAME mechanism (no special-casing): a
    cached vegan=F on `vegan AND christmas` short-circuits christmas away; a cached
    vegan=T leaves only christmas to evaluate. See notes/cascade_escalation_route.md.

    `start_step` = {(asin, leaf): k} makes a leaf RESUME its route at step k instead
    of step 0 (todo ④a confidence recheck): a low-confidence cached answer is left
    pending (not in `known`) and re-evaluated only from the rung past where it settled,
    so we resume-escalate rather than re-run from tier 1. See
    notes/cache_confidence_recheck.md.

    A "step" is a (tier, modality) rung on each predicate's escalation route
    (cfg.route(p)); step index `s` is the s-th escalation level. The loop is
    (step, predicate) x products — NOT product-by-product. That ordering is what
    makes the two execution knobs meaningful: every group handed to `get_many` is
    one predicate at one (tier, modality) over many products, which is exactly
    what batching (same model + same predicate, only product data differs) and
    worker-parallelism can exploit. Product-by-product would leave both knobs dead.

    It is also the faithful reading of §1.3(7), generalised from tier to step:
    finish an escalation level across ALL needed leaves — and all products —
    before going deeper, so a cheap step on one leaf can short-circuit away an
    expensive step on another. Because different predicate types have different
    routes, at one step index two predicates may run different (tier, modality)
    rungs (e.g. objective's step 1 is (2,txt) while subjective's is (1,txt+img)).
    """
    kill = kill if kill is not None else {}
    start_step = start_step or {}              # {(asin, leaf): first route step index to run}
    events: list[LeafOutcome] = []
    total_cost = 0.0
    calls = 0

    state: dict[str, dict[str, str]] = {a: {p: U for p in tree.leaves} for a in asins}
    if known:                                  # pre-seed cached (prior-query) values
        for a in asins:
            for p, v in known.get(a, {}).items():
                if p in state[a] and v in (T, F):
                    state[a][p] = v
    ordered = leaf_order(tree, kill, reuse_score=cfg.reuse_score,
                         reuse_weight=cfg.reuse_weight)

    n_steps = max((len(cfg.route(p)) for p in ordered), default=0)
    for step_idx in range(n_steps):
        for p in ordered:
            route = cfg.route(p)
            if step_idx >= len(route):
                continue                       # this predicate's route is shorter
            step = route[step_idx]
            # the LAST rung of the route is the oracle: no escalation left, so it
            # force-commits to T/F (never U). For the full live route that is
            # (3, text_image); for an offline-trimmed route it is whatever rung is
            # last. Only this rung force-commits — earlier tier-3 rungs may still U.
            is_terminal = step_idx == len(route) - 1
            # every product in this chunk whose answer for p can still move the root.
            # `start_step` (todo ④a confidence recheck) holds a leaf back until its
            # RESUME step: a low-confidence cached answer stays U here and is only
            # evaluated once we reach step k+1 (the rung past where it originally
            # settled), so we resume-escalate instead of re-running from tier 1. It can
            # still be short-circuited away for free by a sibling before that step.
            todo = [a for a in asins
                    if p in needed_leaves(tree, state[a])
                    and step_idx >= start_step.get((a, p), 0)]
            if not todo:
                continue                       # fully short-circuited at this leaf
            answers = src.get_many(todo, p, step.tier, step.modality,
                                   force_commit=is_terminal)  # <- batch+workers here
            tau = cfg.tau(p, step.tier)
            for a, ans in zip(todo, answers):
                total_cost += ans.cost
                calls += 1
                state[a][p] = _settled(ans, tau, force=is_terminal)
                events.append(LeafOutcome(
                    asin=a, predicate=p, tier=step.tier, modality=step.modality,
                    direction=ans.direction, confidence=ans.confidence, tau=tau,
                    value=state[a][p], cost=ans.cost, latency=ans.latency))
        if all(eval_tree(tree, state[a]) != U for a in asins):
            break                              # whole chunk settled

    membership = {}
    for a in asins:
        v = eval_tree(tree, state[a])
        membership[a] = True if v == T else False if v == F else None

    return CascadeResult(membership=membership, kill_rate=kill, order_by_node={},
                         events=events, cost=total_cost, calls=calls)


def chunks_of(asins: list[str], cfg: CascadeConfig) -> list[list[str]]:
    """Split into execution chunks. Chunks MUST be random samples (unbiased k);
    chunk_size is a granularity/streaming knob and is cost-neutral (§7.0)."""
    import random
    order = list(asins)
    random.Random(cfg.seed).shuffle(order)
    return [order[i:i + cfg.chunk_size] for i in range(0, len(order), cfg.chunk_size)]
