#!/usr/bin/env python3
"""
SINGLE SOURCE OF TRUTH for the vision-paper movie protocol.

Every runner in this folder imports its configuration from here, so "what
configuration was this number produced under" has exactly one answer.

THE PROTOCOL (2026-08-10)
-------------------------
1. ONE model for every system: gemini-2.5-flash, text only. No flash-lite, no
   pro, no image modality anywhere. This is what makes the head-to-head about
   the SYSTEM rather than about model routing.
2. LOTUS and Palimpzest run at UPSTREAM SemBench settings, not at our
   matched-cell settings. See LOTUS_UPSTREAM / PZ_UPSTREAM below for the exact
   provenance of every knob.
3. Our cascade is collapsed to a SINGLE stage: Step(2, "text") = gemini-2.5-flash
   on text. There is no escalation, so there is no cascade contribution in these
   numbers -- deliberate, per notes/2026-08-10.md §1.2 (cascade demoted from a
   claim to a replaceable optimization) and §3.2 (no cascade in the head-to-head).
4. reasoning_effort / thinking is OFF for every system (SemBench's own protocol,
   validated by their own ablation: thinking cost 17.9x for slightly WORSE F1 --
   notes/cidr/baseline_reasoning_config.md §4.2).

WHAT THIS FOLDER DOES *NOT* DO
------------------------------
It does not reproduce SemBench's published Table 4. We re-run LOTUS and PZ
ourselves so that all three systems see byte-identical input rows (see DEDUP
below) and so the workload-curve experiment can replay the same queries in a
sequence. Our numbers and their Table 4 are therefore NOT interchangeable.

DATA: THE BENCHMARK AS PUBLISHED
--------------------------------
We run sf_2000 exactly as SemBench ships it -- 2000 review rows, including the
135 that are literal duplicates of each other (ant_man is 256 rows = 128 reviews
x 2). Ground truth comes from the same file.

This is what makes our numbers directly comparable to the paper's own table for
LOTUS, Palimpzest, ThalamusDB and BigQuery: same rows, same ground truth, no
adjustment factors and no claim about anyone else's configuration.

A deduplicated copy exists (data/movie_sf2000_dedup, --dedup on any runner) and
was used for the 2026-08-10/11 rounds. Duplicates cost a per-ROW system ~13% and
a per-PAIR system up to 4x, so dedup is not neutral between systems -- another
reason to just use the published data and not have to argue about it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"

# ── data ──────────────────────────────────────────────────────────────────────
DATA_DIR = REPO / "cidr_evaluation" / "sembench" / "files" / "movie" / "data" / "sf_2000"
DEDUP_DATA_DIR = REPO / "vision_evaluation" / "data" / "movie_sf2000_dedup"
RESULTS = REPO / "vision_evaluation" / "results" / "movie"

TAKEN3 = "taken_3"
ANTMAN = "ant_man_and_the_wasp_quantumania"

# ── the one model ─────────────────────────────────────────────────────────────
MODEL = "gemini-2.5-flash"
LOTUS_MODEL = f"gemini/{MODEL}"          # litellm provider prefix
WORKERS = 20                             # SemBench's concurrent_llm_worker

# The two places a BETTER model is affordable, because neither cost grows with
# the data (notes/2026-08-10.md §1.4):
#   Query Parser        -> per QUERY
#   Related questions   -> per PREDICATE, offline, once
# The ANSWERING model stays MODEL above for all three systems -- that is what
# makes the head-to-head about the system rather than about model choice.
# gemini-3.6-flash is $1.50/$7.50 per 1M vs flash's $0.30/$2.50.
PARSER_MODEL = "gemini-3.6-flash"
GENERATOR_MODEL = "gemini-3.6-flash"
# Thinking bills at the output rate, and output is where the parser's cost lives
# (857-1921 output tokens per parse, mostly thought). Measured 2026-08-11:
# "minimal" cuts the parser bill 64% ($0.093 -> $0.034 over 8 parses) and every
# one of the 8 parses returned the identical predicates and structural SQL.
# gemini-3.x rejects thinking_budget=0 -- thinking_level is the 3.x knob.
PARSER_THINKING = "minimal"

# How the 6 related answers come back, and how many rows share a prompt.
# Measured 2026-08-11 (bench_per_row.py, 300 rows, vs a verbose-vs-verbose control):
#   verbose, batch=1 : $0.443/1k rows, 58.7 ms/row   (502 prompt + 117 out tok)
#   array,   batch=6 : $0.141/1k rows, 25.9 ms/row   (157 prompt + 37 out tok)
#   quality F1 0.675 vs 0.648 -- within each config's own run-to-run spread
# The two knobs are NOT independent: output tokens decode serially, so batching a
# VERBOSE payload is slower (55.9 -> 61.2 ms/row) while batching an ARRAY payload
# is faster (41 -> 26). Compress the output first, then batch.
#
# "compact" (one string "TFT?FT") is NOT used: it cut output to 8 tok but dropped
# F1 0.639 -> 0.524, seven times the run-to-run noise. Naming each answer's
# question is part of the method, not packaging.
ANSWER_FORMAT = "array"
BATCH = 6
# effective_batch() is "workers first, batch second": it uses the SMALLEST batch
# that still keeps all 20 workers busy, so BATCH is only a cap. At chunk_size=50
# it resolves to 3, not 6. The chunk must be large enough for the cap to bind --
# 2000 covers the whole movie table in one wave, as the benchmark did.
CHUNK_SIZE = 2000

# SemBench's pricing table, per 1M tokens
# (src/runner/generic_lotus_runner/generic_lotus_runner.py::PRICING)
PRICING = {
    "gemini-2.5-flash":      {"text": 0.3,  "output": 2.5},
    "gemini-2.5-flash-lite": {"text": 0.1,  "output": 0.4},
    "gemini-2.5-pro":        {"text": 1.25, "output": 10.0},
}

# ── upstream settings, quoted so drift is visible in a diff ───────────────────
# src/runner/generic_lotus_runner/generic_lotus_runner.py
LOTUS_UPSTREAM = {
    "model_name": "gemini-2.5-flash",
    "max_tokens": 8192,                  # raised from LOTUS's own 512 default
    "max_batch_size": WORKERS,           # = concurrent_llm_worker
    "reasoning_effort": "disable",       # _configure_lm(), the gemini-2.5-flash branch
    "strategy": None,                    # sem_filter is called bare -> no written CoT
    # GenericLotusRunner.__init__ default. The movie runner never overrides it, so
    # Q5/Q6/Q7 sem_join DOES get cascade_args -- see APPROX_CASCADE below.
    "policy": "approximate",
    "ranking": "map",                    # sem_map (not sem_topk) for Q9/Q10
}
# src/scenario/movie/runner/lotus_runner/lotus_runner.py::__init__
APPROX_CASCADE = {"recall_target": 0.8, "precision_target": 0.8}
APPROX_RM = "intfloat/e5-base-v2"        # SentenceTransformersRM + FaissVS

# config/system/palimpzest/gemini-2.5-flash-maxquality.json
PZ_UPSTREAM = {
    "policy": "MaxQuality",
    "execution_strategy": "parallel",
    "max_workers": WORKERS,
    "join_parallelism": WORKERS,
    "verbose": False,
    "progress": True,
    "reasoning_effort": None,            # NOT passed to QueryProcessorConfig at all
    "available_models": ["GEMINI_2_5_FLASH"],
}


def load_env() -> None:
    """Populate GEMINI_API_KEY etc. from the repo .env without clobbering the shell."""
    envf = REPO / ".env"
    if not envf.exists():
        return
    for line in envf.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def require_key() -> str:
    load_env()
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY not set (looked in Octopus/.env)")
    return key


def data_dir(dedup: bool = False) -> Path:
    d = DEDUP_DATA_DIR if dedup else DATA_DIR
    if not (d / "Reviews.csv").exists():
        sys.exit(f"missing {d/'Reviews.csv'} — run vision_evaluation/movie/prepare_data.py")
    return d


def out_dir(system: str, suffix: str = "") -> Path:
    d = RESULTS / f"{system}{suffix}"
    d.mkdir(parents=True, exist_ok=True)
    return d


QUERIES_JSON = Path(__file__).resolve().parent / "queries.json"


def load_queries() -> dict:
    """The movie queries, keyed by id. Single source of truth for NL text, gold
    SQL, metric kind and limit -- these used to be duplicated across the runner,
    the scorer and each baseline, which is how a benchmark drifts without anyone
    noticing. Placeholders {t3}/{am} are already substituted."""
    import json
    spec = json.loads(QUERIES_JSON.read_text())
    ids = spec["movies"]
    out = {}
    for q in spec["queries"]:
        q = dict(q)
        q["gold_sql"] = q["gold_sql"].format(**ids)
        out[q["id"]] = q
    return out


PREDICATE_REGISTRY = REPO / "vision_evaluation" / "predicates"


def load_predicate_registry(required: tuple[str, ...] | None = None) -> None:
    """Install the OFFLINE-GENERATED related questions over the hand-written ones.

    The hand-written 5 in subquestion_probe.PREDICATES were written in 2026-07-29
    *after* seeing that the generated ones scored worse on this very benchmark.
    Keeping them would be tuning against the test set and the paper could not
    describe how they came to exist. The registry files are produced once, offline,
    by a strong model (vision_evaluation/generate_predicate_questions.py) and carry
    their own provenance.

    The entry is mutated IN PLACE rather than replaced: cascade_agent and
    cascade_planner both build slug/NL indexes from PREDICATES at import time and
    hold references to these dicts, so mutating keeps every index consistent
    without needing to rebuild any of them.
    """
    import json

    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import subquestion_probe as _sq

    # Load EVERY file in the registry, not a hardcoded list: a query's predicate
    # wording is not known until the parser runs, and a registry entry that
    # exists but was not loaded fails exactly like one that does not exist.
    if required is None:
        required = tuple(sorted(f.stem for f in PREDICATE_REGISTRY.glob("*.json")))
        if not required:
            sys.exit(f"no predicate definitions in {PREDICATE_REGISTRY}")

    by_slug = {p["slug"]: p for p in _sq.PREDICATES}
    for slug in required:
        f = PREDICATE_REGISTRY / f"{slug}.json"
        if not f.exists():
            sys.exit(
                f"missing {f}\nGenerate it once (per predicate, offline):\n"
                f"  python vision_evaluation/generate_predicate_questions.py "
                f"--slug {slug}"
            )
        spec = json.loads(f.read_text())
        subqs = [tuple(x) for x in spec["subquestions"]]
        if spec.get("scheme") != "related" or spec["n"] % 2:
            sys.exit(f"{f}: expected an even-n 'related' scheme, got "
                     f"{spec.get('scheme')!r} n={spec.get('n')}")

        # ALIASES exist because the parser is an LLM and its predicate WORDING is
        # not stable across runs: the same Q1 text produced "is clearly positive"
        # on one parse and "is a clearly positive review" on the next. Each
        # wording canonicalises to a different slug, so without aliases the
        # registry lookup misses and the predicate silently falls through to
        # runtime generation -- i.e. back to the 3-content+2-meta scheme this
        # protocol replaced, with nothing in the log to say so.
        #
        # An alias gets its OWN slug and therefore its OWN tag column. It does NOT
        # assert that the two wordings are the same predicate: per §3.2, wrongly
        # asserting a relation corrupts answers while missing one only costs
        # money, so aliases buy reproducibility without buying that risk.
        for name, key in [(slug, spec["key"])] + [(al, spec["key"])
                                                  for al in spec.get("aliases", [])]:
            if name in by_slug:
                by_slug[name]["subquestions"] = subqs
                by_slug[name]["key"] = key
            else:
                entry = {"key": key, "slug": name, "subquestions": subqs}
                _sq.PREDICATES.append(entry)
                by_slug[name] = entry
        extra = spec.get("aliases", [])
        print(f"[protocol] {slug}: {len(subqs)} related questions "
              f"(generated {spec['generated_at']} by {spec['generated_by']})"
              + (f" + aliases {extra}" if extra else ""))


# ── our system: collapse the cascade to one stage ─────────────────────────────

def configure_octopus() -> None:
    """Put Octopus into the single-stage movie protocol. Call BEFORE parsing or
    executing anything -- it rebinds module-level configuration.

    Four things happen, and each one is load-bearing:

    1. ROUTES for BOTH predicate types become the one-element route
       (Step(2, "text")) = gemini-2.5-flash on text. With one rung, run_cascade's
       `is_terminal` is true on step 0, so every cell force-commits to T/F and
       tau never gates anything: no escalation, no U at the end. That is what
       "cascade reduced to a single stage" means operationally.

       This also removes the predicate_type dependence entirely, which quietly
       fixes a reproducibility problem noted in sembench_movie_eval.md §8: the
       same sentiment predicate was classified objective on some queries and
       subjective on others, and the two types had DIFFERENT escalation routes.
       One route for both types means the classification can no longer change
       what gets run.

    2. TERMINAL_STEP / RECHECK_TAU are moved to (2,"text") / {2: 0.0}. Without
       this the cross-query reuse gate keeps believing the route ends at
       (3, text_image) and re-queues flash-settled cells looking for a tier-3
       rung that no longer exists -- i.e. reuse would silently stop working.
       tau 0.0 = "a settled cell is trusted", which is the only coherent policy
       when there is nowhere to escalate.

    3. cascade_live.set_reasoning(thinking="off") is what actually turns thinking
       off. Setting subquestion_probe.THINKING_BUDGET = 0 (what the old movie
       runners did) has been a NO-OP since the 2026-08-03 LLMConfig refactor:
       LiveSource passes llm.thinking_budget explicitly, and the module global is
       only consulted for the _UNSET sentinel. We set both -- the global for any
       code path that still reads it, set_reasoning for the cascade path.

    4. CoT off, matching LOTUS (bare sem_filter prompt) and PZ
       (COT_BOOL_NO_REASONING).
    """
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))

    import cascade
    from cascade import Step

    single = (Step(2, "text"),)
    cascade.ROUTES["objective"] = single
    cascade.ROUTES["subjective"] = single

    import query_parser as qp
    qp.RECHECK_TAU = {2: 0.0}
    qp.DEFAULT_MODEL = PARSER_MODEL
    qp.PARSER_THINKING_LEVEL = PARSER_THINKING
    qp.TERMINAL_STEP = (2, "text")

    import subquestion_probe as _sq
    _sq.THINKING_BUDGET = 0
    _sq.COT = False
    _sq.ANSWER_FORMAT = ANSWER_FORMAT
    # The answering instructions were written for Amazon listings and say
    # "product" throughout. Telling the model a film review is a product is the
    # same domain leakage that broke the first movie decomposition, so give the
    # table its own noun. Default stays "product", so grocery is untouched.
    _sq.RECORD_NOUN = "review"

    import cascade_live as _cl
    _cl.set_reasoning(thinking="off", cot=False)

    # 5. Related questions from the offline registry, and the §1.3 scorer that
    #    goes with them: r = fraction answering TRUE, c = max(r, 1-r), an exact
    #    tie is U. The scorer is selected by the predicate's own shape
    #    (cascade_live.is_related_scheme), so grocery's 3-content+2-meta
    #    predicates keep the old scoring and nothing migrates in lockstep.
    load_predicate_registry()
    forbid_runtime_generation()
    for p in _sq.PREDICATES:
        if p["slug"] == "is_a_positive_review":
            assert _cl.is_related_scheme(p), "movie predicate is not on the §1.3 scheme"

    # Sanity: the only model any Octopus call may reach is gemini-2.5-flash.
    used = {_cl.TIER_MODEL[s.tier] for s in single}
    assert used == {MODEL}, f"single-stage route must use only {MODEL}, got {used}"
    for tier in {s.tier for s in single}:
        assert _cl.TIER_LLM[tier].thinking_budget == 0, "thinking is still on"
        assert _cl.TIER_LLM[tier].cot is False, "CoT is still on"

    print(f"[protocol] Octopus: 1 stage {single[0].label} -> {MODEL}, "
          f"thinking=off, cot=off, answers={ANSWER_FORMAT}, batch<={BATCH}, "
          f"chunk={CHUNK_SIZE}")


def configure_movie_schema() -> None:
    """Teach the parser the movie table's columns. Same bridge as the CIDR
    runners -- the parser's column catalogue is Amazon-shaped by default."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import query_parser as qp
    qp.CATEGORY_TABLES = {"movie_reviews": "movie"}
    # Binary sentiment is a SCHEMA fact (Reviews.scoreSentiment has exactly two
    # values), so the predicate matcher may rely on it: "is negative" really is
    # the strict complement of "is positive" here. Without it the matcher refuses
    # the negation, and it is right to -- in general a review can be neither.
    qp.PREDICATE_DOMAIN_NOTE = (
        "- A review's sentiment in this table is BINARY: every review is either "
        "positive or negative. There is no neutral or mixed category.")
    qp.TABLE_COLUMNS = {
        "id": {"type": "TEXT",
               "description": "The movie this review is about "
                              "(movie id string, e.g. 'taken_3')",
               "nl_aliases": "movie, film, for movie X, movie id, taken_3"},
    }


def forbid_runtime_generation() -> None:
    """Make an unregistered predicate a LOUD failure instead of a silent downgrade.

    cascade_planner._predicate_defs falls back to CascadeAgent(mode="llm") for any
    predicate it does not recognise. That fallback generates 3 content + 2 meta
    sub-questions at query time -- the old scheme, on the old scorer, with a
    model chosen at runtime. Nothing in the output says it happened; the run just
    quietly stops being the system the protocol describes.

    It bit us for real: Q1's parse produced a predicate wording the registry did
    not cover, and the whole 1865-row run went through runtime generation before
    anyone noticed. So: refuse.
    """
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import cascade_agent as ca

    def refuse(self, spec):
        raise SystemExit(
            f"\nPREDICATE NOT IN REGISTRY: {spec.slug!r}  (from {spec.key!r})\n"
            f"Runtime sub-question generation is disabled under this protocol -- it "
            f"would silently switch this predicate back to the 3-content+2-meta "
            f"scheme.\n"
            f"Generate it once, offline:\n"
            f"  python vision_evaluation/generate_predicate_questions.py \\\n"
            f"      --slug {spec.slug} --key {spec.key!r} \\\n"
            f"      --context \"<one sentence describing the record kind>\"\n"
            f"or, if it is a re-wording of a predicate you already generated, add "
            f"{spec.slug!r} to that file's \"aliases\" list."
        )

    ca.CascadeAgent._llm = refuse
    ca.CascadeAgent._related = refuse


DB = dict(host="127.0.0.1", port=5432, dbname="octopus",
          user="octopus_user", password="octopus")

TAG_COL = "tag_is_a_positive_review"


CALLS_LOG = REPO / "results" / "cascade_system_run" / "subq_calls.csv"


def reset_calls_log() -> None:
    """Drop the per-call log AND the parser counter so the next measurement covers
    THIS query only."""
    if CALLS_LOG.exists():
        CALLS_LOG.unlink()
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import query_parser as _qp
    _qp.reset_parse_usage()


def read_calls_log() -> dict:
    """Cost/tokens/calls actually spent since the last reset_calls_log().

    The baselines write their own metrics files; ours did not, so every cost
    number so far had to be scraped from stdout by hand -- which is how a whole
    run ended up reported with quality but no cost. Read the log the cascade
    already writes and record it like they do.

    NOTE: the log has one row per ROW, not per call (batched cost is the
    attributed share), so `calls` counts distinct batch ids."""
    import csv as _csv
    if not CALLS_LOG.exists():
        return dict(cost=0.0, calls=0, rows=0, prompt_tok=0, output_tok=0, thinking_tok=0)
    rows = list(_csv.DictReader(CALLS_LOG.open()))
    return dict(
        cost=round(sum(float(r["cost_usd"]) for r in rows), 6),
        calls=len({r.get("batch_id") or i for i, r in enumerate(rows)}),
        rows=len(rows),
        prompt_tok=sum(int(r["prompt_tokens"]) for r in rows),
        output_tok=sum(int(r["output_tokens"]) for r in rows),
        thinking_tok=sum(int(r["thinking_tokens"] or 0) for r in rows),
    )


def record_metrics(qid, elapsed_s: float, extra: dict | None = None,
                   suffix: str = "", cost_divisor: int = 1) -> dict:
    """Append this query's cost/latency to results/octopus/octopus_metrics.json,
    in the same shape the baselines use, so the three are directly comparable."""
    import json
    d = out_dir("octopus", suffix)
    f = d / "octopus_metrics.json"
    all_ = json.loads(f.read_text()) if f.exists() else {}
    import query_parser as _qp
    pu = dict(_qp.PARSE_USAGE)
    rec = {"status": "success", "elapsed_s": round(elapsed_s, 2),
           "model": MODEL, "answer_format": ANSWER_FORMAT, "batch_cap": BATCH,
           **read_calls_log(), **(extra or {})}
    # Total cost = per-row cascade + the parser/matcher calls this query made.
    # The parser is ours alone -- neither baseline has one -- so leaving it out
    # would be an unreported advantage, however small.
    rec["parser_cost"] = round(pu["cost"], 6)
    rec["parser_calls"] = pu["calls"]
    rec["parser_tokens"] = pu["prompt_tokens"] + pu["output_tokens"]
    rec["cascade_cost"] = round(rec["cost"] / cost_divisor, 6)
    rec["money_cost"] = rec.pop("cost") / cost_divisor + pu["cost"]
    if cost_divisor != 1:
        for k in ("prompt_tok", "output_tok", "thinking_tok", "calls", "rows"):
            rec[k] = rec[k] / cost_divisor if k in ("prompt_tok", "output_tok",
                                                    "thinking_tok") else rec[k]
    all_[f"Q{qid}"] = rec
    f.write_text(json.dumps(all_, indent=2))
    print(f"[Q{qid}] ${rec['money_cost']:.4f}  {rec['calls']} calls  "
          f"{rec['rows']} rows judged  {elapsed_s:.1f}s")
    return rec


def verify_row_parity(dedup: bool = False) -> None:
    """Octopus reads Postgres; LOTUS and PZ read the CSV. "All systems see the
    same rows" is therefore an assumption about two separate stores, and it has
    already been wrong once (we ran deduped, their published numbers were on
    2000). Check it instead of assuming it."""
    import csv as _csv

    import psycopg2

    with (data_dir(dedup) / "Reviews.csv").open(newline="") as f:
        rows = list(_csv.DictReader(f))
    csv_total = len(rows)
    csv_per_movie = {m: sum(r["id"] == m for r in rows) for m in (TAKEN3, ANTMAN)}

    with psycopg2.connect(**DB) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM movie_reviews")
        db_total = cur.fetchone()[0]
        db_per_movie = {}
        for m in (TAKEN3, ANTMAN):
            cur.execute("SELECT count(*) FROM movie_reviews WHERE id=%s", (m,))
            db_per_movie[m] = cur.fetchone()[0]

    if db_total != csv_total or db_per_movie != csv_per_movie:
        sys.exit(
            f"ROW PARITY BROKEN — Postgres and the baselines' CSV disagree.\n"
            f"  csv({data_dir(dedup).name}): total={csv_total} {csv_per_movie}\n"
            f"  postgres            : total={db_total} {db_per_movie}\n"
            f"Reload the movie tables (cidr_evaluation/octopus_runner/load_movie.py "
            f"+ adapt_movie_schema.sql + dedup_movie_reviews.py --apply) or rerun "
            f"prepare_data.py so both stores hold the same rows."
        )
    print(f"[protocol] row parity OK: {db_total} reviews in both stores")


def reset_tag_store() -> None:
    """Wipe the WHOLE movie tag store. This is the fresh-experiment reset: the
    per-movie --clear-tags only touches taken_3 or ant_man, which leaves the
    other movies' tags behind and makes "cold start" mean different things
    depending on what ran last."""
    import psycopg2

    with psycopg2.connect(**DB) as conn, conn.cursor() as cur:
        # EVERY tag_ column, not just the sentiment one. Different queries parse
        # into different predicate strings ("is a positive review" vs "is clearly
        # positive"), each of which gets its own column, so clearing one by name
        # leaves the others warm and "cold start" quietly means something
        # different depending on what ran last.
        cur.execute("SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'movie_reviews' AND column_name LIKE 'tag\\_%'")
        cols = [r[0] for r in cur.fetchall()]
        n_tags = 0
        for col in cols:
            cur.execute(f"UPDATE movie_reviews SET {col} = NULL WHERE {col} IS NOT NULL")
            n_tags += cur.rowcount
        cur.execute("DELETE FROM tag_meta WHERE parent_asin IN "
                    "(SELECT parent_asin FROM movie_reviews)")
        n_meta = cur.rowcount
        conn.commit()
    print(f"[protocol] tag store reset: {n_tags} tags across {len(cols)} column(s) "
          f"{cols}, {n_meta} tag_meta rows cleared")


def warn_on_stale_tags() -> None:
    """Tags written under the OLD 2-stage protocol (flash-lite screen, thinking
    possibly on) are indistinguishable from fresh ones at read time — the reuse
    gate only looks at confidence and settled_tier. A tier-1 tag in the store is
    proof of a pre-protocol run, because the single-stage route never uses tier 1."""
    import psycopg2

    with psycopg2.connect(**DB) as conn, conn.cursor() as cur:
        # The tag columns are created on demand, and a table reload drops them all,
        # so their existence cannot be assumed -- checking for a missing column is
        # not defensive coding here, it is the normal cold state.
        cur.execute("SELECT 1 FROM information_schema.columns WHERE table_name="
                    "'movie_reviews' AND column_name=%s", (TAG_COL,))
        if not cur.fetchone():
            print("[protocol] tag store is empty (no tag columns yet) — cold start")
            return
        cur.execute(f"SELECT count(*) FROM movie_reviews WHERE {TAG_COL} IS NOT NULL")
        n_tagged = cur.fetchone()[0]
        cur.execute("SELECT settled_tier, count(*) FROM tag_meta "
                    "WHERE predicate_canon = %s GROUP BY settled_tier ORDER BY 1",
                    (TAG_COL,))
        by_tier = cur.fetchall()
    if not n_tagged:
        return
    print(f"[protocol] tag store holds {n_tagged} tagged rows "
          f"(settled_tier -> count: {dict(by_tier)})")
    if any(t != 2 for t, _ in by_tier):
        print("[protocol] WARNING: tags settled at a tier the single-stage route "
              "never visits — these came from a pre-2026-08-10 run under a "
              "different configuration. Pass --clear-tags for a clean number.")
