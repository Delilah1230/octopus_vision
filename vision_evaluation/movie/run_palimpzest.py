#!/usr/bin/env python3
"""
Palimpzest on SemBench movie, at UPSTREAM SemBench settings.

Query bodies transcribed from
    src/scenario/movie/runner/palimpzest_runner/palimpzest_runner.py
Config from
    config/system/palimpzest/gemini-2.5-flash-maxquality.json
    src/runner/generic_palimpzest_runner/generic_palimpzest_runner.py::palimpzest_config
(SemBench/SemBench @ main). Note how upstream builds the config: reasoning_effort
is null in the JSON and the runner *only adds the kwarg if it is not null*, so
QueryProcessorConfig never receives it at all. That is load-bearing -- see
force_thinking_off() below.

Deltas from upstream, all forced by our environment:
  * available_models is GEMINI_2_5_FLASH, which in PZ means the VERTEX model id
    "vertex_ai/gemini-2.5-flash". We have no GCP credentials, so we use
    GOOGLE_GEMINI_2_5_FLASH and remap it to "gemini/gemini-2.5-flash". Same
    weights and same single-model candidate set; different endpoint.
  * That endpoint switch is what force_thinking_off() compensates for. See below.
  * Deduped Reviews.csv (--raw restores the 2000-row file); one repetition.

    cidr_evaluation/.conda_pz/bin/python \
        vision_evaluation/movie/run_palimpzest.py --queries 2,3,4,5,6,7,8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import protocol as P

FILTER_POSITIVE = "Determine if the following movie review is clearly positive."
JOIN_SAME = ("These two movie reviews express the same sentiment - either both are "
             "positive or both are negative.")
JOIN_OPPOSITE = ("These two movie reviews express opposite sentiments - one is "
                 "positive and the other is negative.")
POSITIVITY_DESC = ("Return 1 if the following review is positive, and 0 if the review "
                   "is not positive. Only output a single numeric value (1 or 0) with "
                   "no additional commentary")
SENTIMENT_DESC = ("Return POSITIVE if the following review is positive, and NEGATIVE "
                  "if the review is not positive. Only output POSITIVE or NEGATIVE "
                  "with no additional commentary")

SCORE_DESC = """Score from 1 to 5 how much did the reviewer like the movie based on provided rubrics.

Rubrics:
5: Very positive. Strong positive sentiment, indicating high satisfaction.
4: Positive. Noticeably positive sentiment, indicating general satisfaction.
3: Neutral. Expresses no clear positive or negative sentiment. May be factual or descriptive without emotional language.
2: Negative. Noticeably negative sentiment, indicating some level of dissatisfaction but without strong anger or frustration.
1: Very negative. Strong negative sentiment, indicating high dissatisfaction, frustration, or anger.

Review: {reviewText}

Only provide the score number (1-5) with no other comments."""


def force_thinking_off() -> None:
    """Reproduce what upstream gets for free on Vertex.

    generators.py only maps reasoning_effort=None -> "disable" inside
    `if self.model.is_vertex_model():`, and is_vertex_model() is a substring test
    for "vertex_ai" in the model id. Upstream runs "vertex_ai/gemini-2.5-flash",
    so their null becomes "disable" and their PZ is thinking-OFF. We run
    "gemini/gemini-2.5-flash", the branch never fires, the kwarg is dropped, and
    gemini's own dynamic thinking budget stays ON.

    So injecting reasoning_effort="disable" at the litellm boundary is not us
    handicapping PZ -- it is the only way to MATCH upstream on this endpoint.

    Crucially we inject it HERE and not in QueryProcessorConfig: the optimizer
    reads config-level reasoning_effort to pick the prompt strategy
    (rules.py::LLMFilterRule.substitute, `no_reasoning = reasoning_effort in
    [None, "minimal", "low"]`), and passing "disable" there would flip PZ from
    COT_BOOL_NO_REASONING to COT_BOOL -- i.e. it would silently turn written CoT
    ON, which upstream does not have. Leaving the config value at None keeps the
    upstream prompt strategy; the litellm hook keeps the upstream thinking budget.
    """
    import litellm
    orig = litellm.completion

    def completion(*args, **kwargs):
        if str(kwargs.get("model", "")).startswith("gemini/"):
            kwargs["reasoning_effort"] = "disable"
        return orig(*args, **kwargs)

    litellm.completion = completion


def spy_on_prompt_strategies() -> dict:
    """Record which PromptStrategy actually ISSUED calls.

    Upstream's setting (reasoning_effort absent from the config) is supposed to
    yield the *_NO_REASONING strategies for filter / convert / join. That is an
    inference about optimizer internals, so verify it rather than trust it.

    Hook Generator.__call__, NOT Generator.__init__. PZ's optimizer CONSTRUCTS a
    generator for every candidate physical plan it enumerates -- mixture-of-agents,
    critic/refine, and plain COT_QA all get built and then discarded. Counting
    constructions reports written-CoT strategies on a run whose executed plan had
    none. Only a strategy that reaches __call__ actually shaped a prompt we paid
    for.
    """
    from palimpzest.query.generators import generators as gen
    seen: dict[str, int] = {}
    orig = gen.Generator.__call__

    def patched(self, *args, **kwargs):
        name = getattr(self.prompt_strategy, "name", str(self.prompt_strategy))
        seen[name] = seen.get(name, 0) + 1
        return orig(self, *args, **kwargs)

    gen.Generator.__call__ = patched
    return seen


def remap_google_models(Model, MODEL_CARDS) -> None:
    """PZ 0.8.2 spells its AI-Studio Gemini ids "google/...", but the litellm it
    pins (1.94.0) has no "google" provider -- every call dies with "LLM Provider
    NOT provided". Rewrite the enum VALUE (generators passes model.value straight
    to litellm) and re-register the MODEL_CARDS key, or the optimizer KeyErrors."""
    for m in (Model.GOOGLE_GEMINI_2_5_FLASH, Model.GOOGLE_GEMINI_2_5_FLASH_LITE,
              Model.GOOGLE_GEMINI_2_5_PRO):
        if m.value.startswith("google/"):
            old, new = m.value, m.value.replace("google/", "gemini/", 1)
            Model._value2member_map_.pop(old, None)
            m._value_ = new
            Model._value2member_map_[new] = m
            if old in MODEL_CARDS:
                MODEL_CARDS[new] = MODEL_CARDS[old]


class Runner:
    def __init__(self, pz, GroupBySig, reviews_df, make_cfg):
        self.pz = pz
        self.GroupBySig = GroupBySig
        # upstream renames id -> movieId on every query except Q1
        self.df = reviews_df.rename(columns={"id": "movieId"})
        self.raw_df = reviews_df
        self.make_cfg = make_cfg

    def _taken3(self):
        ds = self.pz.MemoryDataset(id="reviews", vals=self.df)
        return ds.filter(lambda r: r["movieId"] == P.TAKEN3)

    def _antman_pair(self):
        left = self.df[self.df["movieId"] == P.ANTMAN]
        right = left.rename(columns={c: f"{c}_right" for c in left.columns})
        return (self.pz.MemoryDataset(id="input1", vals=left),
                self.pz.MemoryDataset(id="input2", vals=right))

    def _join(self, condition, limit=None):
        a, b = self._antman_pair()
        ds = a.sem_join(b, condition=condition,
                        depends_on=["reviewText", "reviewText_right"])
        ds = ds.project(["movieId", "reviewId", "reviewId_right"])
        if limit:
            ds = ds.limit(limit)
        return ds.run(self.make_cfg())

    def q1(self):
        ds = self.pz.MemoryDataset(id="reviews", vals=self.raw_df)
        ds = ds.sem_filter(FILTER_POSITIVE, depends_on=["reviewText"])
        return ds.project(["reviewId"]).limit(5).run(self.make_cfg())

    def q2(self):
        ds = self._taken3().sem_filter(FILTER_POSITIVE, depends_on=["reviewText"])
        return ds.project(["reviewId"]).limit(5).run(self.make_cfg())

    def q3(self):
        ds = self._taken3().sem_filter(FILTER_POSITIVE, depends_on=["reviewText"])
        return ds.count().run(self.make_cfg())

    def q4(self):
        ds = self._taken3().sem_add_columns(
            [{"name": "positivity", "type": int, "desc": POSITIVITY_DESC}],
            depends_on=["reviewText"])
        return ds.project(["positivity"]).average().run(self.make_cfg())

    def q5(self):
        return self._join(JOIN_SAME, limit=10)

    def q6(self):
        return self._join(JOIN_OPPOSITE, limit=10)

    def q7(self):
        return self._join(JOIN_OPPOSITE)

    def q8(self):
        ds = self._taken3().sem_add_columns(
            [{"name": "sentiment", "type": str, "desc": SENTIMENT_DESC}],
            depends_on=["reviewText"])
        ds = ds.project(["sentiment"])
        gby = self.GroupBySig(group_by_fields=["sentiment"],
                              agg_funcs=["count"], agg_fields=["sentiment"])
        return ds.groupby(gby).run(self.make_cfg())

    def q9(self):
        antman = self.df[self.df["movieId"] == P.ANTMAN]
        ds = self.pz.MemoryDataset(id="reviews", vals=antman)
        ds = ds.sem_add_columns([{"name": "reviewScore", "type": int, "desc": SCORE_DESC}],
                                depends_on=["reviewText"])
        return ds.project(["reviewId", "reviewScore"]).run(self.make_cfg())

    def q10(self):
        ds = self.pz.MemoryDataset(id="reviews", vals=self.df)
        ds = ds.sem_add_columns([{"name": "reviewScore", "type": int, "desc": SCORE_DESC}],
                                depends_on=["reviewText"])
        ds = ds.project(["movieId", "reviewScore"])
        gby = self.GroupBySig(group_by_fields=["movieId"],
                              agg_funcs=["average"], agg_fields=["reviewScore"])
        return ds.groupby(gby).run(self.make_cfg())


# Column layout the shared evaluator expects, per query.
RENAME = {
    1: ["reviewId"], 2: ["reviewId"],
    3: ["positive_review_cnt"], 4: ["positivity_ratio"],
    5: ["id", "reviewId", "reviewId2"],
    6: ["id", "reviewId", "reviewId2"],
    7: ["id", "reviewId", "reviewId2"],
    8: ["scoreSentiment", "count"],
    9: ["reviewId", "reviewScore"],
    10: ["movieId", "movieScore"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--out-suffix", default="")
    a = ap.parse_args()

    P.require_key()
    ddir = P.data_dir(raw=a.raw)
    out = P.out_dir("palimpzest", a.out_suffix)

    import pandas as pd
    import palimpzest as pz
    from palimpzest.constants import MODEL_CARDS, Model
    from palimpzest.core.elements.groupbysig import GroupBySig

    remap_google_models(Model, MODEL_CARDS)
    force_thinking_off()
    strategies = spy_on_prompt_strategies()
    print(f"[pz] model -> {Model.GOOGLE_GEMINI_2_5_FLASH.value}, "
          f"reasoning_effort='disable' injected at the litellm boundary")

    def make_cfg():
        # A FRESH config per query: PZ's _normalize_strategies rewrites
        # execution_strategy from str to an enum IN PLACE, so a reused config
        # blows up on the second .run() with "'ParallelExecutionStrategy' object
        # has no attribute 'upper'". reasoning_effort is deliberately NOT passed
        # -- see force_thinking_off().
        return pz.QueryProcessorConfig(
            policy=pz.MaxQuality(),
            max_workers=P.PZ_UPSTREAM["max_workers"],
            join_parallelism=P.PZ_UPSTREAM["join_parallelism"],
            verbose=P.PZ_UPSTREAM["verbose"],
            progress=P.PZ_UPSTREAM["progress"],
            available_models=[Model.GOOGLE_GEMINI_2_5_FLASH],
            use_vertex=False,
        )

    reviews = pd.read_csv(ddir / "Reviews.csv")
    print(f"[pz] {len(reviews)} reviews from {ddir.name}, policy=MaxQuality, "
          f"models=[{P.MODEL}], workers={P.WORKERS}")

    runner = Runner(pz, GroupBySig, reviews, make_cfg)
    mfile = out / "pz_metrics.json"
    metrics = json.loads(mfile.read_text()) if mfile.exists() else {}

    for qid in [int(q) for q in a.queries.split(",") if q.strip()]:
        print(f"\n{'='*66}\n  Palimpzest Q{qid}\n{'='*66}")
        t0 = time.time()
        stats = None
        try:
            res = getattr(runner, f"q{qid}")()
            df = res.to_df() if hasattr(res, "to_df") else res
            stats = getattr(res, "execution_stats", None)
            status, err = "success", None
        except Exception as e:                       # noqa: BLE001
            import traceback
            traceback.print_exc()
            df, status, err = pd.DataFrame(), "failed", f"{type(e).__name__}: {e}"
        elapsed = time.time() - t0

        cost = float(getattr(stats, "total_execution_cost", 0.0) or 0.0)
        toks = int(getattr(stats, "total_tokens", 0) or 0)
        # PZ swallows per-row LLM errors and still returns an (empty) frame, so a
        # run that made zero calls must not be reported as a success.
        if status == "success" and toks == 0 and cost == 0.0:
            status, err = "failed", "no LLM calls were made (0 tokens, $0)"

        metrics[f"Q{qid}"] = {
            "status": status, "error": err,
            "n_rows": int(len(df)), "elapsed_s": round(elapsed, 2),
            "money_cost": cost, "token_usage": toks,
            "model": P.MODEL, "dedup": not a.raw,
            "columns": list(getattr(df, "columns", [])),
        }
        if status == "success" and len(df):
            want = RENAME[qid]
            slim = df.iloc[:, :len(want)].copy()
            slim.columns = want
            slim.to_csv(out / f"Q{qid}.csv", index=False)
        print(f"  rows={len(df)}  ${cost:.4f}  {toks} tok  {elapsed:.1f}s  [{status}]")
        if err:
            print(f"  ERROR: {err}")
        metrics[f"Q{qid}"]["prompt_strategies"] = dict(strategies)
        mfile.write_text(json.dumps(metrics, indent=2))

    print(f"\n[pz] prompt strategies instantiated: {strategies}")
    written_cot = [k for k in strategies if k.startswith("COT_") and "NO_REASONING" not in k]
    if written_cot:
        print(f"[pz] WARNING: {written_cot} means WRITTEN CoT was on. Upstream "
              f"(reasoning_effort absent) gets the *_NO_REASONING variants — "
              f"these numbers are NOT at upstream settings.")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
