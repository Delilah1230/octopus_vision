#!/usr/bin/env python3
"""
LOTUS on SemBench movie, at UPSTREAM SemBench settings.

Every operator body below is transcribed from
    src/scenario/movie/runner/lotus_runner/lotus_runner.py
and every LM knob from
    src/runner/generic_lotus_runner/generic_lotus_runner.py::_configure_lm
(SemBench/SemBench @ main). Prompt strings are byte-identical to upstream --
do not "improve" them, the point is that this is their query, not ours.

Deltas from upstream, all forced by our environment, all listed here:
  * We call the AI-Studio endpoint ("gemini/gemini-2.5-flash") because we
    authenticate with GEMINI_API_KEY. Same weights, different endpoint.
  * We read the DEDUPED Reviews.csv (protocol.py::DEDUP) instead of the raw
    2000-row file, so all three systems see identical rows. --raw restores it.
  * We run one repetition, not five. SemBench reports cost/quality variance as
    small and latency variance as large -- so latency here is qualitative only.

    cidr_evaluation/.venv_baselines/bin/python \
        vision_evaluation/movie/run_lotus.py --queries 2,3,4,5,6,7,8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# MUST precede the torch/faiss import chain. LOTUS's approximate join loads
# sentence-transformers (torch) AND faiss in one process; on Apple Silicon both
# link their own OpenMP runtime and the sim-join index build segfaults outright
# (exit 139, no Python traceback, so the runner's try/except never sees it).
# Pinning OpenMP to one thread avoids the conflict. Environment fix, not a
# protocol change -- it does not alter what LOTUS computes.
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import protocol as P

# ── the queries, transcribed from upstream ────────────────────────────────────
# Upstream calls sem_filter with no `strategy=`, i.e. ReasoningStrategy is None
# and LOTUS's non_cot_prompt_formatter is used: "Answer: <True or False>" with no
# room to reason. That is the setting, not an oversight -- see
# notes/cidr/baseline_reasoning_config.md §2.1.

FILTER_POSITIVE = ('Determine if the following movie review is clearly positive. '
                   'Review: "{reviewText}".')
FILTER_POSITIVE_Q3 = "Determine if the following review is clearly positive. Review: {reviewText}"
FILTER_POSITIVE_Q4 = "Determine if the following review is clearly positive. Review: {reviewText}."

JOIN_SAME = ('These two movie reviews express the same sentiment - either both are '
             'positive or both are negative. Review 1: "{reviewText:left}" '
             'Review 2: "{reviewText:right}"')
JOIN_OPPOSITE = ('These two movie reviews express opposite sentiments - one is '
                 'positive and the other is negative. Review 1: "{reviewText:left}" '
                 'Review 2: "{reviewText:right}"')

MAP_SENTIMENT = ("Classify the sentiment of this review as either 'POSITIVE' or 'NEGATIVE'. "
                 "Only output the exact word 'POSITIVE' or 'NEGATIVE' with no additional text. "
                 "Review: {reviewText}")

# Q9/Q10 scoring rubric, verbatim from upstream (ranking="map" is the
# GenericLotusRunner default, so sem_map -- not sem_topk -- is the upstream path).
SCORING_PROMPT = """Score from 1 to 5 how much did the reviewer like the movie based on provided rubrics.

Rubrics:
5: Very positive. Strong positive sentiment, indicating high satisfaction.
4: Positive. Noticeably positive sentiment, indicating general satisfaction.
3: Neutral. Expresses no clear positive or negative sentiment. May be factual or descriptive without emotional language.
2: Negative. Noticeably negative sentiment, indicating some level of dissatisfaction but without strong anger or frustration.
1: Very negative. Strong negative sentiment, indicating high dissatisfaction, frustration, or anger.

Review: {reviewText}

Only provide the score number (1-5) with no other comments."""


def _num_or_neutral(v: float | str) -> float:
    """Upstream coerces an unparseable or out-of-range score to 3.0 (neutral)."""
    try:
        n = float(v)
    except (ValueError, TypeError):
        return 3.0
    return n if 1 <= n <= 5 else 3.0


class Runner:
    def __init__(self, reviews, lm, cascade_args, join_settings):
        self.reviews = reviews
        self.lm = lm
        self.cascade_args = cascade_args      # None when policy == "exact"
        self.join_settings = join_settings    # callable or None

    # -- helpers ------------------------------------------------------------
    def _movie(self, movie_id):
        return self.reviews[self.reviews["id"] == movie_id]

    def _join(self, df, instruction):
        """Upstream self-join. With policy='approximate' (the SemBench default)
        this passes CascadeArgs(recall_target=0.8, precision_target=0.8), which
        routes most pairs through a cheap proxy and only escalates the uncertain
        ones. Skipping it -- as our grocery baselines did -- runs LOTUS at the
        oracle endpoint of its own cost/quality curve and makes it look far more
        expensive than SemBench reports.

        The proxy here is NOT the helper LM. sem_join.py:418 logs "Helper model is
        not supported yet. Default to similarity join." and uses run_sem_sim_join
        over the embeddings, so CascadeArgs.proxy_model=HELPER_LM is inert and no
        logprob support is required from gemini. That is why rm/vs must be
        configured before the join -- without them the proxy has nothing to score
        with. (Corrects the worry in baseline_reasoning_config.md §2.4, which was
        about sem_filter cascades, not joins.)"""
        if self.join_settings is not None:
            self.join_settings()
        df = df.reset_index(drop=True)
        if self.cascade_args is not None:
            joined = df.sem_join(df, join_instruction=instruction,
                                 cascade_args=self.cascade_args)
        else:
            joined = df.sem_join(df, join_instruction=instruction)
        return joined[joined["reviewId:left"] != joined["reviewId:right"]]

    @staticmethod
    def _pairs_out(joined, limit=None):
        import pandas as pd
        if len(joined) == 0:
            return pd.DataFrame(columns=["id", "reviewId", "reviewId2"])
        if limit:
            joined = joined.head(limit)
        out = joined[["id:left", "reviewId:left", "reviewId:right"]].copy()
        out.columns = ["id", "reviewId", "reviewId2"]
        return out

    # -- queries ------------------------------------------------------------
    def q1(self):
        import pandas as pd
        hits = self.reviews.sem_filter(FILTER_POSITIVE)
        return pd.DataFrame({"reviewId": hits.head(5)["reviewId"]})

    def q2(self):
        import pandas as pd
        hits = self._movie(P.TAKEN3).sem_filter(FILTER_POSITIVE)
        return pd.DataFrame({"reviewId": hits.head(5)["reviewId"]})

    def q3(self):
        import pandas as pd
        hits = self._movie(P.TAKEN3).sem_filter(FILTER_POSITIVE_Q3)
        return pd.DataFrame({"positive_review_cnt": [hits.shape[0]]})

    def q4(self):
        import pandas as pd
        pool = self._movie(P.TAKEN3)
        if len(pool) == 0:
            return pd.DataFrame({"positivity_ratio": [0.0]})
        hits = pool.sem_filter(FILTER_POSITIVE_Q4)
        return pd.DataFrame({"positivity_ratio": [len(hits) / len(pool)]})

    def q5(self):
        return self._pairs_out(self._join(self._movie(P.ANTMAN), JOIN_SAME), limit=10)

    def q6(self):
        return self._pairs_out(self._join(self._movie(P.ANTMAN), JOIN_OPPOSITE), limit=10)

    def q7(self):
        return self._pairs_out(self._join(self._movie(P.ANTMAN), JOIN_OPPOSITE))

    def q8(self):
        import pandas as pd
        mapped = self._movie(P.TAKEN3).sem_map(MAP_SENTIMENT)
        counts = mapped["_map"].value_counts().reset_index()
        counts.columns = ["scoreSentiment", "count"]
        for s in ("POSITIVE", "NEGATIVE"):
            if s not in counts["scoreSentiment"].values:
                counts = pd.concat(
                    [counts, pd.DataFrame({"scoreSentiment": [s], "count": [0]})],
                    ignore_index=True)
        return counts.sort_values("scoreSentiment").reset_index(drop=True)

    def q9(self):
        import pandas as pd
        scored = self._movie(P.ANTMAN).sem_map(SCORING_PROMPT)
        return pd.DataFrame([{"reviewId": r["reviewId"],
                              "reviewScore": _num_or_neutral(r["_map"])}
                             for _, r in scored.iterrows()])

    def q10(self):
        import pandas as pd
        scored = self.reviews.sem_map(SCORING_PROMPT)
        rows = []
        for movie_id, grp in scored.groupby("id"):
            vals = [_num_or_neutral(v) for v in grp["_map"]]
            rows.append({"movieId": movie_id,
                         "movieScore": round(sum(vals) / len(vals), 2) if vals else 3.0})
        return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--policy", choices=["approximate", "exact"], default="approximate",
                    help="upstream default is 'approximate' (GenericLotusRunner.__init__); "
                         "the movie runner never overrides it, so joins get "
                         "CascadeArgs(0.8, 0.8). 'exact' drops cascade_args.")
    ap.add_argument("--raw", action="store_true",
                    help="read the untouched 2000-row Reviews.csv instead of the "
                         "deduped one (breaks row-parity with the other systems)")
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--num-retries", type=int, default=5,
                    help="litellm-level retries per call. NOT a protocol knob -- see "
                         "the comment in main(). 0 reproduces the crash.")
    ap.add_argument("--cost-limit", type=float, default=3.0,
                    help="abort the query once LOTUS's own physical usage passes this "
                         "many USD. A join that silently falls back from the "
                         "approximate policy to the full 128x128 join costs ~$1.4 and "
                         "produces nothing if it then dies -- this bounds the damage.")
    a = ap.parse_args()

    P.require_key()
    ddir = P.data_dir(raw=a.raw)
    out = P.out_dir("lotus", a.out_suffix)

    import pandas as pd
    import lotus
    from lotus.models import LM
    from lotus.types import CascadeArgs, UsageLimit

    # num_retries: LOTUS calls litellm.batch_completion, which puts the EXCEPTION
    # OBJECT into the result list for a failed call instead of raising. LOTUS then
    # does response.choices[0] on it and dies with
    #     AttributeError: 'ServiceUnavailableError' object has no attribute 'choices'
    # -- so ONE transient 503 anywhere in a 16k-call join throws the whole query
    # away after it has already been paid for. Worse, the same 503 during
    # learn_join_cascade_threshold is caught and logged as "Default to full join",
    # which silently abandons the approximate policy and turns the cheap join into
    # the expensive one.
    #
    # This is us surviving OUR endpoint, not us changing LOTUS: upstream ran on
    # Vertex, where a 20-worker burst does not get throttled the way AI Studio
    # throttles it. It changes reliability, not semantics.
    lm = LM(P.LOTUS_MODEL,
            max_batch_size=P.LOTUS_UPSTREAM["max_batch_size"],
            max_tokens=P.LOTUS_UPSTREAM["max_tokens"],
            reasoning_effort=P.LOTUS_UPSTREAM["reasoning_effort"],
            num_retries=a.num_retries,
            physical_usage_limit=UsageLimit(total_cost_limit=a.cost_limit))
    lotus.settings.configure(lm=lm)

    cascade_args, join_settings = None, None
    if a.policy == "approximate":
        from lotus.models import SentenceTransformersRM
        from lotus.vector_store import FaissVS
        rm, vs = SentenceTransformersRM(model=P.APPROX_RM), FaissVS()
        cascade_args = CascadeArgs(**P.APPROX_CASCADE)
        join_settings = lambda: lotus.settings.configure(lm=lm, rm=rm, vs=vs)  # noqa: E731

    reviews = pd.read_csv(ddir / "Reviews.csv")
    print(f"[lotus] {len(reviews)} reviews from {ddir.name}, model={P.LOTUS_MODEL}, "
          f"workers={P.WORKERS}, policy={a.policy}, "
          f"reasoning_effort={P.LOTUS_UPSTREAM['reasoning_effort']!r}")

    runner = Runner(reviews, lm, cascade_args, join_settings)
    mfile = out / "lotus_metrics.json"
    metrics = json.loads(mfile.read_text()) if mfile.exists() else {}

    for qid in [int(q) for q in a.queries.split(",") if q.strip()]:
        print(f"\n{'='*66}\n  LOTUS Q{qid}\n{'='*66}")
        lm.reset_stats()
        t0 = time.time()
        try:
            res = getattr(runner, f"q{qid}")()
            status, err = "success", None
        except Exception as e:                       # noqa: BLE001
            import traceback
            traceback.print_exc()
            res, status, err = pd.DataFrame(), "failed", f"{type(e).__name__}: {e}"
        elapsed = time.time() - t0

        u = lm.stats.physical_usage
        cost = (u.prompt_tokens / 1e6 * P.PRICING[P.MODEL]["text"]
                + u.completion_tokens / 1e6 * P.PRICING[P.MODEL]["output"])
        metrics[f"Q{qid}"] = {
            "status": status, "error": err,
            "n_rows": int(len(res)), "elapsed_s": round(elapsed, 2),
            "prompt_tokens": int(u.prompt_tokens),
            "completion_tokens": int(u.completion_tokens),
            "money_cost": round(cost, 6),
            "lotus_reported_cost": float(getattr(u, "total_cost", 0.0) or 0.0),
            "model": P.MODEL, "policy": a.policy, "dedup": not a.raw,
        }
        if status == "success":
            res.to_csv(out / f"Q{qid}.csv", index=False)
        print(f"  rows={len(res)}  ${cost:.4f}  "
              f"{u.prompt_tokens}+{u.completion_tokens} tok  {elapsed:.1f}s  [{status}]")
        if err:
            print(f"  ERROR: {err}")
        mfile.write_text(json.dumps(metrics, indent=2))

    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
