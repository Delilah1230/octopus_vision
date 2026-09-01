# Octopus: supplemental material

Code for the SemBench movie experiments reported in the paper. Everything here is
what produced the numbers in Section 4: the Octopus runners, the three baseline
runners, the scorer, and the benchmark's query definitions.

## What is and is not included

| | |
|---|---|
| Included | Octopus prototype, runners for all four baselines, the scorer, query and ground-truth definitions |
| Not included | API keys, the SemBench data files, the natural-language parser evaluation |

The data is not redistributed here. SemBench's movie scenario is built from the
Rotten Tomatoes reviews dataset, which carries its own license; `prepare_data.py`
below builds the exact input file the paper used from the upstream release, so the
sample is reproducible without us mirroring the reviews.

## Requirements

* Python 3.10 or later, PostgreSQL 14 or later
* An API key for the model provider (`gemini-2.5-flash` was used for every system
  in the paper)
* Baselines each need their own virtual environment; see `requirements.txt`

```bash
pip install -r requirements.txt
cp .env.example .env        # then fill in GEMINI_API_KEY
```

Keys are read from the environment or from `.env`; nothing is hard-coded.

## Setup

The runners expect a local PostgreSQL. Connection settings are the `DB` dict in
`vision_evaluation/movie/protocol.py` (`127.0.0.1:5432`, database `octopus`, user
`octopus_user`); these are local development credentials, not secrets. Edit that
dict if your instance differs.

```bash
createdb octopus
psql -d octopus -f schema.sql
```

In `schema.sql`, the `tag_*` columns of `movie_reviews` are tags: ordinary Boolean
columns that SQL filters like any other, which is the data model the paper argues
for.

The movie scenario itself comes from SemBench (`SemBench/SemBench` on GitHub,
`src/scenario/movie`). Place its `sf_2000` sample under
`cidr_evaluation/sembench/files/movie/data/sf_2000`, then:

```bash
python vision_evaluation/movie/prepare_data.py
```

This builds the one input every system reads, `sf_2000` with exact-duplicate
reviews removed, so all five systems see the same rows and the same ground truth.

## Running

Octopus, one query at a time:

```bash
python vision_evaluation/movie/run_octopus.py --q 2        # filter / aggregate: Q1-Q4, Q8
python vision_evaluation/movie/run_octopus_join.py         # pair queries: Q5-Q7
python vision_evaluation/movie/run_octopus_score.py --q 9  # ranking: Q9
```

`--clear-tags` cold-starts a single query by dropping the tag it would read;
`--reset-tags` empties the whole tag store. Without either, a run reuses whatever
the tag store already holds, which is the point of the workload but makes a
single-query number meaningless.

The five-run protocol reported in the paper:

```bash
python vision_evaluation/movie/run_repeats.py --reps 5
```

Each repetition starts from an empty tag store and executes Q1 to Q9 in benchmark
order, so tags flow forward within a run and never across runs.

Baselines:

```bash
python vision_evaluation/movie/run_lotus.py
python vision_evaluation/movie/run_palimpzest.py
```

Scoring, using the benchmark's own metrics (F1 for retrieval, normalised relative
error for aggregation, rank correlation for ranking):

```bash
python vision_evaluation/movie/evaluate.py
python vision_evaluation/movie/build_table.py
```

## Notes on the numbers

Baseline results are SemBench's own at scale factor 2000 under
`gemini-2.5-flash`; `sembench_declared.json` records them with their provenance.
ThalamusDB implements neither Q9 nor Q10, so its average covers eight queries.
Q10 is excluded from the paper: it asks for the mean of a per-review score from 1
to 5 while its ground truth is the movie's audience score from 1 to 100. Reported
cost and latency cover the semantic operators' model calls; the parser's single
call per query sits outside them.
