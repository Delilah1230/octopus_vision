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

## Database

The runners expect a local PostgreSQL with the connection settings in
`vision_evaluation/movie/protocol.py` (`DB`). These are local development
credentials, not secrets:

```
host 127.0.0.1  port 5432  dbname octopus  user octopus_user
```

Override them by editing that dict if your instance differs.

## Database schema

```bash
createdb octopus
psql -d octopus -f schema.sql
```

`schema.sql` is the table layout the runners expect. `movie_reviews` is worth a
look: the `tag_*` columns in it are tags, stored as ordinary Boolean columns that
SQL filters like any other, which is the whole of the data model the paper argues
for.

## Data

The movie scenario comes from SemBench (`SemBench/SemBench` on GitHub,
`src/scenario/movie`). Obtain its `sf_2000` sample from that repository and place
it under `cidr_evaluation/sembench/files/movie/data/sf_2000`, then:

```bash
python vision_evaluation/movie/prepare_data.py
```

This produces the single input every system reads: SemBench's `sf_2000` sample
with exact-duplicate reviews removed. All four baselines and Octopus read the same
rows and the same ground truth, so the comparison is like for like.

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

## Layout

```
scripts/                     the Octopus prototype
  query_parser.py            natural language to a plan of tag and relational operators
  octopimizer.py             decides how the plan executes
  state_manager.py           the tag store and its provenance
  cascade*.py                model execution
  connect_api.py             model providers; reads keys from the environment
vision_evaluation/movie/     the SemBench movie experiment
  queries.json               the queries, gold SQL and metric per query
  sembench_declared.json     the baseline numbers as published by SemBench
  protocol.py                settings shared by every runner
  run_*.py                   one runner per system
  evaluate.py                the scorer
vision_evaluation/predicates/  predicate definitions used by the tag operators
```

## Notes on the numbers

Baseline results for LOTUS, Palimpzest, ThalamusDB and BigQuery are SemBench's own
at scale factor 2000 under `gemini-2.5-flash`; `sembench_declared.json` records
them together with their provenance. ThalamusDB does not implement Q9 or Q10, so
its average covers eight queries and is not comparable to a nine-query mean. Q10
is excluded from the paper: it asks for the mean of a per-review score from 1 to 5
while its ground truth is the movie's audience score from 1 to 100, so the two do
not measure the same quantity.

Reported cost and latency cover the semantic operators' model calls. The parser's
single call per query sits outside them.
