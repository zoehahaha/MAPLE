# PREP

Code and offline datasets for **Progressive Rule Mining and
Certification-Aware Execution Planning for Cross-Query Answer Reuse**.
PREP learns signed Boolean reuse rules, refines their input conditions,
selects execution plans by forecast cost, and certifies per-query precision
and recall with Direct fallback.

All six workload inputs, complete query definitions, frozen oracle answers,
and sampling orders are included. Query files also contain the prompt
templates preserved with the datasets. Replay, data checks, and paper report
export run locally without API credentials or model calls.

## Quick start

Run from the repository root with Python 3.10 or newer:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[test]"
prep check-data --output outputs/data-check.json
python -m pytest -q
prep smoke --output outputs/smoke
prep validate
prep report --output outputs/paper
```

`prep check-data` checks every input row, query definition, oracle answer,
sampling order, duplicate-query mapping, and answer record. `prep smoke`
runs the small CPU example. `prep validate` checks the paper reference
tables; `prep report` exports those tables and the original figures.

## Included datasets

| Dataset ID | Records | Query instances | Oracle answers |
|---|---:|---:|---:|
| `snips` | 2,000 | 40 | 80,000 |
| `arxiv` | 2,000 | 40 | 80,000 |
| `amazon_elec_reviews` | 5,000 | 40 | 200,000 |
| `20ng_real` | 2,000 | 47 | 94,000 |
| `20ng_synthetic` | 2,000 | 120 | 240,000 |
| `cfpb` | 1,000 | 40 | 40,000 |
| Total | | | **734,000** |

The files under `data/workloads/` include input records, query definitions,
oracle answers, answer records, sampling settings, oracle metadata, and
ready-to-load `workload.json` files. See [data/README.md](data/README.md).

## Run PREP

```bash
prep replay --dataset snips --output outputs/snips
prep replay --dataset cfpb --config experiments/configs/cfpb_case.json \
  --output outputs/cfpb
```

The default configuration is `experiments/configs/paper.json`: precision and
recall targets of 0.90 and a joint per-query failure budget of 0.05. Exact
finite-population calculations can be expensive on full workloads; use batch
jobs for long runs. For a quick check on actual dataset inputs:

```bash
prep replay --dataset snips --limit-rows 20 --limit-queries 4 \
  --config experiments/configs/tiny_demo.json --output outputs/snips-check
```

The limits select prefixes of the stored input and query order, preserving
the relative Discovery, scoring, and certification priorities. Omit both limits for
a complete workload replay.

Replay writes `RESULT.json` with measured request counts, savings,
forecasts, certification outcomes, and per-query precision/recall.
`reference/` contains the manuscript's reported results; `prep report`
exports those reference tables and figures separately.

See [RUNNING.md](RUNNING.md) for all-dataset runs, query scaling, custom
inputs, outputs, and validation. Scientific parameters are described in
[experiments/PARAMETERS.md](experiments/PARAMETERS.md).

## Repository layout

```text
src/prep/                 Discovery, planning, certification, execution, CLI
data/workloads/           Six complete offline workloads
data/datasets.json        Dataset IDs, input paths, and dimensions
data/tiny_demo.json       Small CPU example
experiments/configs/      Paper, demo, CFPB, and scaling configurations
experiments/run.py        Configuration-driven replay script
experiments/baselines/    Frozen-output quality evaluator
reference/                Paper Tables I-IV and Figures 3-6
tests/                    Method and data checks
```

## License

PREP code is MIT licensed. Dataset and upstream baseline licenses remain
their own. Public input texts and frozen answer banks are provided for
offline evaluation.
