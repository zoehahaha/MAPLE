# Running PREP

Run commands from the repository root after installing with
`python -m pip install -e ".[test]"`. Installation obtains dependencies;
the checks, reports, and replay commands use local files only.

## Check the submission

```bash
prep check-data --output outputs/data-check.json
python -m pytest -q
prep smoke --output outputs/smoke
prep validate
prep report --output outputs/paper
```

Data validation checks all 734,000 answers against the input records, query
definitions, separate answer records, order permutations, query-instance
mappings, and the synthetic workload's label-membership rule. Data and
paper validation commands exit with a nonzero status on failure.

These checks cover data integrity and execution. Numerical reproduction of
the manuscript experiments requires comparing complete replay outputs with
the corresponding paper reference values.

## Replay complete workloads

Dataset IDs are `snips`, `arxiv`, `amazon_elec_reviews`, `20ng_real`,
`20ng_synthetic`, and `cfpb`.

```bash
prep replay --dataset snips --output outputs/snips
prep replay --dataset cfpb --config experiments/configs/cfpb_case.json \
  --output outputs/cfpb
```

The default configuration is `experiments/configs/paper.json`.
To run all six sequentially in one batch job:

```bash
set -e
for dataset in snips arxiv amazon_elec_reviews 20ng_real 20ng_synthetic cfpb; do
  prep replay --dataset "$dataset" --output "outputs/$dataset"
done
```

Exact support summation and sample-size allocation can be expensive on full
workloads. Duration depends on the inputs and selected plans. Use the small
execution checks below before starting a long job, and select a separate
output directory for every experiment.

## Small execution checks on all datasets

```bash
set -e
for dataset in snips arxiv amazon_elec_reviews 20ng_real 20ng_synthetic cfpb; do
  prep replay --dataset "$dataset" --limit-rows 20 --limit-queries 4 \
    --config experiments/configs/tiny_demo.json \
    --output "outputs/check-$dataset"
done
```

The limits take prefixes of stored row IDs and query-instance IDs. Answer
mappings and sampling priorities are filtered to those IDs while preserving
order. The result records the executed `rows`, `queries`, and
`required_outputs`. These checks run PREP on actual dataset inputs at small
scale; full runs omit both limits.

## Query scaling

The real 20NG workload has 47 instances and 46 distinct queries. The
synthetic workload has 120 instances and 74 distinct queries. Repeated query
instances remain in the input when selecting prefixes.

```bash
set -e
for queries in 10 20 40 47; do
  prep replay --dataset 20ng_real --limit-queries "$queries" \
    --config experiments/configs/20ng_scaling.json \
    --output "outputs/scaling-real-$queries"
done
for queries in 10 20 40 47 80 120; do
  prep replay --dataset 20ng_synthetic --limit-queries "$queries" \
    --config experiments/configs/20ng_scaling.json \
    --output "outputs/scaling-synthetic-$queries"
done
```

`query_batches.json` maps each instance to its original query. Synthetic
records include leaf labels and queries list their accepted labels.

## Explicit inputs and configurations

```bash
prep replay --workload data/workloads/snips/workload.json \
  --config experiments/configs/paper.json --output outputs/snips-explicit
python experiments/run.py --workload data/workloads/snips/workload.json \
  --config experiments/configs/paper.json --output outputs/snips-script
```

The script also accepts `--limit-rows` and `--limit-queries`. For custom
inputs, follow [data/FULL_REPLAY_INPUTS.md](data/FULL_REPLAY_INPUTS.md).
`base_config` loads a JSON file relative to the selected configuration;
the selected file's fields override inherited settings. CFPB and scaling
configurations inherit `paper.json`, including its failure budget and
control request count.

When running outside the repository root, provide `--repo`:

```bash
prep replay --repo /path/to/prep --dataset snips --output /path/to/results/snips
prep check-data --repo /path/to/prep
```

## Output fields

Each replay writes `RESULT.json`:

- `workload`, `rows`, `queries`, `required_outputs`: executed input size.
- `alpha_per_query`: joint per-query failure budget.
- `exclusive_request_counts`: distinct acquired answers, assigned once to
  Discovery, scoring example, source, completion, certification, fallback, or control roles.
- `calls_saved`, `saving_percent`: savings relative to one Direct answer per
  query-record pair, including charged control requests.
- `initial_forecast`, `final_forecast`, and traces: plan forecasts.
- `per_query`: precision, recall, known-positive count before certification,
  certification size, and certificate outcomes.
- `observed_pr_misses`: queries falling below either 0.90 target.
- `provider_attempts`: zero for offline replay.
- `elapsed_seconds`: local replay duration.

`prep report` exports manuscript Tables I-IV, original Figures 3-6, and
summaries of printed numbers. Replay computes new measurements from the
selected bank and configuration. The frozen-output baseline evaluator is
documented in [experiments/baselines/README.md](experiments/baselines/README.md).

## Validation environment

The submission was checked with Python 3.12.7, NumPy 1.26.4, SciPy 1.13.1,
and pytest 7.4.4. Declared requirements are Python >=3.10, NumPy >=1.24,
and SciPy >=1.10. Outputs, virtual environments, and caches are excluded
from the submission directory.
