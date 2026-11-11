# Offline datasets

`datasets.json` registers six complete workloads with 734,000 Boolean
oracle answers. Each can be loaded by `prep replay --dataset ID`.

| ID | Records | Query instances | Answers |
|---|---:|---:|---:|
| snips | 2,000 | 40 | 80,000 |
| arxiv | 2,000 | 40 | 80,000 |
| amazon_elec_reviews | 5,000 | 40 | 200,000 |
| 20ng_real | 2,000 | 47 | 94,000 |
| 20ng_synthetic | 2,000 | 120 | 240,000 |
| cfpb | 1,000 | 40 | 40,000 |

## Files in each directory

- `workload.json`: execution input combining text, oracle answers, and full
  Discovery, scoring, and per-query certification priorities.
- `records.jsonl`: model-visible input text keyed by `record_id`.
  Synthetic records also have `leaf_label`.
- `queries.json`: complete query definitions, decision semantics, parser
  descriptions, and available prompt templates.
- `oracle_answers.json`: query-by-record Boolean matrix with ordered IDs
  and answer counts.
- `answer_records.jsonl.gz`: one parsed answer per query-record pair with
  request identity and any recorded token usage or latency.
- `sampling_config.json`: registered Discovery and scoring priorities,
  corresponding paid prefixes, and ordering settings.
- `oracle_metadata.json`: the frozen oracle contract and available usage
  metadata. Snips, arXiv, Amazon-Elec, and CFPB use `gpt-5.6-luna` oracle
  answers; real 20NG uses LLM answers; the synthetic bank uses leaf-label
  membership.

`workload.json` is ready to execute. Separate records, definitions, and
answer files make the inputs inspectable; `prep check-data` checks that
these views agree.

## Query instances and batches

The 20NG directories also include `query_batches.json` with `instance_order`,
`instance_to_original_query_id`, and batch sizes. The real workload has one
repeated instance; the synthetic workload has 46. Repeated instances remain
separate entries in the workload.

The synthetic directory includes `taxonomy.json`. An answer is true exactly
when a record's leaf label belongs to the query's `accepted_leaf_labels`.
The data check verifies this rule for all 240,000 answers.

## Run and check

```bash
prep check-data --output outputs/data-check.json
prep replay --dataset snips --output outputs/snips
```

Replay uses the stored answer bank without model calls. See
[FULL_REPLAY_INPUTS.md](FULL_REPLAY_INPUTS.md) for the input schema and
[../RUNNING.md](../RUNNING.md) for all-dataset and query-scaling commands.
