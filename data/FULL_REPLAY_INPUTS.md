# Replay input format

Complete inputs for all six datasets are included under `data/workloads/`.
`tiny_demo.json` illustrates the same schema on a small CPU example.

`workload.json` contains:

- `row_ids`: ordered record identifiers.
- `query_ids`: ordered query-instance identifiers.
- `texts`: a record-ID/input-text mapping covering every record.
- `answers`: a query-ID mapping to record-ID/Boolean mappings.
- `discovery_order`: a label-independent permutation of all record IDs.
- `score_order`: a label-independent permutation of all record IDs.
- `certification_orders`: a full record-priority permutation for each query.

Every answer map covers all rows. IDs remain stable across input records,
query definitions, answer records, and oracle matrices. Bundled answers
are all Boolean. The general loader also accepts `null` for unavailable
custom references; the bundled-data check requires complete Boolean banks.

Discovery selects the first records in its predefined priority order.
Scoring examples are selected from records not already acquired by Discovery.
For certification, each query's priority order is restricted to its frozen
prediction frame before taking the sample. Subset runs preserve the original
relative order of retained records.

```bash
prep replay --workload data/workloads/snips/workload.json \
  --config experiments/configs/paper.json --output outputs/snips
```

`queries.json` and `records.jsonl` contain query definitions and input text.
`oracle_answers.json` and `answer_records.jsonl.gz` provide separate answer
views. `prep check-data` verifies the consistency of all bundled views.
