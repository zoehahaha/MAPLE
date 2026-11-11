"""Bundled workload selection and cross-file data checks."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from .io import load_workload, read_json


def dataset_entries(repo: Path) -> list[dict[str, object]]:
    return read_json(repo / "data" / "datasets.json")["workloads"]


def dataset_path(repo: Path, name: str) -> Path:
    for entry in dataset_entries(repo):
        if entry["id"] == name:
            return repo / str(entry["path"])
    choices = ", ".join(str(entry["id"]) for entry in dataset_entries(repo))
    raise ValueError(f"unknown dataset {name!r}; choose from: {choices}")


def validate_data(repo: Path) -> dict[str, object]:
    results: list[dict[str, object]] = []
    for entry in dataset_entries(repo):
        checks: dict[str, bool] = {}
        result: dict[str, object] = {"dataset": entry["id"], "checks": checks}
        try:
            path = repo / str(entry["path"])
            directory = path.parent
            workload = load_workload(path)
            rows, queries = set(workload.row_ids), set(workload.query_ids)
            checks["registered_dimensions"] = (
                len(workload.row_ids) == len(rows) == entry["rows"]
                and len(workload.query_ids) == len(queries) == entry["queries"]
                and workload.required_outputs == entry["answers"]
            )
            checks["complete_boolean_answers"] = all(
                type(value) is bool for cells in workload.answers.values() for value in cells.values()
            )
            checks["full_sampling_permutations"] = all(
                len(order) == len(rows) and set(order) == rows
                for order in (workload.discovery_order, workload.score_order, *workload.certification_orders.values())
            ) and set(workload.certification_orders) == queries
            definitions = read_json(directory / "queries.json")
            definition_ids = [query["query_id"] for query in definitions]
            checks["query_definitions"] = (
                len(definition_ids) == len(set(definition_ids)) == len(queries)
                and set(definition_ids) == queries
                and all(any(isinstance(query.get(field), str) and query[field].strip()
                            for field in ("question", "definition", "predicate", "predicate_text"))
                        for query in definitions)
            )
            with (directory / "records.jsonl").open(encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle]
            record_texts = {record["record_id"]: record["text"] for record in records}
            checks["input_records"] = len(records) == len(record_texts) == len(rows) and record_texts == workload.texts
            oracle = read_json(directory / "oracle_answers.json")
            checks["oracle_matrix"] = (
                oracle["row_ids"] == list(workload.row_ids)
                and oracle["query_ids"] == list(workload.query_ids)
                and oracle["answers"] == workload.answers
                and oracle["available_answers"] == workload.required_outputs
                and oracle["missing_answers"] == 0
            )
            seen: set[tuple[str, str]] = set()
            records_match = True
            with gzip.open(directory / "answer_records.jsonl.gz", "rt", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    pair = (record["query_id"], record["record_id"])
                    if (pair in seen or pair[0] not in queries or pair[1] not in rows
                            or record["available"] is not True
                            or record["value"] is not workload.answers[pair[0]][pair[1]]):
                        records_match = False
                    seen.add(pair)
            checks["answer_records"] = records_match and len(seen) == workload.required_outputs
            sampling = read_json(directory / "sampling_config.json")
            discovery, score = sampling["discovery_rows"], sampling["score_rows"]
            discovery_set = set(discovery)
            complement = [row for row in workload.score_order if row not in discovery_set]
            checks["registered_sampling_prefixes"] = (
                sampling["discovery_order"] == list(workload.discovery_order)
                and sampling["score_order"] == list(workload.score_order)
                and discovery == list(workload.discovery_order[:len(discovery)])
                and score == complement[:len(score)]
            )
            batches_path = directory / "query_batches.json"
            if batches_path.exists():
                batches = read_json(batches_path)
                mapping = batches["instance_to_original_query_id"]
                original_answers: dict[str, dict[str, bool | None]] = {}
                duplicates_match = True
                for query, original in mapping.items():
                    if original in original_answers and original_answers[original] != workload.answers[query]:
                        duplicates_match = False
                    original_answers[original] = dict(workload.answers[query])
                checks["query_batches"] = (
                    batches["instance_order"] == list(workload.query_ids)
                    and set(mapping) == queries and duplicates_match
                    and all(0 < count <= len(queries) for count in batches["batch_sizes"])
                )
            if entry["id"] == "20ng_synthetic":
                labels = {record["record_id"]: record["leaf_label"] for record in records}
                checks["label_oracle"] = all(
                    workload.answers[query["query_id"]][row] == (label in query["accepted_leaf_labels"])
                    for query in definitions for row, label in labels.items()
                )
            result.update(rows=len(rows), queries=len(queries), answers=workload.required_outputs)
            result["status"] = "PASS" if all(checks.values()) else "FAIL"
        except (KeyError, OSError, ValueError, TypeError) as error:
            result.update(status="FAIL", error=str(error))
        results.append(result)
    total = sum(int(result.get("answers", 0)) for result in results)
    expected_total = read_json(repo / "data" / "datasets.json")["total_answers"]
    return {"status": "PASS" if all(result["status"] == "PASS" for result in results)
            and total == expected_total else "FAIL", "total_answers": total, "workloads": results}
