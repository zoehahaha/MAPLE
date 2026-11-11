"""Evaluate a frozen baseline output without model or network access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _matrix(value: dict) -> dict:
    if isinstance(value.get("query_ids"), list) and isinstance(value.get("answers"), dict):
        return value["answers"]
    return value


def evaluate(outputs: dict, references: dict) -> dict:
    outputs, references = _matrix(outputs), _matrix(references)
    rows = []
    for query in sorted(references):
        if not isinstance(references[query], dict):
            raise ValueError(f"reference cells must be a mapping for {query}")
        comparable = [key for key, value in references[query].items() if value is not None]
        if any(type(references[query][key]) is not bool for key in comparable):
            raise ValueError(f"reference cells must be Boolean for {query}")
        if query not in outputs or not isinstance(outputs[query], dict):
            raise ValueError(f"output is required for {query}")
        if any(type(outputs[query].get(key)) is not bool for key in comparable):
            raise ValueError(f"every comparable record requires a Boolean output for {query}")
        tp = sum(outputs[query].get(key) is True and references[query][key] is True for key in comparable)
        fp = sum(outputs[query].get(key) is True and references[query][key] is False for key in comparable)
        fn = sum(outputs[query].get(key) is False and references[query][key] is True for key in comparable)
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        rows.append({"query": query, "precision": precision, "recall": recall,
                     "miss_pr": precision < 0.90 or recall < 0.90,
                     "valid_references": len(comparable)})
    return {"per_query": rows, "miss_pr": sum(row["miss_pr"] for row in rows),
            "queries": len(rows), "provider_attempts": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    outputs = json.loads(args.frozen_output.read_text(encoding="utf-8"))
    references = json.loads(args.reference.read_text(encoding="utf-8"))
    try:
        result = evaluate(outputs, references)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

