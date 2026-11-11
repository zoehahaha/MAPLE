from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from .model import Workload


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path: Path, _parents: frozenset[Path] = frozenset()) -> dict[str, Any]:
    path = path.resolve()
    if path in _parents:
        raise ValueError(f"configuration inheritance cycle at {path.name}")
    config = dict(read_json(path))
    if "late_score_fraction" in config:
        config.setdefault("scoring_example_fraction", config.pop("late_score_fraction"))
    base = config.pop("base_config", None)
    if base is None:
        return config
    inherited = load_config(path.parent / base, _parents | {path})
    return {**inherited, **config}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_workload(path: Path) -> Workload:
    raw = read_json(path)
    certification_orders = raw.get("certification_orders", raw.get("audit_orders"))
    if certification_orders is None:
        raise ValueError("workload requires certification_orders")
    workload = Workload(
        row_ids=tuple(raw["row_ids"]),
        query_ids=tuple(raw["query_ids"]),
        texts=raw["texts"],
        answers={q: {row: value for row, value in values.items()}
                 for q, values in raw["answers"].items()},
        discovery_order=tuple(raw["discovery_order"]),
        score_order=tuple(raw["score_order"]),
        certification_orders={q: tuple(values) for q, values in certification_orders.items()},
    )
    workload.validate()
    return workload

