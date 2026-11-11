import json
from pathlib import Path

import pytest

from prep.datasets import validate_data
from prep.io import load_config, load_workload


ROOT = Path(__file__).resolve().parents[1]


def test_bundled_inputs_are_complete_and_consistent():
    result = validate_data(ROOT)
    assert result["status"] == "PASS", result
    assert len(result["workloads"]) == 6
    assert result["total_answers"] == 734000


def test_prefix_selection_preserves_registered_orders():
    workload = load_workload(ROOT / "data/tiny_demo.json")
    subset = workload.subset(7, 2)
    assert subset.query_ids == workload.query_ids[:2]
    assert subset.discovery_order == tuple(row for row in workload.discovery_order if row in workload.row_ids[:7])
    assert subset.score_order == tuple(row for row in workload.score_order if row in workload.row_ids[:7])
    assert all(subset.answers[q][row] is workload.answers[q][row] for q in subset.query_ids for row in subset.row_ids)
    with pytest.raises(ValueError):
        workload.subset(0)


def test_config_inherits_base_and_overrides_values(tmp_path):
    (tmp_path / "base.json").write_text(json.dumps({"alpha_per_query": 0.05, "control_requests": 7}))
    (tmp_path / "child.json").write_text(json.dumps({"base_config": "base.json", "control_requests": 2}))
    assert load_config(tmp_path / "child.json") == {"alpha_per_query": 0.05, "control_requests": 2}
    (tmp_path / "base.json").write_text(json.dumps({"base_config": "child.json"}))
    with pytest.raises(ValueError, match="cycle"):
        load_config(tmp_path / "child.json")
