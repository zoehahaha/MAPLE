from pathlib import Path
import runpy

import pytest

from prep.execution import execute_plan
from prep.model import Action, Plan, Rule, Workload


ROOT = Path(__file__).resolve().parents[1]
evaluate = runpy.run_path(str(ROOT / "experiments/baselines/evaluate.py"))["evaluate"]


@pytest.mark.parametrize("outputs", [{"q": {}}, {"q": {"r": None}}, {"q": {"r": "true"}}])
def test_incomplete_predictions_cannot_pass_quality_checks(outputs):
    with pytest.raises(ValueError):
        evaluate(outputs, {"q": {"r": True}})


def test_baseline_can_read_the_bundled_reference_format():
    cells = {"q": {"positive": True, "negative": False}}
    result = evaluate(cells, {"query_ids": ["q"], "answers": cells})
    assert result["miss_pr"] == 0
    assert result["per_query"][0]["valid_references"] == 2
    assert result["per_query"][0]["precision"] == result["per_query"][0]["recall"] == 1.0


def test_certification_counts_previously_acquired_positive_answers():
    rows = tuple(f"r{i}" for i in range(102))
    known, plus, minus = rows[:100], rows[100], rows[101]
    answers = {
        "source": {row: row in (plus, minus) for row in rows},
        "target": {row: row != minus for row in rows},
    }
    workload = Workload(
        row_ids=rows, query_ids=("source", "target"),
        texts={row: "plus" if row == plus else "minus" if row == minus else "known" for row in rows},
        answers=answers, discovery_order=rows, score_order=rows,
        certification_orders={"source": rows, "target": (minus, plus, *known)},
    )
    rules = (
        Rule("positive", "source", True, "target", True, (("tok:plus", True),), select_support=100),
        Rule("negative", "source", True, "target", False, (("tok:minus", True),), select_support=100),
    )
    plan = Plan(actions={"target": tuple(Action(rule, i) for i, rule in enumerate(rules))},
                source_queries=("source",))
    result = execute_plan(workload, plan, discovery_rows=(), score_rows=())
    target = next(row for row in result.per_query if row["query"] == "target")
    assert target["known_positives"] == 100
    assert target["certificate"] == "ADMIT_PR"
    assert target["retained"] == 1
    assert result.final_outputs["target"] == answers["target"]
