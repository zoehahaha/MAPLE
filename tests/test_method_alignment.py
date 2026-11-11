from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pytest

import prep.discovery as discovery_module
import prep.replay as replay_module
from prep.discovery import (
    base_rule_statistics, conditional_rules, conditional_shortlist_check,
    progressive_discovery, proposal_evaluation_split, rank_base_rules, shortlist_check,
)
from prep.execution import execute_plan, exclusive_role_counts
from prep.model import Action, Plan, Rule, Workload
from prep.planning import (
    ErrorEvidence, PlanForecast, PlanningContext, build_initial_plan,
    greedy_guard_refinement, make_plan, scenario_identifiers,
)


def workload(vectors, names=("s", "t"), texts=None):
    rows = tuple(f"r{i}" for i in range(len(vectors)))
    return Workload(rows, names,
                    {row: texts[i] if texts else "plain text" for i, row in enumerate(rows)},
                    {query: {row: bool(vectors[i][j]) for i, row in enumerate(rows)}
                     for j, query in enumerate(names)}, rows, rows, {query: rows for query in names})


def context(w, discovery=("r0", "r1"), scoring=("r2", "r3"), **kwargs):
    return PlanningContext(w, discovery_rows=discovery, score_rows=scoring,
                           scenario_ids=scenario_identifiers("method-test", 3), **kwargs)


def base_rule(identifier="base", source="s", target="t", prediction=True, support=20, errors=0):
    return Rule(identifier, source, True, target, prediction, select_support=support, select_errors=errors)


def test_planning_never_reads_unacquired_answers():
    w = workload([(True, True), (False, False)] * 4)
    allowed = {"r0", "r1", "r2", "r3"}
    reads = []

    class AcquiredOnly(dict):
        def __getitem__(self, row):
            assert row in allowed, f"unacquired oracle answer was opened: {row}"
            reads.append(row)
            return super().__getitem__(row)

    guarded = replace(w, answers={q: AcquiredOnly(values) for q, values in w.answers.items()})
    frozen = context(guarded)
    rules = tuple(r for rs in rank_base_rules(base_rule_statistics(guarded, ("r0", "r1"))).values() for r in rs)
    initial, _, _ = build_initial_plan(guarded, rules, scenario_ids=frozen.scenario_ids, context=frozen)
    variant = replace(rules[0], rule_id="condition", guard=(("tok:plain", True),))
    greedy_guard_refinement(guarded, initial, (*rules, variant), scenario_ids=frozen.scenario_ids,
                            context=frozen, complete_score_cap=2, move_cap=1)
    assert reads and set(reads) <= allowed


def test_hidden_oracle_values_cannot_change_scenarios_or_forecasts():
    w = workload([(True, True), (False, False)] * 4)
    paid = {"r0", "r1", "r2", "r3"}
    altered = replace(w, answers={q: {r: value if r in paid else not value for r, value in values.items()}
                                 for q, values in w.answers.items()})
    first, second = context(w), context(altered)
    plan = make_plan([base_rule()])
    assert first.forecast(plan) == second.forecast(plan)
    assert all(np.array_equal(a, b) for a, b in zip(first.scenarios, second.scenarios))


def test_joint_scenarios_preserve_observed_cooccurrence_and_acquired_cells():
    w = workload([(True, True), (False, False)] * 6)
    frozen = context(w)
    for matrix in frozen.scenarios:
        assert np.array_equal(matrix[:, 0], matrix[:, 1])
        for row in frozen.observed_rows:
            assert matrix[frozen.row_index[row]].tolist() == [w.answers[q][row] for q in w.query_ids]
        assert not matrix.flags.writeable
    scenarios = frozen.scenarios
    frozen.forecast(make_plan([base_rule()]))
    assert frozen.scenarios is scenarios


def test_scoring_error_evidence_changes_cost_with_identical_source_scenarios():
    w = workload([(True, True), (False, False)] * 20)
    scoring = w.row_ids[20:24]
    altered = replace(w, answers={**w.answers, "t": {
        row: not value if row in scoring else value for row, value in w.answers["t"].items()}})
    first = context(w, discovery=w.row_ids[:20], scoring=scoring)
    second = context(altered, discovery=w.row_ids[:20], scoring=scoring)
    positive = base_rule()
    negative = replace(positive, rule_id="negative", source_value=False, prediction=False)
    plan = make_plan([positive, negative])
    assert all(np.array_equal(a[:, 0], b[:, 0]) for a, b in zip(first.scenarios, second.scenarios))
    a, b = first.forecast(plan), second.forecast(plan)
    qa = next(q for q in a.query_forecasts if q.query == "t")
    qb = next(q for q in b.query_forecasts if q.query == "t")
    assert qa.known_positives == qb.known_positives
    assert qa.frame == qb.frame
    assert a.expected_total < b.expected_total


def test_scoring_counts_resolve_overlaps_once_and_ignore_discovery_rule_errors():
    w = workload([(True, False, True), (True, False, True),
                  (True, True, True), (True, False, True), (False, True, False)], names=("s", "t", "u"))
    first = base_rule("first", errors=500, support=500)
    second = base_rule("second", source="u", prediction=False)
    plan = Plan({"t": (Action(first, 0), Action(second, 1))}, ("s", "u"))
    frozen = context(w)
    assert frozen.scoring_evidence(plan)["t"] == ErrorEvidence(2, 1, 0, 0)
    assert frozen.error_beliefs(plan)["t"] == (1.5, 1.5, 0.5, 0.5)
    changed = replace(w, answers={**w.answers, "t": {**w.answers["t"], "r2": False}})
    assert context(changed).error_beliefs(plan)["t"] == (2.5, 0.5, 0.5, 0.5)
    reversed_plan = Plan({"t": (Action(first, 1), Action(second, 0))}, ("s", "u"))
    assert frozen.scoring_evidence(reversed_plan)["t"] == ErrorEvidence(0, 0, 2, 1)


def test_complete_forecast_counts_known_positives_shared_sources_and_controls():
    w = workload([(True, True)] * 102)
    frozen = context(w, discovery=w.row_ids[:100], scoring=(), control_count=7)
    plan = make_plan([base_rule()])
    forecast = frozen.forecast(plan)
    target = next(q for q in forecast.query_forecasts if q.query == "t")
    assert target.known_positives == 100
    assert target.frame == 2 and target.expected_retained == pytest.approx(1)
    assert forecast.learning_calls == 200 and forecast.source_calls == 2
    assert forecast.expected_total == pytest.approx(210)
    assert forecast.expected_total == pytest.approx(w.required_outputs + 7 - sum(q.expected_retained for q in forecast.query_forecasts))
    executed = execute_plan(w, plan, discovery_rows=w.row_ids[:100], score_rows=(),
                            control_count=7, error_beliefs=frozen.error_beliefs(plan))
    assert exclusive_role_counts(executed)["total"] == pytest.approx(forecast.expected_total)


def test_base_check_rescores_previous_members_on_current_prefix():
    old = base_rule("old", prediction=False, support=20, errors=0)
    old_now = replace(old, select_errors=10)
    new = base_rule("new", prediction=False, support=20, errors=9)
    result = shortlist_check({("t", False): (old,)}, {("t", False): (new,)},
                             current_statistics={"old": old_now, "new": new})
    assert not result["passed"] and result["bad_gain_groups"] == 1


def test_base_membership_can_change_when_current_score_gain_is_small():
    old = base_rule("old", prediction=False, support=100, errors=10)
    new = base_rule("new", prediction=False, support=100, errors=9)
    result = shortlist_check({("t", False): (old,)}, {("t", False): (new,)},
                             current_statistics={"old": old, "new": new})
    assert result["passed"]


def test_new_positive_rule_blocks_a_pass_even_with_unchanged_group_mean():
    old = (base_rule("old-worst", support=100, errors=10), base_rule("old-best", support=100, errors=0))
    new = (base_rule("new-a", support=100, errors=5), base_rule("new-b", support=100, errors=5))
    stats = {r.rule_id: r for r in (*old, *new)}
    result = shortlist_check({("t", True): old}, {("t", True): new},
                             previous_statistics=stats, current_statistics=stats)
    assert result["bad_gain_groups"] == 0
    assert result["positive_emergence_groups"] == 1 and not result["passed"]


def test_newly_supported_positive_rule_blocks_a_pass_without_membership_change():
    old = base_rule("same", support=9, errors=4)
    new = replace(old, select_support=10)
    result = shortlist_check({("t", True): (old,)}, {("t", True): (new,)},
                             previous_statistics={"same": old}, current_statistics={"same": new})
    assert result["positive_emergence_groups"] == 1 and not result["passed"]


def test_conditional_check_uses_stable_membership_and_own_round_evaluation():
    a, b = base_rule("a", support=20, errors=2), base_rule("b", support=20, errors=2)
    new = (replace(b, select_support=21), replace(a, select_support=21))
    old_map = {("t", True): (a, b)}
    assert conditional_shortlist_check(old_map, {("t", True): new})["passed"]
    improved = tuple(replace(r, select_errors=0) for r in new)
    assert not conditional_shortlist_check(old_map, {("t", True): improved})["passed"]
    different = (replace(new[0], rule_id="different"), new[1])
    assert not conditional_shortlist_check(old_map, {("t", True): different})["passed"]


def test_proposal_assignment_is_fixed_for_all_discovery_prefixes():
    rows = tuple(f"r{i}" for i in range(100))
    fit, select = proposal_evaluation_split(rows)
    for length in (3, 7, 16, 64):
        a, b = proposal_evaluation_split(rows[:length])
        assert a == tuple(r for r in fit if r in rows[:length])
        assert b == tuple(r for r in select if r in rows[:length])
        assert not set(a) & set(b) and set(a) | set(b) == set(rows[:length])


def test_conditions_keep_the_signed_parent_prediction():
    w = workload([(True, False)] * 4)
    parent = base_rule(prediction=True)
    bank = conditional_rules(w, [parent], ("r0", "r1"), ("r2", "r3"), minimum_fit=1, minimum_select=1)
    assert bank and all(r.prediction is True and r.guard for r in bank)


def test_progressive_mining_cannot_read_labels_after_its_current_prefix(monkeypatch):
    w = workload([(True, True), (False, False)] * 10)
    allowed = set()
    seen = set()

    class PrefixOnly(dict):
        def __getitem__(self, row):
            assert row in allowed, f"future Discovery label was opened: {row}"
            seen.add(row)
            return super().__getitem__(row)

    w = replace(w, answers={q: PrefixOnly(values) for q, values in w.answers.items()})
    original = discovery_module.base_rule_statistics

    def acquire(workload, rows):
        rows = tuple(rows)
        allowed.clear()
        allowed.update(rows)
        return original(workload, rows)

    monkeypatch.setattr(discovery_module, "base_rule_statistics", acquire)
    result = progressive_discovery(w, batch_fraction=0.2, maximum_rounds=4)
    assert seen == set(result["discovery_rows"])
    assert len(seen) < len(w.row_ids)


def test_conditional_bank_deduplicates_trigger_sets_across_sources():
    w = workload([(True, True, True)] * 4, names=("s", "t", "u"))
    parents = [base_rule("one"), base_rule("two", source="u")]
    bank = conditional_rules(w, parents, ("r0", "r1"), ("r2", "r3"), minimum_fit=1, minimum_select=1)
    assert len(bank) == 1


@pytest.mark.parametrize("labels,expected_stop", [
    ({4: "a", 6: "b", 8: "b", 10: "b"}, 5),
    ({4: "a", 6: "b", 8: "b", 10: "c", 12: "c", 14: "c"}, 7),
])
def test_every_round_updates_conditions_and_requires_both_stopping_checks(monkeypatch, labels, expected_stop):
    w = workload([(True, False)] * 20)
    assignments = []

    def bank(workload, parents, fit, select, **kwargs):
        assignments.append((set(fit), set(select)))
        key = labels[len(fit) + len(select)]
        return [Rule(key, "s", True, "t", False, (("tok:" + key, True),), select_support=20)]

    monkeypatch.setattr(discovery_module, "conditional_rules", bank)
    result = progressive_discovery(w, batch_fraction=0.1, maximum_rounds=8, minimum_rounds=3,
                                    consecutive_passes=2, informative_support=1)
    assert result["stop_round"] == expected_stop
    assert len(assignments) == expected_stop - 1
    assert result["rounds"][0]["conditional_bank_size"] == 0
    assert not result["rounds"][2]["conditional_check"]["passed"]
    for (old_fit, old_select), (new_fit, new_select) in zip(assignments, assignments[1:]):
        assert old_fit <= new_fit and old_select <= new_select


def test_initial_candidates_include_direct_and_use_base_rules_only():
    w = workload([(True, True), (False, False)] * 4)
    frozen = context(w)
    base = base_rule()
    _, _, trace = build_initial_plan(w, [base], scenario_ids=frozen.scenario_ids, context=frozen)
    scored = [entry for entry in trace if entry.get("phase") == "COMPLETE_SCORE"]
    assert scored and scored[0]["source_count"] == 0
    with pytest.raises(ValueError, match="base rules only"):
        build_initial_plan(w, [replace(base, guard=(("tok:plain", True),))],
                            scenario_ids=frozen.scenario_ids, context=frozen)


def test_initial_search_can_add_rules_served_by_an_already_open_source(monkeypatch):
    w = workload([(True, True), (False, False)] * 4)
    frozen = context(w)
    positive = base_rule()
    negative = replace(positive, rule_id="negative", source_value=False, prediction=False)
    seen = []
    original = frozen.forecast

    def score(plan):
        seen.append(plan)
        return original(plan)

    monkeypatch.setattr(frozen, "forecast", score)
    build_initial_plan(w, [positive, negative], scenario_ids=frozen.scenario_ids,
                        context=frozen, rules_per_target=1)
    assert any(not plan.source_queries for plan in seen)
    assert any(plan.source_queries == ("s",) and len(plan.actions["t"]) == 2 for plan in seen)


def test_refinement_searches_beyond_first_batch_and_freezes_structure():
    w = workload([(True, True)] * 4)
    base = base_rule()
    bad = replace(base, rule_id="bad", guard=(("tok:plain", True),))
    good = replace(base, rule_id="good", guard=(("tok:text", True),))
    initial = make_plan([base])

    class Costs:
        alpha_query = 0.05

        def __init__(self):
            self.scored = []

        def refinement_proxy(self, plan):
            rule = plan.actions["t"][0].rule.rule_id
            return {"base": 0, "bad": 10, "good": 1}[rule]

        def forecast(self, plan):
            rule = plan.actions["t"][0].rule.rule_id
            self.scored.append(rule)
            return PlanForecast(0, plan.identity(), 0, 0, 0, 0, {"base": 10, "bad": 12, "good": 8}[rule], ())

    costs = Costs()
    selected, _, _ = greedy_guard_refinement(w, initial, [base, bad, good], scenario_ids=("one",),
                                            context=costs, shortlist_size=1, complete_score_cap=2, move_cap=1)
    assert costs.scored == ["base", "bad", "good"]
    assert selected.source_queries == initial.source_queries
    assert selected.actions["t"][0].rule.rule_id == "good"
    assert selected.actions["t"][0].priority == initial.actions["t"][0].priority
    assert len(selected.actions["t"]) == len(initial.actions["t"])


def test_replay_passes_base_bank_to_initial_stage_and_frozen_variants_to_refinement(tmp_path, monkeypatch):
    w = workload([(True, True)] * 12)
    path = tmp_path / "workload.json"
    path.write_text(json.dumps(asdict(w)))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"monte_carlo_scenarios": 1, "scoring_example_fraction": 0.1,
                                  "ggr_complete_score_cap": 1, "ggr_move_cap": 1}))
    base = base_rule()
    variant = replace(base, rule_id="variant", guard=(("tok:plain", True),))
    monkeypatch.setattr(replay_module, "progressive_discovery", lambda *a, **k: {
        "base_rules": [asdict(base)], "conditional_rules": [asdict(variant)],
        "discovery_rows": ("r0", "r1"), "stop_round": 2, "stop_reason": "TEST", "rounds": []})
    stages = []
    original_initial = replay_module.build_initial_plan
    original_refine = replay_module.greedy_guard_refinement

    def initial(workload, rules, **kwargs):
        assert all(not rule.guard for rule in rules)
        stages.append("base-only")
        return original_initial(workload, rules, **kwargs)

    def refine(workload, plan, rules, **kwargs):
        assert any(rule.guard for rule in rules)
        stages.append("frozen-conditions")
        return original_refine(workload, plan, rules, **kwargs)

    monkeypatch.setattr(replay_module, "build_initial_plan", initial)
    monkeypatch.setattr(replay_module, "greedy_guard_refinement", refine)
    result = replay_module.run_replay(path, config, tmp_path / "output")
    assert result["status"] == "COMPLETE" and stages == ["base-only", "frozen-conditions"]
    assert result["scoring_evidence"]["t"]["positive_predictions"] == 2
