"""Progressive base/conditional rule mining (paper Algorithm 1)."""
from __future__ import annotations

import itertools
import math
from collections import defaultdict
from dataclasses import asdict
from fractions import Fraction
from typing import Iterable, Mapping

from .features import guard_fires, input_features
from .model import Rule, Workload, stable_id


def disagreement(rule: Rule, smoothing: float = 0.5) -> float:
    return (rule.select_errors + smoothing) / (rule.select_support + 2.0 * smoothing)


def base_rule_statistics(workload: Workload, rows: Iterable[str]) -> list[Rule]:
    rows = tuple(rows)
    rules: list[Rule] = []
    for source in workload.query_ids:
        for source_value in (False, True):
            for target in workload.query_ids:
                if target == source:
                    continue
                eligible = [row for row in rows
                            if workload.answers[source][row] is source_value
                            and workload.answers[target][row] is not None]
                positives = sum(workload.answers[target][row] is True for row in eligible)
                for prediction in (False, True):
                    errors = len(eligible) - positives if prediction else positives
                    rules.append(Rule(
                        rule_id="B-" + stable_id(source, source_value, target, prediction)[:18],
                        source=source, source_value=source_value, target=target, prediction=prediction,
                        fit_support=len(eligible), select_support=len(eligible), select_errors=errors,
                    ))
    return rules


def smoothed_disagreement(rule: Rule) -> float:
    return disagreement(rule)


def rank_base_rules(rules: Iterable[Rule], top_k: int = 2,
                    smoothing: float = 0.5) -> dict[tuple[str, bool], tuple[Rule, ...]]:
    groups: dict[tuple[str, bool], list[Rule]] = defaultdict(list)
    for rule in rules:
        groups[rule.target, rule.prediction].append(rule)
    return {group: tuple(sorted(part, key=lambda rule: (
        disagreement(rule, smoothing), -rule.select_support, len(rule.guard), rule.rule_id,
    ))[:top_k]) for group, part in groups.items()}


def _mean(rules: Iterable[Rule], smoothing: float) -> float | None:
    rules = tuple(rules)
    return sum(disagreement(rule, smoothing) for rule in rules) / len(rules) if rules else None


def shortlist_check(
    previous: Mapping[tuple[str, bool], tuple[Rule, ...]],
    current: Mapping[tuple[str, bool], tuple[Rule, ...]],
    *,
    previous_statistics: Mapping[str, Rule] | None = None,
    current_statistics: Mapping[str, Rule] | None = None,
    informative_support: int = 10,
    gain_threshold: float = 0.01,
    smoothing: float = 0.5,
) -> dict[str, object]:
    """Base check: rescore both shortlists on the current prefix, Eq. (6)."""
    old_stats = dict(previous_statistics or {r.rule_id: r for rs in previous.values() for r in rs})
    new_stats = dict(current_statistics or {
        **{r.rule_id: r for rs in previous.values() for r in rs},
        **{r.rule_id: r for rs in current.values() for r in rs},
    })
    details: list[dict[str, object]] = []
    informative = bad_gain = positive_emergence = 0
    for group in sorted(set(previous) | set(current)):
        old, new = previous.get(group, ()), current.get(group, ())
        old_current = tuple(new_stats[r.rule_id] for r in old)
        new_current = tuple(new_stats[r.rule_id] for r in new)
        supported = any(r.select_support >= informative_support for r in (*old_current, *new_current))
        old_mean, new_mean = _mean(old_current, smoothing), _mean(new_current, smoothing)
        gain = old_mean - new_mean if old_mean is not None and new_mean is not None else None
        emergence = False
        if group[1]:
            old_ids = {rule.rule_id for rule in old}
            weakest = max((disagreement(r, smoothing) for r in old_current), default=None)
            for rule in new_current:
                if rule.select_support < informative_support:
                    continue
                old_rule = old_stats.get(rule.rule_id)
                newly_entered = rule.rule_id not in old_ids and (
                    weakest is None or weakest - disagreement(rule, smoothing) > gain_threshold)
                newly_supported = (old_rule is not None
                                   and old_rule.select_support < informative_support
                                   and disagreement(old_rule, smoothing) - disagreement(rule, smoothing) > gain_threshold)
                emergence = emergence or newly_entered or newly_supported
        informative += int(supported)
        bad_gain += int(supported and (gain is None or gain > gain_threshold))
        positive_emergence += int(emergence)
        details.append({"target": group[0], "prediction": group[1], "informative": supported,
                        "mean_change": gain, "positive_emergence": emergence})
    return {"passed": informative > 0 and bad_gain == 0 and positive_emergence == 0,
            "informative_groups": informative, "bad_gain_groups": bad_gain,
            "positive_emergence_groups": positive_emergence, "details": details}


def conditional_shortlist_check(
    previous: Mapping[tuple[str, bool], tuple[Rule, ...]],
    current: Mapping[tuple[str, bool], tuple[Rule, ...]],
    *, informative_support: int = 10, gain_threshold: float = 0.01, smoothing: float = 0.5,
) -> dict[str, object]:
    """Conditional check: stable membership and each round's own evaluation data."""
    details = []
    for group in sorted(set(previous) | set(current)):
        old, new = previous.get(group, ()), current.get(group, ())
        same = bool(old) and sorted(r.rule_id for r in old) == sorted(r.rule_id for r in new)
        old_mean, new_mean = _mean(old, smoothing), _mean(new, smoothing)
        gain = old_mean - new_mean if old_mean is not None and new_mean is not None else None
        supported = bool(new) and all(r.select_support >= informative_support for r in new)
        passed = same and supported and gain is not None and gain <= gain_threshold
        details.append({"target": group[0], "prediction": group[1], "same_members": same,
                        "supported": supported, "mean_change": gain, "passed": passed})
    return {"passed": bool(details) and all(item["passed"] for item in details), "details": details}


def proposal_evaluation_split(order: Iterable[str], fraction: float = 0.64) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """A prefix-stable, label-independent assignment, fixed before reading labels."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("proposal fraction must be between zero and one")
    weight = Fraction(str(fraction))
    proposal, evaluation = [], []
    for index, row in enumerate(order):
        destination = proposal if math.floor((index + 1) * weight) > math.floor(index * weight) else evaluation
        destination.append(row)
    return tuple(proposal), tuple(evaluation)


def conditional_rules(
    workload: Workload, base_rules: Iterable[Rule], fit_rows: Iterable[str], select_rows: Iterable[str],
    *, max_base_features: int = 24, single_feature_cap: int = 12,
    conjunction_frontier: int = 6, per_target_cap: int = 8,
    minimum_fit: int = 3, minimum_select: int = 2, minimum_select_rate: float = 0.10,
    disagreement_penalty: float = 3.0, bank_smoothing: float = 1.0,
    features: Mapping[str, frozenset[str]] | None = None,
) -> list[Rule]:
    """Generate conditions for fixed (source, b, target, v); never change v."""
    if (bank_smoothing <= 0 or disagreement_penalty <= 0 or min(minimum_fit, minimum_select) < 1
            or min(max_base_features, single_feature_cap, conjunction_frontier, per_target_cap) < 0
            or not 0 <= minimum_select_rate <= 1):
        raise ValueError("invalid conditional-bank settings")
    fit_rows, select_rows = tuple(fit_rows), tuple(select_rows)
    if set(fit_rows) & set(select_rows):
        raise ValueError("proposal and evaluation records must be disjoint")
    features = features if features is not None else {row: input_features(workload.texts[row])
                                                     for row in (*fit_rows, *select_rows)}
    candidates: list[tuple[Rule, tuple[str, ...]]] = []
    prepared: dict[tuple[str, bool, str], tuple[list[str], list[str], list[str]]] = {}
    for parent in base_rules:
        key = parent.source, parent.source_value, parent.target
        if key not in prepared:
            fit = [row for row in fit_rows if workload.answers[parent.source][row] is parent.source_value
                   and workload.answers[parent.target][row] is not None]
            evaluation = [row for row in select_rows if workload.answers[parent.source][row] is not None
                          and workload.answers[parent.target][row] is not None]
            vocabulary = sorted({feature for row in fit for feature in features[row]})
            prepared[key] = fit, evaluation, vocabulary
        fit, evaluation, vocabulary = prepared[key]
        if len(fit) < minimum_fit or len(evaluation) < minimum_select:
            continue
        scored = []
        for feature in vocabulary:
            for present in (True, False):
                firing = [row for row in fit if (feature in features[row]) is present]
                if len(firing) < minimum_fit:
                    continue
                errors = sum(workload.answers[parent.target][row] is not parent.prediction for row in firing)
                scored.append((len(firing) - disagreement_penalty * errors, (feature, present)))
        literals = [literal for _, literal in sorted(scored, key=lambda item: (-item[0], item[1]))[:max_base_features]]
        guards = [(literal,) for literal in literals[:single_feature_cap]]
        guards.extend(tuple(sorted(pair)) for pair in itertools.combinations(literals[:conjunction_frontier], 2)
                      if pair[0][0] != pair[1][0])
        for guard in guards:
            fitted = [row for row in fit if guard_fires(guard, features[row])]
            firing = tuple(row for row in evaluation
                           if workload.answers[parent.source][row] is parent.source_value
                           and guard_fires(guard, features[row]))
            required = max(minimum_select, math.ceil(minimum_select_rate * len(evaluation)))
            if len(fitted) < minimum_fit or len(firing) < required:
                continue
            errors = sum(workload.answers[parent.target][row] is not parent.prediction for row in firing)
            rule = Rule("G-" + stable_id(parent.rule_id, guard, parent.prediction)[:18],
                        parent.source, parent.source_value, parent.target, parent.prediction, guard,
                        len(fitted), len(firing), errors)
            candidates.append((rule, firing))
    candidates.sort(key=lambda item: (disagreement(item[0], bank_smoothing), -item[0].select_support,
                                      len(item[0].guard), item[0].rule_id))
    retained: list[Rule] = []
    seen: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    counts: dict[str, int] = defaultdict(int)
    for rule, behavior in candidates:
        if counts[rule.target] >= per_target_cap or behavior in seen[rule.target]:
            continue
        seen[rule.target].add(behavior)
        counts[rule.target] += 1
        retained.append(rule)
    return retained


def progressive_discovery(
    workload: Workload, *, batch_fraction: float = 0.02, maximum_rounds: int = 8,
    minimum_rounds: int = 4, consecutive_passes: int = 2, proposal_fraction: float = 0.64,
    informative_support: int = 10, gain_threshold: float = 0.01, shortlist_size: int = 2,
    stopping_smoothing: float = 0.5, conditional_settings: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not 0.0 < batch_fraction <= 1.0 or min(maximum_rounds, minimum_rounds, consecutive_passes, shortlist_size) < 1:
        raise ValueError("Discovery requires positive rounds, shortlist size, and a fraction in (0, 1]")
    if informative_support < 1 or gain_threshold < 0 or stopping_smoothing <= 0:
        raise ValueError("invalid Discovery stopping settings")
    workload.validate()
    step = Fraction(str(batch_fraction)) * len(workload.row_ids)
    schedule = tuple(min(len(workload.row_ids), math.ceil(round_id * step))
                     for round_id in range(1, maximum_rounds + 1))
    proposal_order, _ = proposal_evaluation_split(workload.discovery_order, proposal_fraction)
    proposal_set = set(proposal_order)
    features = {row: input_features(workload.texts[row]) for row in workload.row_ids}
    previous_base: dict = {}
    previous_conditional: dict = {}
    previous_statistics: dict = {}
    rounds = []
    passes = 0
    stop_reason = "MAXIMUM_ROUND_CAP"
    stop_round = maximum_rounds
    for round_id in range(1, maximum_rounds + 1):
        rows = workload.discovery_order[:schedule[round_id - 1]]
        full_base = base_rule_statistics(workload, rows)
        statistics = {rule.rule_id: rule for rule in full_base}
        base = rank_base_rules(full_base, shortlist_size, stopping_smoothing)
        proposal = tuple(row for row in rows if row in proposal_set)
        evaluation = tuple(row for row in rows if row not in proposal_set)
        bank = conditional_rules(workload, full_base, proposal, evaluation, features=features,
                                 **dict(conditional_settings or {})) if round_id >= 2 else []
        conditional = rank_base_rules(bank, shortlist_size, stopping_smoothing)
        base_check = conditional_check = None
        if round_id >= 3:
            base_check = shortlist_check(previous_base, base, previous_statistics=previous_statistics,
                                         current_statistics=statistics, informative_support=informative_support,
                                         gain_threshold=gain_threshold, smoothing=stopping_smoothing)
            conditional_check = conditional_shortlist_check(previous_conditional, conditional,
                                                            informative_support=informative_support,
                                                            gain_threshold=gain_threshold,
                                                            smoothing=stopping_smoothing)
            passes = passes + 1 if base_check["passed"] and conditional_check["passed"] else 0
        rounds.append({"round": round_id, "rows": len(rows), "proposal_rows": len(proposal),
                       "evaluation_rows": len(evaluation), "conditional_bank_size": len(bank),
                       "base_check": base_check, "conditional_check": conditional_check,
                       "consecutive_passes": passes})
        if round_id >= max(3, minimum_rounds) and passes >= consecutive_passes:
            stop_round, stop_reason = round_id, "CONSECUTIVE_STABILITY_CHECKS"
            break
        previous_base, previous_conditional, previous_statistics = base, conditional, statistics
    return {"stop_round": stop_round, "stop_reason": stop_reason, "discovery_rows": tuple(rows),
            "proposal_rows": proposal, "evaluation_rows": evaluation,
            "base_rules": [asdict(rule) for group in sorted(base) for rule in base[group]],
            "conditional_rules": [asdict(rule) for rule in bank], "scheduled_rows": schedule, "rounds": rounds}
