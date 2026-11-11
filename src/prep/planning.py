"""Shared joint scenarios, scoring-example evidence, and paper Algorithm 2."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache
from hashlib import sha256
from typing import Iterable, Mapping, Sequence

import numpy as np

from .certification import PAPER_ALPHA_PER_QUERY, allocate_total_n
from .features import guard_fires, input_features
from .model import Action, Plan, Rule, Workload, stable_id


@dataclass(frozen=True)
class ErrorEvidence:
    positive_predictions: int = 0
    positive_errors: int = 0
    negative_predictions: int = 0
    negative_errors: int = 0

    def posterior(self, a0: float = 0.5, b0: float = 0.5) -> tuple[float, float, float, float]:
        return (a0 + self.positive_errors, b0 + self.positive_predictions - self.positive_errors,
                a0 + self.negative_errors, b0 + self.negative_predictions - self.negative_errors)


@dataclass(frozen=True)
class QueryForecast:
    query: str
    frame: float
    certification: float
    expected_fallback: float
    expected_retained: float
    expected_direct: float
    known_positives: float = 0.0


@dataclass(frozen=True)
class PlanForecast:
    learning_calls: int
    plan_id: str
    source_calls: int
    mandatory_direct: float
    expected_certification: float
    expected_fallback: float
    expected_total: float
    query_forecasts: tuple[QueryForecast, ...]
    control_calls: int = 0


def scenario_identifiers(namespace: str, count: int) -> tuple[str, ...]:
    return tuple(f"{namespace}|scenario|{index:03d}" for index in range(count))


def _scenario_rng(identifier: str) -> np.random.Generator:
    seed = int.from_bytes(sha256(identifier.encode("utf-8")).digest()[:8], "big")
    return np.random.default_rng(seed)


@lru_cache(maxsize=100_000)
def _allocation(t_plus: int, t_minus: int, c0: int, a_plus: float, b_plus: float,
                a_minus: float, b_minus: float, alpha_query: float) -> tuple[int, float, float]:
    result = allocate_total_n(t_plus=t_plus, t_minus=t_minus, c0=c0,
                              a_plus=a_plus, b_plus=b_plus, a_minus=a_minus, b_minus=b_minus,
                              alpha_query=alpha_query)
    if not result["complete"]:
        raise RuntimeError("a complete certification forecast is required")
    return int(result["n"]), float(result["pass_probability"]), float(result["expected_retained"])


def _validate_plan(plan: Plan, query_ids: Iterable[str]) -> None:
    queries, sources = set(query_ids), set(plan.source_queries)
    if len(sources) != len(plan.source_queries) or not sources <= queries or not set(plan.actions) <= queries:
        raise ValueError("plan sources and targets must be registered, unique query IDs")
    for target, actions in plan.actions.items():
        if any(action.rule.source not in sources or action.rule.target != target for action in actions):
            raise ValueError("every rule must use an acquired source and its declared target")


class PlanningContext:
    """Frozen evidence and common scenarios. No unacquired oracle bank is stored.

    A symmetric Dirichlet prior smooths the categorical distribution over
    observed joint Boolean vectors. Scenarios jointly sample all unqueried
    query components from this fixed posterior-predictive distribution.
    """

    def __init__(self, workload: Workload, *, discovery_rows: Iterable[str], score_rows: Iterable[str],
                 scenario_ids: Sequence[str], alpha_query: float = PAPER_ALPHA_PER_QUERY,
                 control_count: int = 0, dirichlet_smoothing: float = 0.5,
                 error_prior_alpha: float = 0.5, error_prior_beta: float = 0.5):
        self.row_ids, self.query_ids = tuple(workload.row_ids), tuple(workload.query_ids)
        self.discovery_rows, self.score_rows = tuple(discovery_rows), tuple(score_rows)
        if set(self.discovery_rows) & set(self.score_rows):
            raise ValueError("scoring examples must be outside the Discovery prefix")
        self.observed_rows = (*self.discovery_rows, *self.score_rows)
        if (not self.observed_rows or len(set(self.observed_rows)) != len(self.observed_rows)
                or not set(self.observed_rows) <= set(self.row_ids)):
            raise ValueError("planning requires distinct acquired records within the workload")
        if not scenario_ids or len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError("planning requires a positive number of distinct shared scenarios")
        if min(dirichlet_smoothing, error_prior_alpha, error_prior_beta) <= 0.0:
            raise ValueError("forecast priors must be positive")
        if not 0.0 < alpha_query < 1.0 or control_count < 0:
            raise ValueError("invalid certification budget or control request count")
        self.alpha_query, self.control_count = alpha_query, control_count
        self.error_prior_alpha, self.error_prior_beta = error_prior_alpha, error_prior_beta
        self.scenario_ids = tuple(scenario_ids)
        self.row_index = {row: index for index, row in enumerate(self.row_ids)}
        self.query_index = {query: index for index, query in enumerate(self.query_ids)}
        self.features = {row: input_features(workload.texts[row]) for row in self.row_ids}
        # This is the only oracle read performed by planning: D union S.
        vectors = [tuple(workload.answers[query][row] for query in self.query_ids) for row in self.observed_rows]
        if not all(type(value) is bool for vector in vectors for value in vector):
            raise ValueError("acquired planning evidence must contain Boolean oracle answers")
        self._observed = np.asarray(vectors, dtype=bool)
        self._observed.setflags(write=False)
        counts = Counter(vectors)
        patterns = sorted(counts)
        self._patterns = np.asarray(patterns, dtype=bool)
        parameters = np.asarray([counts[pattern] + dirichlet_smoothing for pattern in patterns])
        self._pattern_probabilities = parameters / parameters.sum()
        self._pattern_probabilities.setflags(write=False)
        self._paid_indices = np.asarray([self.row_index[row] for row in self.observed_rows], dtype=int)
        self._paid_mask = np.zeros(len(self.row_ids), dtype=bool)
        self._paid_mask[self._paid_indices] = True
        observed_index = {row: index for index, row in enumerate(self.observed_rows)}
        self._score_indices = np.asarray([observed_index[row] for row in self.score_rows], dtype=int)
        self._scenarios: tuple[np.ndarray, ...] | None = None
        self._guards: dict[tuple, np.ndarray] = {}
        self._evidence: dict[str, dict[str, ErrorEvidence]] = {}
        self._forecasts: dict[str, PlanForecast] = {}
        self._proxies: dict[str, tuple[int, int]] = {}

    @property
    def scenarios(self) -> tuple[np.ndarray, ...]:
        if self._scenarios is None:
            unknown = np.flatnonzero(~self._paid_mask)
            scenarios = []
            for identifier in self.scenario_ids:
                rng = _scenario_rng(identifier + "|joint-source-vectors")
                matrix = np.empty((len(self.row_ids), len(self.query_ids)), dtype=bool)
                matrix[unknown] = self._patterns[rng.choice(len(self._patterns), size=len(unknown), p=self._pattern_probabilities)]
                matrix[self._paid_indices] = self._observed
                matrix.setflags(write=False)
                scenarios.append(matrix)
            self._scenarios = tuple(scenarios)
        return self._scenarios

    def _guard(self, guard: tuple) -> np.ndarray:
        key = tuple(guard)
        if key not in self._guards:
            self._guards[key] = np.asarray([guard_fires(key, self.features[row]) for row in self.row_ids], dtype=bool)
        return self._guards[key]

    def _predictions(self, plan: Plan, matrix: np.ndarray, rows: Sequence[str], *,
                     exclude_paid: bool) -> dict[str, np.ndarray]:
        _validate_plan(plan, self.query_ids)
        indices = np.asarray([self.row_index[row] for row in rows], dtype=int)
        eligible = ~self._paid_mask[indices] if exclude_paid else np.ones(len(rows), dtype=bool)
        result = {}
        sources = set(plan.source_queries)
        for query in self.query_ids:
            predicted = np.full(len(rows), -1, dtype=np.int8)
            if query not in sources:
                for action in sorted(plan.actions.get(query, ()), key=lambda item: (item.priority, item.rule.rule_id)):
                    rule = action.rule
                    triggers = ((predicted < 0) & eligible & self._guard(rule.guard)[indices]
                                & (matrix[:, self.query_index[rule.source]] == rule.source_value))
                    predicted[triggers] = int(rule.prediction)
            result[query] = predicted
        return result

    def scoring_evidence(self, plan: Plan) -> dict[str, ErrorEvidence]:
        key = plan.identity()
        if key not in self._evidence:
            matrix = self._observed[self._score_indices]
            predicted = self._predictions(plan, matrix, self.score_rows, exclude_paid=False)
            evidence = {}
            for query, values in predicted.items():
                truth = matrix[:, self.query_index[query]]
                positive, negative = values == 1, values == 0
                evidence[query] = ErrorEvidence(int(positive.sum()), int((positive & ~truth).sum()),
                                                int(negative.sum()), int((negative & truth).sum()))
            self._evidence[key] = evidence
        return dict(self._evidence[key])

    def error_beliefs(self, plan: Plan) -> dict[str, tuple[float, float, float, float]]:
        return {query: counts.posterior(self.error_prior_alpha, self.error_prior_beta)
                for query, counts in self.scoring_evidence(plan).items()}

    def _proxy_counts(self, plan: Plan) -> tuple[int, int]:
        key = plan.identity()
        if key not in self._proxies:
            predicted = self._predictions(plan, self._observed, self.observed_rows, exclude_paid=False)
            coverage = errors = 0
            for query, values in predicted.items():
                selected = values >= 0
                coverage += int(selected.sum())
                errors += int((selected & (values != self._observed[:, self.query_index[query]])).sum())
            self._proxies[key] = coverage, errors
        return self._proxies[key]

    def proxy_coverage(self, plan: Plan) -> float:
        coverage, _ = self._proxy_counts(plan)
        return coverage * (len(self.row_ids) - len(self.observed_rows)) / len(self.observed_rows)

    def refinement_proxy(self, plan: Plan, penalty: float = 3.0) -> float:
        coverage, errors = self._proxy_counts(plan)
        return (coverage - penalty * errors) * (len(self.row_ids) - len(self.observed_rows)) / len(self.observed_rows)

    def forecast(self, plan: Plan) -> PlanForecast:
        key = plan.identity()
        if key in self._forecasts:
            return self._forecasts[key]
        beliefs = self.error_beliefs(plan)
        totals = {query: np.zeros(7, dtype=float) for query in self.query_ids}
        sources = set(plan.source_queries)
        for matrix in self.scenarios:
            predicted = self._predictions(plan, matrix, self.row_ids, exclude_paid=True)
            for query, values in predicted.items():
                frame_mask = values >= 0
                t_plus, t_minus = int((values == 1).sum()), int((values == 0).sum())
                frame = t_plus + t_minus
                # Simulate positives among future source/uncovered oracle calls;
                # acquired target answers are already fixed in every scenario.
                known = int(matrix[~frame_mask, self.query_index[query]].sum())
                n, _, retained = _allocation(t_plus, t_minus, known, *beliefs[query], self.alpha_query)
                fallback = max(0.0, frame - n - retained)
                mandatory = 0 if query in sources else len(self.row_ids) - len(self.observed_rows) - frame
                totals[query] += (frame, n, fallback, retained, mandatory + n + fallback, known, mandatory)
        forecasts = []
        mandatory = 0.0
        for query, sums in totals.items():
            frame, n, fallback, retained, direct, known, queried = sums / len(self.scenario_ids)
            forecasts.append(QueryForecast(query, float(frame), float(n), float(fallback), float(retained),
                                            float(direct), float(known)))
            mandatory += float(queried)
        learning = len(self.observed_rows) * len(self.query_ids)
        source_calls = len(sources) * (len(self.row_ids) - len(self.observed_rows))
        certification = sum(item.certification for item in forecasts)
        fallback = sum(item.expected_fallback for item in forecasts)
        result = PlanForecast(learning, key, source_calls, mandatory, certification, fallback,
                              learning + source_calls + mandatory + certification + fallback + self.control_count,
                              tuple(forecasts), self.control_count)
        self._forecasts[key] = result
        return result


def score_plan_evidence(workload: Workload, plan: Plan, score_rows: Iterable[str]) -> dict[str, ErrorEvidence]:
    rows = tuple(score_rows)
    if not rows:
        return {query: ErrorEvidence() for query in workload.query_ids}
    return PlanningContext(workload, discovery_rows=(), score_rows=rows,
                           scenario_ids=("scoring-evidence",)).scoring_evidence(plan)


def rule_mask(workload: Workload, rule: Rule) -> dict[str, bool]:
    """Execution-only helper: the source must already have been acquired."""
    return {row: workload.answers[rule.source][row] is rule.source_value
            and guard_fires(rule.guard, input_features(workload.texts[row])) for row in workload.row_ids}


def owner_resolved_predictions(workload: Workload, plan: Plan) -> dict[str, dict[str, tuple[bool, str]]]:
    """Execution routing after source acquisition; planning uses simulated matrices."""
    _validate_plan(plan, workload.query_ids)
    result: dict[str, dict[str, tuple[bool, str]]] = {query: {} for query in workload.query_ids}
    masks: dict[str, dict[str, bool]] = {}
    for query, actions in plan.actions.items():
        if query in plan.source_queries:
            continue
        for action in sorted(actions, key=lambda item: (item.priority, item.rule.rule_id)):
            rule = action.rule
            if rule.rule_id not in masks:
                masks[rule.rule_id] = rule_mask(workload, rule)
            for row in workload.row_ids:
                if row not in result[query] and masks[rule.rule_id][row]:
                    result[query][row] = rule.prediction, rule.rule_id
    return result


def _context(workload: Workload, scenario_ids: Sequence[str], paid_rows: Iterable[str],
             score_rows: Iterable[str], alpha_query: float, control_count: int,
             context: PlanningContext | None) -> PlanningContext:
    if context is not None:
        if context.alpha_query != alpha_query:
            raise ValueError("all candidate plans must use the frozen certification budget")
        return context
    score_rows = tuple(score_rows)
    scoring = set(score_rows)
    discovery = tuple(row for row in paid_rows if row not in scoring)
    return PlanningContext(workload, discovery_rows=discovery, score_rows=score_rows,
                           scenario_ids=scenario_ids, alpha_query=alpha_query, control_count=control_count)


def forecast_plan(workload: Workload, plan: Plan, rules: Mapping[str, Rule], *,
                  scenario_ids: Sequence[str], paid_rows: Iterable[str] = (), score_rows: Iterable[str] = (),
                  alpha_query: float = PAPER_ALPHA_PER_QUERY, control_count: int = 0,
                  context: PlanningContext | None = None) -> PlanForecast:
    return _context(workload, scenario_ids, paid_rows, score_rows, alpha_query, control_count, context).forecast(plan)


def make_plan(rules: Iterable[Rule]) -> Plan:
    actions: dict[str, list[Action]] = defaultdict(list)
    for rule in sorted(rules, key=lambda item: (item.target, item.error_score, -item.select_support, item.rule_id)):
        actions[rule.target].append(Action(rule, len(actions[rule.target])))
    return Plan(actions={query: tuple(items) for query, items in actions.items()},
                source_queries=tuple(sorted({action.rule.source for items in actions.values() for action in items})))


def build_initial_plan(
    workload: Workload, rules: Sequence[Rule], *, scenario_ids: Sequence[str],
    paid_rows: Iterable[str] = (), score_rows: Iterable[str] = (), candidate_source_cap: int = 6,
    rules_per_target: int = 2, candidate_plan_cap: int = 32,
    alpha_query: float = PAPER_ALPHA_PER_QUERY, control_count: int = 0,
    context: PlanningContext | None = None,
) -> tuple[Plan, PlanForecast, list[dict[str, object]]]:
    """Build a proxy-guided base-only P0, then fully score its retained plans."""
    if any(rule.guard for rule in rules):
        raise ValueError("initial plans must be constructed from base rules only")
    if candidate_plan_cap < 1 or candidate_source_cap < 0 or rules_per_target < 1:
        raise ValueError("invalid initial-plan search bounds")
    context = _context(workload, scenario_ids, paid_rows, score_rows, alpha_query, control_count, context)
    by_source: dict[str, list[Rule]] = defaultdict(list)
    for rule in rules:
        by_source[rule.source].append(rule)
    sources = sorted(by_source, key=lambda source: (
        min(rule.error_score for rule in by_source[source]),
        -sum(rule.select_support for rule in by_source[source]), source))[:candidate_source_cap]
    chosen: list[Rule] = []
    current = make_plan(chosen)
    pool = [current]
    seen = {current.identity()}
    trace: list[dict[str, object]] = []
    while len(pool) < candidate_plan_cap:
        baseline = context.proxy_coverage(current)
        proposals = []
        for source in sources:
            if source in current.source_queries:
                continue
            targets: dict[str, list[Rule]] = defaultdict(list)
            for rule in by_source[source]:
                targets[rule.target].append(rule)
            additions = [rule for target in sorted(targets) for rule in sorted(targets[target], key=lambda r: (
                r.error_score, -r.select_support, r.rule_id))[:rules_per_target]]
            candidates = chosen + additions
            proposals.append(("OPEN_SOURCE", source, candidates, make_plan(candidates)))
        chosen_ids = {rule.rule_id for rule in chosen}
        for rule in rules:
            if rule.source in current.source_queries and rule.rule_id not in chosen_ids:
                candidates = chosen + [rule]
                proposals.append(("ADD_RULE", rule.source, candidates, make_plan(candidates)))
        ranked = [(context.proxy_coverage(plan) - baseline, plan.identity(), kind, source, candidates, plan)
                  for kind, source, candidates, plan in proposals if plan.identity() not in seen]
        if not ranked:
            break
        gain, _, kind, source, candidates, proposal = min(ranked, key=lambda item: (-item[0], item[1]))
        if gain <= 1e-12:
            break
        chosen, current = candidates, proposal
        seen.add(current.identity())
        pool.append(current)
        trace.append({"move": kind, "phase": "PROXY", "source": source, "proxy_gain": gain})
    scored = []
    for index, plan in enumerate(pool):
        forecast = context.forecast(plan)
        scored.append((forecast.expected_total, len(plan.source_queries), plan.identity(), plan, forecast))
        trace.append({"phase": "COMPLETE_SCORE", "candidate": index,
                      "forecast_total": forecast.expected_total, "source_count": len(plan.source_queries)})
    winner = min(scored, key=lambda item: item[:3])
    return winner[3], winner[4], trace


def greedy_guard_refinement(
    workload: Workload, initial: Plan, all_rules: Sequence[Rule], *, scenario_ids: Sequence[str],
    paid_rows: Iterable[str] = (), score_rows: Iterable[str] = (), shortlist_size: int = 8,
    complete_score_cap: int = 32, move_cap: int = 8, improvement_tolerance: float = 1e-6,
    alpha_query: float = PAPER_ALPHA_PER_QUERY, control_count: int = 0,
    context: PlanningContext | None = None,
) -> tuple[Plan, PlanForecast, list[dict[str, object]]]:
    """Score successive proxy batches; apply only fully scored improvements."""
    if shortlist_size < 1 or min(complete_score_cap, move_cap, improvement_tolerance) < 0:
        raise ValueError("invalid condition-refinement bounds")
    context = _context(workload, scenario_ids, paid_rows, score_rows, alpha_query, control_count, context)
    current, current_score = initial, context.forecast(initial)
    trace: list[dict[str, object]] = []
    complete_scores = accepted = 0
    while accepted < move_cap and complete_scores < complete_score_cap:
        baseline = context.refinement_proxy(current)
        neighbors = []
        for target in sorted(current.actions):
            for slot, action in enumerate(current.actions[target]):
                parent = action.rule
                compatible = [rule for rule in all_rules if
                              (rule.source, rule.source_value, rule.target, rule.prediction)
                              == (parent.source, parent.source_value, parent.target, parent.prediction)]
                if not any(not rule.guard for rule in compatible):
                    compatible.append(Rule("B-" + stable_id(parent.source, parent.source_value, parent.target, parent.prediction)[:18],
                                           parent.source, parent.source_value, parent.target, parent.prediction))
                seen_guards = set()
                for replacement in compatible:
                    if replacement.guard == parent.guard or replacement.guard in seen_guards:
                        continue
                    seen_guards.add(replacement.guard)
                    updated = dict(current.actions)
                    actions = list(updated[target])
                    actions[slot] = Action(replacement, action.priority)
                    updated[target] = tuple(actions)
                    candidate = Plan(actions=updated, source_queries=current.source_queries)
                    gain = context.refinement_proxy(candidate) - baseline
                    neighbors.append((-gain, candidate.identity(), candidate, parent.rule_id, replacement.rule_id))
        neighbors.sort(key=lambda item: item[:2])
        if not neighbors:
            trace.append({"stop": "NO_LEGAL_MOVES"})
            break
        best_plan, best_score = current, current_score
        offset = 0
        while offset < len(neighbors) and complete_scores < complete_score_cap:
            size = min(shortlist_size, complete_score_cap - complete_scores)
            batch = neighbors[offset:offset + size]
            offset += len(batch)
            for _, _, candidate, old_rule, new_rule in batch:
                forecast = context.forecast(candidate)
                complete_scores += 1
                trace.append({"move": "SCORE_SINGLE_SLOT", "old_rule": old_rule, "new_rule": new_rule,
                              "forecast_total": forecast.expected_total, "accepted": False})
                if (forecast.expected_total, candidate.identity()) < (best_score.expected_total, best_plan.identity()):
                    best_plan, best_score = candidate, forecast
            if current_score.expected_total - best_score.expected_total > improvement_tolerance:
                break
        if current_score.expected_total - best_score.expected_total <= improvement_tolerance:
            trace.append({"stop": "COMPLETE_SCORE_CAP" if complete_scores >= complete_score_cap else "NO_GAIN_IN_NEIGHBORHOOD",
                          "complete_scores": complete_scores})
            break
        current, current_score = best_plan, best_score
        accepted += 1
        trace.append({"move": "ACCEPT_SINGLE_SLOT", "forecast_total": current_score.expected_total, "accepted": True})
    if complete_scores >= complete_score_cap:
        trace.append({"stop": "COMPLETE_SCORE_CAP", "complete_scores": complete_scores})
    elif accepted >= move_cap:
        trace.append({"stop": "MOVE_CAP", "accepted_moves": accepted})
    return current, current_score, trace
