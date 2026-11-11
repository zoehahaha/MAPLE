"""Frozen-plan execution, exact request accounting, certification, and fallback."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

from .certification import PAPER_ALPHA_PER_QUERY, allocate_total_n, certificate_total_sample
from .model import ExecutionResult, Plan, Workload
from .planning import owner_resolved_predictions, score_plan_evidence


def request_id(query: str, row: str) -> str:
    return f"oracle::{query}::{row}"


def execute_plan(
    workload: Workload,
    plan: Plan,
    *,
    discovery_rows: Iterable[str],
    score_rows: Iterable[str],
    alpha_query: float = PAPER_ALPHA_PER_QUERY,
    control_count: int = 0,
    error_beliefs: Mapping[str, tuple[float, float, float, float]] | None = None,
) -> ExecutionResult:
    """Execute one plan after it is frozen.

    The answer bank acts only as the offline Direct oracle.  Discovery and
    scoring example rows determine beliefs before this function is called.  Here,
    the same bank is opened in registered certification order and finally for Direct
    completion/fallback.  Request sets are unioned across roles, so an answer
    acquired for learning, a source, or certification is never charged twice.
    """
    workload.validate()
    discovery_rows, score_rows = tuple(discovery_rows), tuple(score_rows)
    roles: dict[str, set[str]] = defaultdict(set)
    discovery = set(discovery_rows)
    score = set(score_rows)
    for row in discovery:
        for query in workload.query_ids:
            roles["discovery"].add(request_id(query, row))
    for row in score:
        for query in workload.query_ids:
            roles["scoring_example"].add(request_id(query, row))
    for source in plan.source_queries:
        for row in workload.row_ids:
            roles["source"].add(request_id(source, row))
    for index in range(control_count):
        roles["control"].add(f"control::{index:03d}")
    # Sources have now been acquired. Only execution may read these answers.
    owned = owner_resolved_predictions(workload, plan)
    if error_beliefs is None:
        error_beliefs = {query: counts.posterior() for query, counts in
                        score_plan_evidence(workload, plan, score_rows).items()}

    final: dict[str, dict[str, bool | None]] = {q: {} for q in workload.query_ids}
    producer: dict[str, dict[str, str]] = {q: {} for q in workload.query_ids}
    per_query: list[dict[str, object]] = []
    prediction_frame = retained_predictions = 0

    for query in workload.query_ids:
        prepaid_rows = set(discovery) | set(score)
        if query in plan.source_queries:
            prepaid_rows.update(workload.row_ids)
        predictions = {row: value for row, value in owned[query].items() if row not in prepaid_rows}
        prediction_frame += len(predictions)
        uncovered = [row for row in workload.row_ids if row not in predictions]
        for row in uncovered:
            roles["completion"].add(request_id(query, row))
            final[query][row] = workload.answers[query][row]
            producer[query][row] = "DIRECT_COMPLETION"
        known_positives = sum(value is True for value in final[query].values())
        t_plus = sum(value[0] for value in predictions.values())
        t_minus = len(predictions) - t_plus
        a_plus, b_plus, a_minus, b_minus = error_beliefs[query]
        allocation = allocate_total_n(
            t_plus=t_plus, t_minus=t_minus, c0=known_positives,
            a_plus=a_plus, b_plus=b_plus,
            a_minus=a_minus, b_minus=b_minus,
            alpha_query=alpha_query,
        )
        n = int(allocation["n"])
        eligible = set(predictions)
        certification_rows = [row for row in workload.certification_orders[query] if row in eligible][:n]
        for row in certification_rows:
            roles["certification"].add(request_id(query, row))
        e_plus = sum(
            predictions[row][0] and workload.answers[query][row] is False
            for row in certification_rows if workload.answers[query][row] is not None
        )
        e_minus = sum(
            not predictions[row][0] and workload.answers[query][row] is True
            for row in certification_rows if workload.answers[query][row] is not None
        )
        n_plus = sum(predictions[row][0] for row in certification_rows)
        certificate = certificate_total_sample(
            t_plus=t_plus, t_minus=t_minus, n=n, n_plus=n_plus,
            e_plus=e_plus, e_minus=e_minus, c0=known_positives,
            alpha_query=alpha_query,
        )
        sampled = set(certification_rows)
        if certificate.admitted:
            for row, (prediction, rule_id) in predictions.items():
                if row in sampled:
                    final[query][row] = workload.answers[query][row]
                    producer[query][row] = "DIRECT_CERTIFICATION"
                else:
                    final[query][row] = prediction
                    producer[query][row] = f"PREDICTION::{rule_id}"
                    retained_predictions += 1
        else:
            for row in predictions:
                roles["fallback"].add(request_id(query, row))
                final[query][row] = workload.answers[query][row]
                producer[query][row] = "DIRECT_FALLBACK"

        comparable = [row for row in workload.row_ids if workload.answers[query][row] is not None]
        tp = sum(final[query][row] is True and workload.answers[query][row] is True for row in comparable)
        fp = sum(final[query][row] is True and workload.answers[query][row] is False for row in comparable)
        fn = sum(final[query][row] is False and workload.answers[query][row] is True for row in comparable)
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        per_query.append({
            "query": query,
            "known_positives": known_positives,
            "frame": len(predictions),
            "certification": len(certification_rows),
            "certificate": certificate.status,
            "retained": len(predictions) - len(certification_rows) if certificate.admitted else 0,
            "precision": precision,
            "recall": recall,
            "miss_pr": precision < 0.90 or recall < 0.90,
            "valid_references": len(comparable),
            "missing_references": len(workload.row_ids) - len(comparable),
        })

    return ExecutionResult(
        plan_id=plan.identity(),
        final_outputs=final,
        producer=producer,
        role_requests={role: frozenset(values) for role, values in roles.items()},
        retained_predictions=retained_predictions,
        prediction_frame=prediction_frame,
        per_query=tuple(per_query),
    )


def exclusive_role_counts(result: ExecutionResult) -> dict[str, int]:
    """Assign each charged identity once in the fixed first-acquisition order."""
    order = ("discovery", "scoring_example", "source", "completion", "certification", "fallback", "control")
    charged: set[str] = set()
    counts: dict[str, int] = {}
    for role in order:
        values = set(result.role_requests.get(role, ()))
        counts[role] = len(values - charged)
        charged.update(values)
    counts["total"] = len(charged)
    return counts

