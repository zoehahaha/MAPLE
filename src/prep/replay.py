from __future__ import annotations

import math
import time
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

from .certification import PAPER_ALPHA_PER_QUERY
from .discovery import Rule, progressive_discovery
from .execution import exclusive_role_counts, execute_plan
from .io import atomic_json, load_config, load_workload
from .planning import (
    PlanningContext,
    build_initial_plan,
    greedy_guard_refinement,
    scenario_identifiers,
)


def run_replay(workload_path: Path, config_path: Path, output: Path, *,
               row_limit: int | None = None, query_limit: int | None = None) -> dict[str, object]:
    workload = load_workload(workload_path)
    if row_limit is not None or query_limit is not None:
        workload = workload.subset(row_limit, query_limit)
    config = load_config(config_path)
    alpha = float(config.get("alpha_per_query", PAPER_ALPHA_PER_QUERY))
    started = time.perf_counter()
    discovery = progressive_discovery(
        workload,
        batch_fraction=float(config.get("discovery_fraction_per_round", 0.02)),
        maximum_rounds=int(config.get("discovery_maximum_rounds", 8)),
        minimum_rounds=int(config.get("discovery_minimum_rounds", 4)),
        consecutive_passes=int(config.get("discovery_consecutive_passes", 2)),
        proposal_fraction=float(config.get("discovery_proposal_fraction", 0.64)),
        informative_support=int(config.get("discovery_informative_support", 10)),
        gain_threshold=float(config.get("discovery_gain_threshold", 0.01)),
        shortlist_size=int(config.get("discovery_shortlist_size", 2)),
        conditional_settings={
            "max_base_features": int(config.get("guard_base_feature_cap", 24)),
            "single_feature_cap": int(config.get("guard_single_cap", 12)),
            "conjunction_frontier": int(config.get("guard_conjunction_frontier", 6)),
            "per_target_cap": int(config.get("guard_per_target_cap", 8)),
            "disagreement_penalty": float(config.get("guard_disagreement_penalty", 3.0)),
        },
    )
    discovered = [Rule(**value) for value in discovery["base_rules"]]
    d_rows = tuple(discovery["discovery_rows"])
    score_fraction = Fraction(str(config.get("scoring_example_fraction", 0.05)))
    if not 0 <= score_fraction <= 1:
        raise ValueError("scoring example fraction must be between zero and one")
    score_count = math.ceil(score_fraction * len(workload.row_ids))
    score_rows = tuple(row for row in workload.score_order if row not in set(d_rows))[:score_count]
    guarded = [Rule(**value) for value in discovery["conditional_rules"]]
    rules = tuple(discovered + guarded)
    scenario_ids = scenario_identifiers(
        str(config.get("scenario_namespace", "prep-public-replay")),
        int(config.get("monte_carlo_scenarios", 32)),
    )
    planning = PlanningContext(
        workload, discovery_rows=d_rows, score_rows=score_rows, scenario_ids=scenario_ids,
        alpha_query=alpha, control_count=int(config.get("control_requests", 0)),
        dirichlet_smoothing=float(config.get("source_dirichlet_smoothing", 0.5)),
        error_prior_alpha=float(config.get("error_prior_alpha", 0.5)),
        error_prior_beta=float(config.get("error_prior_beta", 0.5)),
    )
    initial, initial_forecast, initial_trace = build_initial_plan(
        workload, discovered, scenario_ids=scenario_ids, context=planning, alpha_query=alpha,
        candidate_source_cap=int(config.get("initial_source_cap", 6)),
        candidate_plan_cap=int(config.get("initial_plan_cap", 32)),
    )
    selected, final_forecast, refinement_trace = greedy_guard_refinement(
        workload, initial, rules, scenario_ids=scenario_ids, context=planning,
        shortlist_size=int(config.get("ggr_shortlist", 8)),
        complete_score_cap=int(config.get("ggr_complete_score_cap", 32)),
        move_cap=int(config.get("ggr_move_cap", 8)),
        improvement_tolerance=float(config.get("ggr_improvement_tolerance", 1e-6)),
        alpha_query=alpha,
    )
    execution = execute_plan(
        workload, selected,
        discovery_rows=d_rows,
        score_rows=score_rows,
        alpha_query=alpha,
        control_count=int(config.get("control_requests", 0)),
        error_beliefs=planning.error_beliefs(selected),
    )
    counts = exclusive_role_counts(execution)
    initial_metrics = asdict(initial_forecast)
    final_metrics = asdict(final_forecast)
    initial_metrics.pop("plan_id")
    final_metrics.pop("plan_id")
    initial_trace = [{key: value for key, value in item.items() if key != "plan_id"}
                     for item in initial_trace]
    result = {
        "status": "COMPLETE",
        "workload": workload_path.parent.name if workload_path.name == "workload.json" else workload_path.stem,
        "rows": len(workload.row_ids),
        "queries": len(workload.query_ids),
        "required_outputs": workload.required_outputs,
        "alpha_per_query": alpha,
        "discovery_stop_round": discovery["stop_round"],
        "discovery_stop_reason": discovery["stop_reason"],
        "discovery_rounds": discovery["rounds"],
        "discovery_rows": len(d_rows),
        "scoring_example_rows": len(score_rows),
        "base_rules": len(discovered),
        "conditional_rules": len(guarded),
        "scenario_count": len(scenario_ids),
        "scoring_evidence": {query: asdict(counts) for query, counts in planning.scoring_evidence(selected).items()},
        "initial_forecast": initial_metrics,
        "final_forecast": final_metrics,
        "initial_trace": initial_trace,
        "refinement_trace": refinement_trace,
        "exclusive_request_counts": counts,
        "calls_saved": workload.required_outputs - counts["total"],
        "saving_percent": 100.0 * (workload.required_outputs - counts["total"]) / workload.required_outputs,
        "prediction_frame": execution.prediction_frame,
        "retained_predictions": execution.retained_predictions,
        "observed_pr_misses": sum(bool(row["miss_pr"]) for row in execution.per_query),
        "pr_denominator": len(workload.query_ids),
        "per_query": list(execution.per_query),
        "provider_attempts": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(output / "RESULT.json", result)
    return result

