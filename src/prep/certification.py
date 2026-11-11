"""Exact finite-population precision/recall certification used by PREP.

The public paper configuration uses a joint per-query failure budget of 0.05.
If both prediction signs occur, the budget is split equally between them; a
single present sign receives the whole budget.  The same deltas are used for
allocation and the final certificate.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np
from scipy.stats import betabinom, hypergeom

PAPER_ALPHA_PER_QUERY = 0.05
PRECISION_TARGET = 0.90
RECALL_TARGET = 0.90


@dataclass(frozen=True)
class PRCertificate:
    t_plus: int
    t_minus: int
    n: int
    n_plus: int
    e_plus: int
    e_minus: int
    c0: int
    u_plus: int
    u_minus: int
    returned_positive: int
    lower_true_positive: int
    precision_pass: bool
    recall_pass: bool
    admitted: bool
    status: str


def direction_deltas(t_plus: int, t_minus: int,
                     alpha_query: float = PAPER_ALPHA_PER_QUERY) -> tuple[float, float]:
    if not 0.0 < alpha_query < 1.0:
        raise ValueError("alpha_query must be in (0,1)")
    if t_plus > 0 and t_minus > 0:
        return alpha_query / 2.0, alpha_query / 2.0
    if t_plus > 0:
        return alpha_query, alpha_query
    if t_minus > 0:
        return alpha_query, alpha_query
    return alpha_query, alpha_query


def _validate_counts(population: int, sample: int, errors: int) -> None:
    if not (0 <= sample <= population and 0 <= errors <= sample):
        raise ValueError(
            f"invalid finite-population counts N={population}, n={sample}, e={errors}"
        )


@lru_cache(maxsize=500_000)
def residual_upper(population: int, sample: int, errors: int, delta: float) -> int:
    """Exact one-sided hypergeometric residual-error upper endpoint."""
    _validate_counts(population, sample, errors)
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0,1)")
    if population == 0 or sample == population:
        return 0
    if sample == 0:
        return population
    lo, hi = errors, population - sample + errors
    while lo < hi:
        mid = (lo + hi + 1) // 2
        cdf = float(hypergeom.cdf(errors, population, mid, sample))
        if math.isfinite(cdf) and cdf >= delta:
            lo = mid
        else:
            hi = mid - 1
    return int(lo - errors)


@lru_cache(maxsize=100_000)
def residual_upper_array(population: int, sample: int, delta: float) -> tuple[int, ...]:
    return tuple(residual_upper(population, sample, e, delta) for e in range(sample + 1))


def certificate_total_sample(
    *,
    t_plus: int,
    t_minus: int,
    n: int,
    n_plus: int,
    e_plus: int,
    e_minus: int,
    c0: int,
    alpha_query: float = PAPER_ALPHA_PER_QUERY,
) -> PRCertificate:
    values = tuple(map(int, (t_plus, t_minus, n, n_plus, e_plus, e_minus, c0)))
    t_plus, t_minus, n, n_plus, e_plus, e_minus, c0 = values
    total = t_plus + t_minus
    n_minus = n - n_plus
    if not (0 <= n <= total and 0 <= n_plus <= t_plus and 0 <= n_minus <= t_minus):
        raise ValueError("invalid whole-frame sample counts")
    if not (0 <= e_plus <= n_plus and 0 <= e_minus <= n_minus and c0 >= 0):
        raise ValueError("invalid error or known-positive counts")
    delta_plus, delta_minus = direction_deltas(t_plus, t_minus, alpha_query)
    up = residual_upper(t_plus, n_plus, e_plus, delta_plus) if t_plus else 0
    um = residual_upper(t_minus, n_minus, e_minus, delta_minus) if t_minus else 0
    returned_positive = c0 + (n_plus - e_plus) + e_minus + (t_plus - n_plus)
    lower_true_positive = returned_positive - up
    precision_pass = returned_positive == 0 or 10 * lower_true_positive >= 9 * returned_positive
    recall_denominator = lower_true_positive + um
    recall_pass = recall_denominator == 0 or 10 * lower_true_positive >= 9 * recall_denominator
    admitted = bool(precision_pass and recall_pass and 0 < n < total)
    if total == 0:
        status = "NO_CANDIDATE_FRAME"
    elif n == 0:
        status = "NO_CERTIFICATION_ALLOCATED"
    elif n == total:
        status = "CENSUS_ZERO_RETAINED_BENEFIT"
    elif admitted:
        status = "ADMIT_PR"
    elif not precision_pass and not recall_pass:
        status = "FALLBACK_PRECISION_AND_RECALL"
    elif not precision_pass:
        status = "FALLBACK_PRECISION"
    else:
        status = "FALLBACK_RECALL"
    return PRCertificate(
        t_plus=t_plus,
        t_minus=t_minus,
        n=n,
        n_plus=n_plus,
        e_plus=e_plus,
        e_minus=e_minus,
        c0=c0,
        u_plus=up,
        u_minus=um,
        returned_positive=returned_positive,
        lower_true_positive=lower_true_positive,
        precision_pass=bool(precision_pass),
        recall_pass=bool(recall_pass),
        admitted=admitted,
        status=status,
    )


def _pass_vector(
    t_plus: int,
    t_minus: int,
    n_plus: int,
    n_minus: int,
    e_plus: np.ndarray,
    e_minus: np.ndarray,
    c0: int,
    delta_plus: float,
    delta_minus: float,
) -> np.ndarray:
    up = np.asarray(residual_upper_array(t_plus, n_plus, delta_plus), dtype=np.int64)[e_plus]
    um = np.asarray(residual_upper_array(t_minus, n_minus, delta_minus), dtype=np.int64)[e_minus]
    returned_positive = c0 + t_plus - e_plus + e_minus
    lower_true_positive = returned_positive - up
    return ((10 * lower_true_positive >= 9 * returned_positive)
            & (10 * lower_true_positive >= 9 * (lower_true_positive + um)))


@lru_cache(maxsize=200_000)
def joint_pass_probability(
    t_plus: int,
    t_minus: int,
    c0: int,
    n: int,
    a_plus: float,
    b_plus: float,
    a_minus: float,
    b_minus: float,
    alpha_query: float = PAPER_ALPHA_PER_QUERY,
) -> float:
    """Exact-support P/R pass forecast under Beta-Binomial error beliefs."""
    t_plus, t_minus, c0, n = map(int, (t_plus, t_minus, c0, n))
    total = t_plus + t_minus
    if total == 0 or n <= 0 or n >= total:
        return 0.0
    delta_plus, delta_minus = direction_deltas(t_plus, t_minus, alpha_query)
    j_lo, j_hi = max(0, n - t_minus), min(n, t_plus)
    answer = 0.0
    for n_plus, p_n_plus in zip(
        range(j_lo, j_hi + 1),
        hypergeom.pmf(np.arange(j_lo, j_hi + 1), total, t_plus, n).tolist(),
    ):
        if p_n_plus <= 0.0:
            continue
        n_minus = n - n_plus
        plus_errors = np.arange(n_plus + 1, dtype=np.int64)
        plus_pmf = betabinom.pmf(plus_errors, n_plus, a_plus, b_plus)
        minus_errors = np.arange(n_minus + 1, dtype=np.int64)
        minus_pmf = betabinom.pmf(minus_errors, n_minus, a_minus, b_minus)
        passed = _pass_vector(
            t_plus,
            t_minus,
            n_plus,
            n_minus,
            plus_errors[:, None],
            minus_errors[None, :],
            c0,
            delta_plus,
            delta_minus,
        )
        answer += float(p_n_plus) * float(np.sum(plus_pmf[:, None] * minus_pmf[None, :] * passed))
    if not math.isfinite(answer):
        raise FloatingPointError("non-finite pass probability")
    return min(1.0, max(0.0, answer))


def allocate_total_n(
    *,
    t_plus: int,
    t_minus: int,
    c0: int,
    a_plus: float,
    b_plus: float,
    a_minus: float,
    b_minus: float,
    alpha_query: float = PAPER_ALPHA_PER_QUERY,
    deadline: float | None = None,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """Minimize expected certification plus fallback cost over every competitive n."""
    total = int(t_plus) + int(t_minus)
    started = time.perf_counter()
    if total <= 1:
        return {"n": 0, "pass_probability": 0.0, "expected_retained": 0.0,
                "evaluated_n": 0, "complete": True, "seconds": time.perf_counter() - started}
    best_n, best_probability, best_retained = 0, 0.0, 0.0
    evaluated: dict[int, tuple[float, float]] = {0: (0.0, 0.0), total: (0.0, 0.0)}

    def evaluate(candidate_n: int) -> bool:
        nonlocal best_n, best_probability, best_retained
        if candidate_n in evaluated:
            return True
        if deadline is not None and time.perf_counter() >= deadline:
            return False
        probability = joint_pass_probability(
            t_plus, t_minus, c0, candidate_n,
            float(a_plus), float(b_plus), float(a_minus), float(b_minus), alpha_query,
        )
        retained = (total - candidate_n) * probability
        evaluated[candidate_n] = (probability, retained)
        if retained > best_retained + 1e-12 or (
            abs(retained - best_retained) <= 1e-12 and candidate_n < best_n
        ):
            best_n, best_probability, best_retained = candidate_n, probability, retained
        if progress and len(evaluated) % 25 == 0:
            progress({"evaluated_n": len(evaluated), "best_n": best_n,
                      "best_expected_retained": best_retained, "frame": total})
        return True

    probes: list[int] = []
    candidate_n = 1
    while candidate_n < total:
        probes.append(candidate_n)
        candidate_n *= 2
    if total - 1 not in probes:
        probes.append(total - 1)
    complete = True
    for candidate_n in probes:
        if not evaluate(candidate_n):
            complete = False
            break
        if best_retained > 0 and candidate_n > total - best_retained:
            break
    competitive_max = min(total - 1, max(1, int(math.floor(total - best_retained + 1e-12))))
    if complete:
        for candidate_n in range(1, competitive_max + 1):
            if not evaluate(candidate_n):
                complete = False
                break
    return {
        "n": int(best_n),
        "pass_probability": float(best_probability),
        "expected_retained": float(best_retained),
        "expected_cost": float(total - best_retained),
        "evaluated_n": len(evaluated) - 2,
        "competitive_max_n": competitive_max,
        "complete": complete,
        "seconds": time.perf_counter() - started,
    }

