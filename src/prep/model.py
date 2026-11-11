from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Mapping, Sequence


def stable_id(*parts: object) -> str:
    """Return a deterministic identifier for rules and plans."""
    return sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Workload:
    row_ids: tuple[str, ...]
    query_ids: tuple[str, ...]
    texts: Mapping[str, str]
    answers: Mapping[str, Mapping[str, bool | None]]
    discovery_order: tuple[str, ...]
    score_order: tuple[str, ...]
    certification_orders: Mapping[str, tuple[str, ...]]

    def validate(self) -> None:
        rows = set(self.row_ids)
        if set(self.discovery_order) != rows or set(self.score_order) != rows:
            raise ValueError("discovery and scoring orders must each cover the row universe")
        if set(self.texts) != rows:
            raise ValueError("texts must cover every row")
        if set(self.answers) != set(self.query_ids):
            raise ValueError("answer matrix must cover every query")
        for query in self.query_ids:
            if set(self.answers[query]) != rows:
                raise ValueError(f"answer row mismatch for {query}")
            if set(self.certification_orders[query]) != rows:
                raise ValueError(f"certification order mismatch for {query}")

    @property
    def required_outputs(self) -> int:
        return len(self.row_ids) * len(self.query_ids)

    def subset(self, row_limit: int | None = None, query_limit: int | None = None) -> Workload:
        if row_limit is not None and not 1 <= row_limit <= len(self.row_ids):
            raise ValueError(f"row limit must be between 1 and {len(self.row_ids)}")
        if query_limit is not None and not 1 <= query_limit <= len(self.query_ids):
            raise ValueError(f"query limit must be between 1 and {len(self.query_ids)}")
        rows, queries = self.row_ids[:row_limit], self.query_ids[:query_limit]
        row_set = set(rows)
        result = Workload(
            row_ids=rows, query_ids=queries,
            texts={row: self.texts[row] for row in rows},
            answers={query: {row: self.answers[query][row] for row in rows} for query in queries},
            discovery_order=tuple(row for row in self.discovery_order if row in row_set),
            score_order=tuple(row for row in self.score_order if row in row_set),
            certification_orders={query: tuple(row for row in self.certification_orders[query] if row in row_set)
                          for query in queries},
        )
        result.validate()
        return result


@dataclass(frozen=True)
class Rule:
    rule_id: str
    source: str
    source_value: bool
    target: str
    prediction: bool
    guard: tuple[tuple[str, bool], ...] = ()
    fit_support: int = 0
    select_support: int = 0
    select_errors: int = 0

    @property
    def error_score(self) -> float:
        return (self.select_errors + 1.0) / (self.select_support + 2.0)


@dataclass(frozen=True)
class Action:
    rule: Rule
    priority: int


@dataclass(frozen=True)
class Plan:
    actions: Mapping[str, tuple[Action, ...]]
    source_queries: tuple[str, ...]
    plan_id: str = field(default="")

    def identity(self) -> str:
        if self.plan_id:
            return self.plan_id
        parts: list[object] = [*self.source_queries]
        for target in sorted(self.actions):
            parts.append(target)
            for action in self.actions[target]:
                parts.extend((action.priority, action.rule.rule_id))
        return stable_id(*parts)


@dataclass(frozen=True)
class ExecutionResult:
    plan_id: str
    final_outputs: Mapping[str, Mapping[str, bool | None]]
    producer: Mapping[str, Mapping[str, str]]
    role_requests: Mapping[str, frozenset[str]]
    retained_predictions: int
    prediction_frame: int
    per_query: Sequence[Mapping[str, object]]

    @property
    def unique_requests(self) -> frozenset[str]:
        result: set[str] = set()
        for requests in self.role_requests.values():
            result.update(requests)
        return frozenset(result)

