"""Best-first strategy — always expand the highest-scoring node."""

from __future__ import annotations

from typing import Any

from arborist.strategies.base import Strategy


class BestFirstStrategy(Strategy):
    """Greedy strategy that always expands the node with the highest parent score.

    Good for exploitation-heavy searches where you want to drill into the best
    known region quickly.
    """

    def __init__(
        self,
        prune_threshold: float = 0.3,
        plateau_window: int = 15,
    ) -> None:
        self.prune_threshold = prune_threshold
        self.plateau_window = plateau_window

    def select(
        self,
        candidates: list[dict[str, Any]],
        completed: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        # Build parent score lookup
        score_map = {n["id"]: n["score"] for n in completed if n["score"] is not None}

        def sort_key(node: dict[str, Any]) -> float:
            parent_id = node.get("parent_id")
            if parent_id and parent_id in score_map:
                return score_map[parent_id]
            return 0.0

        return sorted(candidates, key=sort_key, reverse=True)

    def should_prune(
        self,
        node: dict[str, Any],
        best_score: float,
        siblings: list[dict[str, Any]],
    ) -> tuple[bool, str]:
        if node["score"] is None or best_score <= 0:
            return False, ""

        if node["score"] < best_score * self.prune_threshold:
            return True, (
                f"Score {node['score']:.4f} < {self.prune_threshold:.0%} of best "
                f"({best_score:.4f})"
            )
        return False, ""

    def should_terminate(
        self,
        tree: dict[str, Any],
        completed: list[dict[str, Any]],
        config: dict[str, Any],
    ) -> tuple[bool, str]:
        max_experiments = config.get("max_experiments")
        if max_experiments and len(completed) >= max_experiments:
            return True, f"Reached max experiments ({max_experiments})"

        budget_usd = config.get("budget_usd")
        if budget_usd:
            total_cost = sum(n.get("cost_usd") or 0 for n in completed)
            if total_cost >= budget_usd:
                return True, f"Budget exhausted (${total_cost:.2f} >= ${budget_usd:.2f})"

        target_score = config.get("target_score")
        if target_score:
            best = max((n["score"] for n in completed if n["score"] is not None), default=0)
            if best >= target_score:
                return True, f"Target score reached ({best:.4f} >= {target_score})"

        plateau_window = config.get("plateau_window", self.plateau_window)
        if len(completed) >= plateau_window:
            scores = [n["score"] for n in completed if n["score"] is not None]
            if scores and len(scores) > plateau_window:
                best_older = max(scores[:-plateau_window])
                best_recent = max(scores[-plateau_window:])
                if best_recent <= best_older:
                    return True, f"Plateau: no improvement in last {plateau_window} experiments"

        return False, ""
