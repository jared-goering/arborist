"""Breadth-first strategy — level-by-level exploration."""

from __future__ import annotations

from typing import Any

from arborist.strategies.base import Strategy


class BreadthFirstStrategy(Strategy):
    """Explores the tree level by level, expanding all nodes at depth d before d+1.

    Good for systematic exploration where you want full coverage of each level
    before going deeper.
    """

    def __init__(
        self,
        prune_threshold: float = 0.25,
        plateau_window: int = 20,
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
        # Sort by depth (shallowest first), then by creation time
        return sorted(candidates, key=lambda n: (n["depth"], n["created_at"]))

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

        return False, ""
