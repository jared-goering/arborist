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
