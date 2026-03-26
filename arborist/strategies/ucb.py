"""Upper Confidence Bound (UCB) strategy — MCTS-style exploration/exploitation."""

from __future__ import annotations

import math
from typing import Any

from arborist.strategies.base import Strategy


class UCBStrategy(Strategy):
    """UCB1 strategy balancing exploitation of high-scoring nodes with exploration.

    Score = exploitation (normalized node score) + C * sqrt(ln(N) / n_i)

    Where:
        - C = exploration weight (default sqrt(2))
        - N = total completed nodes
        - n_i = visits to this node's branch
    """

    def __init__(
        self,
        exploration_weight: float = math.sqrt(2),
        prune_threshold: float = 0.5,
        min_samples: int = 3,
        plateau_window: int = 20,
    ) -> None:
        self.exploration_weight = exploration_weight
        self.prune_threshold = prune_threshold
        self.min_samples = min_samples
        self.plateau_window = plateau_window

    def select(
        self,
        candidates: list[dict[str, Any]],
        completed: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        total_visits = max(len(completed), 1)
        best_score = max((n["score"] for n in completed if n["score"] is not None), default=0)
        min_score = min((n["score"] for n in completed if n["score"] is not None), default=0)
        score_range = best_score - min_score if best_score != min_score else 1.0

        # Count visits per branch (by root ancestor)
        branch_visits: dict[str | None, int] = {}
        for node in completed:
            root = node.get("parent_id")
            branch_visits[root] = branch_visits.get(root, 0) + 1

        def ucb_score(node: dict[str, Any]) -> float:
            # Exploitation: use parent score if available, else 0
            parent_id = node.get("parent_id")
            parent_score = 0.0
            for c in completed:
                if c["id"] == parent_id and c["score"] is not None:
                    parent_score = (c["score"] - min_score) / score_range
                    break

            # Exploration
            n_i = branch_visits.get(parent_id, 1)
            exploration = self.exploration_weight * math.sqrt(
                math.log(total_visits) / n_i
            )

            return parent_score + exploration

        return sorted(candidates, key=ucb_score, reverse=True)

    def should_prune(
        self,
        node: dict[str, Any],
        best_score: float,
        siblings: list[dict[str, Any]],
    ) -> tuple[bool, str]:
        if node["score"] is None:
            return False, ""

        # Need minimum samples before pruning
        completed_siblings = [s for s in siblings if s["score"] is not None]
        if len(completed_siblings) < self.min_samples:
            return False, ""

        # Prune if scoring below threshold of best
        if best_score > 0 and node["score"] < best_score * self.prune_threshold:
            return True, (
                f"Score {node['score']:.4f} < {self.prune_threshold:.0%} of best "
                f"({best_score:.4f})"
            )

        return False, ""
