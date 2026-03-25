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

        # Plateau detection
        plateau_window = config.get("plateau_window", self.plateau_window)
        if len(completed) >= plateau_window:
            scores = [n["score"] for n in completed if n["score"] is not None]
            if scores:
                scores_sorted_by_time = scores  # Already in creation order
                recent = scores_sorted_by_time[-plateau_window:]
                older = scores_sorted_by_time[:-plateau_window]
                if older:
                    best_older = max(older)
                    best_recent = max(recent)
                    if best_recent <= best_older:
                        return True, (
                            f"Plateau: no improvement in last {plateau_window} experiments"
                        )

        return False, ""
