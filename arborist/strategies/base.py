"""Abstract strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Strategy(ABC):
    """Base class for tree search strategies.

    Strategies decide which nodes to expand, when to prune, and when to stop.
    Subclasses must set ``prune_threshold`` and ``plateau_window`` attributes
    (typically in ``__init__``).
    """

    prune_threshold: float
    plateau_window: int

    @abstractmethod
    def select(
        self,
        candidates: list[dict[str, Any]],
        completed: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Select which pending nodes to expand next.

        Args:
            candidates: Pending nodes available for expansion.
            completed: All completed nodes in the tree.

        Returns:
            Ordered list of nodes to expand (first = highest priority).
        """
        ...

    def should_prune(
        self,
        node: dict[str, Any],
        best_score: float,
        siblings: list[dict[str, Any]],
    ) -> tuple[bool, str]:
        """Decide whether to prune a node.

        Args:
            node: The node to consider pruning.
            best_score: The current best score across all nodes.
            siblings: Sibling nodes of this node.

        Returns:
            Tuple of (should_prune, reason).
        """
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
        """Decide whether the entire search should stop.

        Args:
            tree: The tree record.
            completed: All completed nodes.
            config: Search configuration.

        Returns:
            Tuple of (should_stop, reason).
        """
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
            best = max(
                (n["score"] for n in completed if n.get("score") is not None),
                default=0,
            )
            if best >= target_score:
                return True, f"Target score reached ({best:.4f} >= {target_score})"

        # Plateau detection
        plateau_window = config.get("plateau_window", self.plateau_window)
        if len(completed) >= plateau_window:
            scores = [n["score"] for n in completed if n.get("score") is not None]
            if scores and len(scores) > plateau_window:
                best_older = max(scores[:-plateau_window])
                best_recent = max(scores[-plateau_window:])
                if best_recent <= best_older:
                    return True, (
                        f"Plateau: no improvement in last {plateau_window} experiments"
                    )

        return False, ""
