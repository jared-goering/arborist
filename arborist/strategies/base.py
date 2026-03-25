"""Abstract strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Strategy(ABC):
    """Base class for tree search strategies.

    Strategies decide which nodes to expand, when to prune, and when to stop.
    """

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

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...
