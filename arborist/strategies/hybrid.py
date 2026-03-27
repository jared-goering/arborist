"""Hybrid strategy — oscillates between UCB exploration and greedy hill-climbing."""

from __future__ import annotations

import logging
import math
from typing import Any

from arborist.strategies.base import Strategy
from arborist.strategies.ucb import UCBStrategy

logger = logging.getLogger(__name__)


class HybridStrategy(Strategy):
    """Two-phase strategy combining tree exploration (UCB) with greedy exploitation.

    Phase 1 (EXPLORE): Uses UCB1 to explore the search tree broadly.
    Runs until ``explore_plateau`` consecutive experiments show no improvement,
    then transitions to exploit phase starting from the best node found.

    Phase 2 (EXPLOIT): Greedy hill-climb from a single root node.
    Only selects children/descendants of the exploit root, preferring those
    with the highest parent scores. Runs until ``exploit_plateau`` consecutive
    experiments show no improvement.

    Phases cycle until budget is exhausted. Previously exploited roots are
    tracked to avoid re-climbing the same hill.
    """

    def __init__(
        self,
        explore_plateau: int = 8,
        exploit_plateau: int = 5,
        exploration_weight: float = math.sqrt(2),
        prune_threshold: float = 0.5,
        plateau_window: int = 50,
    ) -> None:
        self.explore_plateau = explore_plateau
        self.exploit_plateau = exploit_plateau
        self.exploration_weight = exploration_weight
        self.prune_threshold = prune_threshold
        self.plateau_window = plateau_window

        # Compose UCB for explore phase
        self._ucb = UCBStrategy(exploration_weight=exploration_weight)

        # Phase state
        self._phase: str = "explore"
        self._phase_best_score: float = float("-inf")
        self._no_improve_count: int = 0
        self._last_completed_count: int = 0

        # Exploit tracking
        self._exploit_root_id: str | None = None
        self._exploited_roots: set[str] = set()

    def _update_phase(self, completed: list[dict[str, Any]]) -> None:
        """Check for plateau and transition phases if needed."""
        n_completed = len(completed)
        n_new = n_completed - self._last_completed_count
        if n_new <= 0:
            return

        # Check new completions for improvement
        recent = completed[-n_new:]
        for node in recent:
            score = node.get("score")
            if score is not None and score > self._phase_best_score:
                self._phase_best_score = score
                self._no_improve_count = 0
            else:
                self._no_improve_count += 1

        self._last_completed_count = n_completed

        # Check for phase transition
        if self._phase == "explore":
            if self._no_improve_count >= self.explore_plateau:
                self._transition_to_exploit(completed)
        elif self._phase == "exploit":
            if self._no_improve_count >= self.exploit_plateau:
                self._transition_to_explore(completed)

    def _transition_to_exploit(self, completed: list[dict[str, Any]]) -> None:
        """Switch from explore to exploit phase."""
        # Find best node not already exploited
        scored = [
            n for n in completed
            if n.get("score") is not None and n["id"] not in self._exploited_roots
        ]
        if not scored:
            # All good nodes already exploited — stay in explore
            self._no_improve_count = 0
            logger.info("Hybrid: no unexploited nodes available, staying in explore")
            return

        scored.sort(key=lambda n: n["score"], reverse=True)
        root = scored[0]

        self._phase = "exploit"
        self._exploit_root_id = root["id"]
        self._exploited_roots.add(root["id"])
        self._no_improve_count = 0
        self._phase_best_score = root["score"]

        logger.info(
            "Hybrid: EXPLORE → EXPLOIT (root=%s, score=%.4f, exploited=%d)",
            root["id"][:8],
            root["score"],
            len(self._exploited_roots),
        )

    def _transition_to_explore(self, completed: list[dict[str, Any]]) -> None:
        """Switch from exploit back to explore phase."""
        best_score = max(
            (n["score"] for n in completed if n.get("score") is not None),
            default=0,
        )
        self._phase = "explore"
        self._exploit_root_id = None
        self._no_improve_count = 0
        self._phase_best_score = best_score

        logger.info(
            "Hybrid: EXPLOIT → EXPLORE (best=%.4f, exploited_roots=%d)",
            best_score,
            len(self._exploited_roots),
        )

    def _is_descendant_of(
        self,
        node: dict[str, Any],
        root_id: str,
        completed: list[dict[str, Any]],
    ) -> bool:
        """Check if node is a child or descendant of root_id."""
        # Build parent lookup from completed nodes
        parent_lookup: dict[str, str | None] = {
            n["id"]: n.get("parent_id") for n in completed
        }
        # Also include the candidate itself
        current_parent = node.get("parent_id")
        visited: set[str | None] = set()
        while current_parent and current_parent not in visited:
            if current_parent == root_id:
                return True
            visited.add(current_parent)
            current_parent = parent_lookup.get(current_parent)
        return False

    def select(
        self,
        candidates: list[dict[str, Any]],
        completed: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        # Update phase based on new completions
        self._update_phase(completed)

        if self._phase == "explore":
            return self._ucb.select(candidates, completed)

        # EXPLOIT: filter to descendants of exploit root
        assert self._exploit_root_id is not None

        exploit_candidates = [
            c for c in candidates
            if c.get("parent_id") == self._exploit_root_id
            or self._is_descendant_of(c, self._exploit_root_id, completed)
        ]

        if not exploit_candidates:
            # No candidates in exploit subtree — fall back to explore selection
            logger.info("Hybrid: no exploit candidates, using UCB fallback")
            return self._ucb.select(candidates, completed)

        # Greedy: sort by parent score descending
        parent_scores: dict[str, float] = {
            n["id"]: n["score"]
            for n in completed
            if n.get("score") is not None
        }

        def greedy_key(node: dict[str, Any]) -> float:
            parent_id = node.get("parent_id")
            return parent_scores.get(parent_id, 0.0) if parent_id else 0.0

        return sorted(exploit_candidates, key=greedy_key, reverse=True)

    @property
    def phase(self) -> str:
        """Current phase: 'explore' or 'exploit'."""
        return self._phase
