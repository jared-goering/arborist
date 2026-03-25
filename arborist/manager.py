"""Tree manager — DAG state management and node lifecycle."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from arborist.store import Store

logger = logging.getLogger(__name__)


@dataclass
class BranchContext:
    """Context passed to executors and mutators for a given node."""

    goal: str
    depth: int
    parent_config: dict[str, Any] | None = None
    parent_results: dict[str, Any] | None = None
    parent_score: float | None = None
    sibling_scores: list[float] = field(default_factory=list)


def _parse_json(value: str | dict | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


class TreeManager:
    """Manages tree lifecycle: creation, node operations, state queries."""

    def __init__(self, store: Store) -> None:
        self.store = store

    def create_tree(
        self,
        goal: str,
        strategy: str = "ucb",
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        tree = self.store.create_tree(goal=goal, strategy=strategy, config=config)
        logger.info("Created tree %s: %s", tree["id"], goal)
        return tree

    def add_seed_nodes(
        self,
        tree_id: str,
        seed_configs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        nodes = []
        for cfg in seed_configs:
            node = self.store.create_node(
                tree_id=tree_id,
                config=cfg,
                parent_id=None,
                depth=0,
                hypothesis=f"Seed config: {_summarize_config(cfg)}",
            )
            nodes.append(node)
            logger.debug("Added seed node %s", node["id"])
        return nodes

    def add_child_nodes(
        self,
        parent_id: str,
        configs: list[dict[str, Any]],
        hypotheses: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        parent = self.store.get_node(parent_id)
        if not parent:
            raise ValueError(f"Parent node {parent_id} not found")

        nodes = []
        for i, cfg in enumerate(configs):
            hypothesis = hypotheses[i] if hypotheses and i < len(hypotheses) else None
            node = self.store.create_node(
                tree_id=parent["tree_id"],
                config=cfg,
                parent_id=parent_id,
                depth=parent["depth"] + 1,
                hypothesis=hypothesis,
            )
            nodes.append(node)
        return nodes

    def mark_running(self, node_id: str) -> None:
        self.store.update_node(node_id, status="running")

    def mark_completed(
        self,
        node_id: str,
        results: dict[str, Any],
        score: float | None = None,
        cost_usd: float | None = None,
        duration_ms: int | None = None,
    ) -> None:
        self.store.update_node(
            node_id,
            status="completed",
            results=results,
            score=score,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        logger.debug("Node %s completed, score=%.4f", node_id, score or 0)

    def mark_failed(self, node_id: str, error: str) -> None:
        self.store.update_node(
            node_id,
            status="failed",
            error=error,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        logger.warning("Node %s failed: %s", node_id, error)

    def prune_node(self, node_id: str, reason: str) -> None:
        self.store.update_node(
            node_id,
            status="pruned",
            pruned=1,
            prune_reason=reason,
        )
        logger.info("Pruned node %s: %s", node_id, reason)

    def get_branch_context(self, node: dict[str, Any], goal: str) -> BranchContext:
        parent_config = None
        parent_results = None
        parent_score = None

        if node["parent_id"]:
            parent = self.store.get_node(node["parent_id"])
            if parent:
                parent_config = _parse_json(parent["config"])
                parent_results = _parse_json(parent["results"])
                parent_score = parent["score"]

        siblings = self.store.get_siblings(node["id"])
        sibling_scores = [s["score"] for s in siblings if s["score"] is not None]

        return BranchContext(
            goal=goal,
            depth=node["depth"],
            parent_config=parent_config,
            parent_results=parent_results,
            parent_score=parent_score,
            sibling_scores=sibling_scores,
        )

    def get_pending_nodes(self, tree_id: str) -> list[dict[str, Any]]:
        return self.store.get_tree_nodes(tree_id, status="pending")

    def get_completed_nodes(self, tree_id: str) -> list[dict[str, Any]]:
        return self.store.get_tree_nodes(tree_id, status="completed")

    def get_best_score(self, tree_id: str) -> float | None:
        best = self.store.get_best_node(tree_id)
        return best["score"] if best else None

    def complete_tree(self, tree_id: str) -> None:
        self.store.update_tree(
            tree_id,
            status="completed",
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        logger.info("Tree %s completed", tree_id)

    def fail_tree(self, tree_id: str) -> None:
        self.store.update_tree(tree_id, status="failed")

    def pause_tree(self, tree_id: str) -> None:
        self.store.update_tree(tree_id, status="paused")


def _summarize_config(config: dict[str, Any], max_len: int = 60) -> str:
    parts = []
    for k, v in config.items():
        parts.append(f"{k}={v}")
    s = ", ".join(parts)
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s
