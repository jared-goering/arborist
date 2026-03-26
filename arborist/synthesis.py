"""Synthesis engine — cross-branch insight extraction and report generation."""

from __future__ import annotations

import json
from typing import Any

from arborist.store import Store
from arborist.utils import parse_json_field


class SearchResults:
    """Results from a completed tree search."""

    def __init__(self, tree_id: str, store: Store) -> None:
        self.tree_id = tree_id
        self._store = store

    @property
    def best(self) -> dict[str, Any] | None:
        """Best scoring node."""
        node = self._store.get_best_node(self.tree_id)
        if node:
            node["config"] = parse_json_field(node["config"])
            node["results"] = parse_json_field(node["results"])
        return node

    def top_k(self, k: int = 5) -> list[dict[str, Any]]:
        """Top k scoring nodes."""
        nodes = self._store.get_top_nodes(self.tree_id, k=k)
        for n in nodes:
            n["config"] = parse_json_field(n["config"])
            n["results"] = parse_json_field(n["results"])
        return nodes

    @property
    def insights(self) -> list[dict[str, Any]]:
        """All insights discovered during search."""
        return self._store.get_insights(self.tree_id)

    def report(self) -> str:
        """Generate a markdown report of the search."""
        return generate_report(self.tree_id, self._store)


def generate_report(tree_id: str, store: Store) -> str:
    """Generate a markdown report for a completed search tree."""
    tree = store.get_tree(tree_id)
    if not tree:
        return f"# Error\n\nTree {tree_id} not found."

    all_nodes = store.get_tree_nodes(tree_id)
    completed = [n for n in all_nodes if n["status"] == "completed"]
    failed = [n for n in all_nodes if n["status"] == "failed"]
    pruned = [n for n in all_nodes if n["pruned"]]
    insights = store.get_insights(tree_id)
    best = store.get_best_node(tree_id)
    top5 = store.get_top_nodes(tree_id, k=5)

    # Cost and duration
    total_cost = sum(n.get("cost_usd") or 0 for n in all_nodes)
    total_duration_ms = sum(n.get("duration_ms") or 0 for n in all_nodes)
    max_depth = max((n["depth"] for n in all_nodes), default=0)

    lines = [
        f"# Arborist Search Report",
        f"",
        f"**Goal:** {tree['goal']}",
        f"**Strategy:** {tree['strategy']}",
        f"**Status:** {tree['status']}",
        f"**Tree ID:** `{tree_id}`",
        f"",
        f"## Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total experiments | {len(all_nodes)} |",
        f"| Completed | {len(completed)} |",
        f"| Failed | {len(failed)} |",
        f"| Pruned | {len(pruned)} |",
        f"| Max depth | {max_depth} |",
        f"| Total cost | ${total_cost:.4f} |",
        f"| Total duration | {total_duration_ms / 1000:.1f}s |",
        f"",
    ]

    # Best result
    if best:
        best_config = parse_json_field(best["config"])
        best_results = parse_json_field(best["results"])
        lines.extend([
            f"## Best Result",
            f"",
            f"**Score:** {best['score']:.4f}",
            f"**Node:** `{best['id']}` (depth {best['depth']})",
            f"",
            f"**Config:**",
            f"```json",
            json.dumps(best_config, indent=2),
            f"```",
            f"",
        ])
        if best_results:
            lines.extend([
                f"**Results:**",
                f"```json",
                json.dumps(best_results, indent=2),
                f"```",
                f"",
            ])

    # Top 5
    if top5:
        lines.extend([
            f"## Top 5 Results",
            f"",
            f"| Rank | Node | Score | Depth | Config |",
            f"|------|------|-------|-------|--------|",
        ])
        for i, node in enumerate(top5, 1):
            cfg = parse_json_field(node["config"])
            cfg_str = json.dumps(cfg) if cfg else ""
            if len(cfg_str) > 60:
                cfg_str = cfg_str[:57] + "..."
            lines.append(
                f"| {i} | `{node['id']}` | {node['score']:.4f} | {node['depth']} | `{cfg_str}` |"
            )
        lines.append("")

    # Insights
    if insights:
        lines.extend([
            f"## Insights",
            f"",
        ])
        for insight in insights:
            conf = f" (confidence: {insight['confidence']:.2f})" if insight.get("confidence") else ""
            lines.append(f"- **{insight['type']}**: {insight['content']}{conf}")
        lines.append("")

    # Tree structure (simple text visualization)
    lines.extend([
        f"## Tree Structure",
        f"",
        f"```",
    ])
    _render_tree(lines, all_nodes)
    lines.extend([
        f"```",
        f"",
    ])

    return "\n".join(lines)


def _render_tree(lines: list[str], nodes: list[dict[str, Any]]) -> None:
    """Render a simple text tree of nodes."""
    # Build parent->children map
    children_map: dict[str | None, list[dict[str, Any]]] = {}
    for n in nodes:
        parent = n.get("parent_id")
        if parent not in children_map:
            children_map[parent] = []
        children_map[parent].append(n)

    roots = children_map.get(None, [])
    for root in roots:
        _render_node(lines, root, children_map, prefix="", is_last=True)


def _render_node(
    lines: list[str],
    node: dict[str, Any],
    children_map: dict[str | None, list[dict[str, Any]]],
    prefix: str,
    is_last: bool,
) -> None:
    connector = "└── " if is_last else "├── "
    score_str = f" score={node['score']:.4f}" if node["score"] is not None else ""
    status = node["status"]
    if node["pruned"]:
        status = "pruned"
    lines.append(f"{prefix}{connector}[{node['id']}] {status}{score_str}")

    children = children_map.get(node["id"], [])
    child_prefix = prefix + ("    " if is_last else "│   ")
    for i, child in enumerate(children):
        _render_node(lines, child, children_map, child_prefix, i == len(children) - 1)


def extract_basic_insights(tree_id: str, store: Store) -> list[dict[str, Any]]:
    """Extract basic insights from completed nodes."""
    completed = store.get_tree_nodes(tree_id, status="completed")
    if len(completed) < 2:
        return []

    insights = []
    scores = [n["score"] for n in completed if n["score"] is not None]
    if not scores:
        return []

    best_score = max(scores)
    worst_score = min(scores)
    avg_score = sum(scores) / len(scores)

    # Find the best node
    best_nodes = [n for n in completed if n["score"] == best_score]
    best_node = best_nodes[0]
    best_config = parse_json_field(best_node["config"])

    insights.append(store.create_insight(
        tree_id=tree_id,
        source_node_ids=[best_node["id"]],
        insight_type="discovery",
        content=(
            f"Best configuration achieved score {best_score:.4f}. "
            f"Config: {json.dumps(best_config)}"
        ),
        confidence=0.9,
    ))

    # Score distribution insight
    if len(scores) >= 3:
        insights.append(store.create_insight(
            tree_id=tree_id,
            source_node_ids=[n["id"] for n in completed[:5]],
            insight_type="convergence",
            content=(
                f"Score range: {worst_score:.4f} to {best_score:.4f} "
                f"(avg: {avg_score:.4f}) across {len(scores)} experiments."
            ),
            confidence=0.95,
        ))

    # Check for convergence — top nodes have similar configs
    top_nodes = store.get_top_nodes(tree_id, k=3)
    if len(top_nodes) >= 2:
        configs = [parse_json_field(n["config"]) for n in top_nodes]
        if configs[0] and configs[1]:
            common_keys = set(configs[0].keys()) & set(configs[1].keys())
            similar = []
            for key in common_keys:
                if configs[0][key] == configs[1][key]:
                    similar.append(f"{key}={configs[0][key]}")
            if similar:
                insights.append(store.create_insight(
                    tree_id=tree_id,
                    source_node_ids=[n["id"] for n in top_nodes[:2]],
                    insight_type="convergence",
                    content=(
                        f"Top results share parameters: {', '.join(similar)}. "
                        f"These may be important for good performance."
                    ),
                    confidence=0.7,
                ))

    return insights
