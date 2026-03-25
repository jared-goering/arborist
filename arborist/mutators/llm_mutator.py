"""LLM-powered mutator that sees the full tree state to propose novel configs."""

from __future__ import annotations

import json
import logging
import random
import re
from typing import Any

from arborist.manager import BranchContext
from arborist.store import Store

logger = logging.getLogger(__name__)


def _parse_json_robustly(text: str) -> Any:
    """Parse JSON from LLM output, handling markdown fences, comments, etc."""
    text = text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip single-line comments (// ...)
    cleaned = re.sub(r"//[^\n]*", "", text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to extract a JSON array from the text
    array_match = re.search(r"\[[\s\S]*\]", text)
    if array_match:
        try:
            return json.loads(array_match.group())
        except json.JSONDecodeError:
            pass

    # Try to extract a JSON object with a "configs" key
    obj_match = re.search(r"\{[\s\S]*\}", text)
    if obj_match:
        try:
            parsed = json.loads(obj_match.group())
            if isinstance(parsed, dict) and "configs" in parsed:
                return parsed["configs"]
            return parsed
        except json.JSONDecodeError:
            pass

    return None


def _perturb_config_fallback(
    config: dict[str, Any],
    n: int = 2,
    param_bounds: dict | None = None,
) -> list[dict[str, Any]]:
    """Fallback: random perturbation of config values."""
    children = []
    for _ in range(n):
        child = dict(config)
        for key, value in config.items():
            if isinstance(value, (int, float)):
                factor = 1 + random.uniform(-0.2, 0.2)
                new_val = value * factor
                # Enforce bounds if available
                if param_bounds and key in param_bounds:
                    lo, hi, dtype = param_bounds[key]
                    new_val = max(lo, min(hi, new_val))
                    if dtype == "int":
                        new_val = int(round(new_val))
                child[key] = type(value)(new_val) if isinstance(value, int) and not (param_bounds and key in param_bounds) else new_val
        children.append(child)
    return children


def _clip_config(config: dict[str, Any], param_bounds: dict) -> dict[str, Any]:
    """Clip config values to valid ranges defined by param_bounds."""
    clipped = dict(config)
    for key, (lo, hi, dtype) in param_bounds.items():
        if key in clipped:
            val = clipped[key]
            if not isinstance(val, (int, float)):
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    continue
            val = max(lo, min(hi, val))
            if dtype == "int":
                val = int(round(val))
            else:
                val = round(float(val), 6)
            clipped[key] = val
    return clipped


def _safe_json(value: str | dict | None) -> dict | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def _build_tree_context_prompt(
    config: dict[str, Any],
    results: dict[str, Any],
    context: BranchContext,
    store: Store | None,
    tree_id: str | None,
    max_children: int,
    param_bounds: dict | None,
) -> str:
    """Build a rich prompt with full tree state for the LLM."""
    sections = []

    # Goal
    sections.append(f"GOAL: {context.goal}")

    # Param bounds
    if param_bounds:
        bounds_str = "\n".join(
            f"  {k}: [{lo}, {hi}] ({dtype})"
            for k, (lo, hi, dtype) in param_bounds.items()
        )
        sections.append(f"PARAMETER BOUNDS:\n{bounds_str}")

    # Parent config and score
    sections.append(f"PARENT CONFIG: {json.dumps(config, indent=2)}")
    if results:
        sections.append(f"PARENT RESULTS: {json.dumps(results, indent=2)}")
    if context.parent_score is not None:
        sections.append(f"PARENT SCORE: {context.parent_score:.4f}")

    # Full tree state from store
    if store and tree_id:
        # Top 10 completed experiments
        top_nodes = store.get_top_nodes(tree_id, k=10)
        if top_nodes:
            top_list = []
            for n in top_nodes:
                cfg = _safe_json(n["config"]) or {}
                top_list.append({
                    "score": round(n["score"], 4) if n["score"] else None,
                    "depth": n["depth"],
                    "config": cfg,
                })
            sections.append(
                f"TOP {len(top_list)} EXPERIMENTS (best scores):\n"
                f"{json.dumps(top_list, indent=2)}"
            )

        # Sibling configs and scores
        # Find parent node to get its children (our siblings)
        all_nodes = store.get_tree_nodes(tree_id)
        # Find the current node's parent_id — it's the node whose config matches
        # Actually we don't have node_id here, use sibling_scores from context
        if context.sibling_scores:
            sections.append(f"SIBLING SCORES: {context.sibling_scores}")

        # Failed/pruned branches
        failed_nodes = [
            n for n in all_nodes
            if n.get("pruned") or n.get("status") == "failed"
        ]
        if failed_nodes:
            failed_list = []
            for n in failed_nodes[:10]:
                cfg = _safe_json(n["config"]) or {}
                reason = n.get("prune_reason") or n.get("error") or "unknown"
                score = n.get("score")
                failed_list.append({
                    "config": cfg,
                    "score": round(score, 4) if score else None,
                    "reason": reason[:100],
                })
            sections.append(
                f"FAILED/PRUNED BRANCHES (avoid these regions):\n"
                f"{json.dumps(failed_list, indent=2)}"
            )

        # Stats
        completed_count = store.count_nodes(tree_id, status="completed")
        best_node = store.get_best_node(tree_id)
        best_score = round(best_node["score"], 4) if best_node and best_node["score"] else "N/A"
        sections.append(
            f"SEARCH STATS: {completed_count} completed experiments, best score = {best_score}"
        )

    prompt = "\n\n".join(sections)

    instruction = f"""
You are an expert experiment optimizer guiding a tree search. You have access to the full search history.

{prompt}

YOUR TASK: Propose exactly {max_children} new experiment configurations that are children of the parent config above.

STRATEGY:
- Look at what configs score well and what patterns emerge
- Propose configs that EXPLORE underexplored regions of the space
- Also propose configs that EXPLOIT promising patterns from the top experiments
- Avoid configs similar to failed/pruned branches
- Each proposed config should be meaningfully different from existing experiments
- Think about combinations that haven't been tried

Return ONLY a JSON array of {max_children} config objects. Each must have the same keys as the parent config.
Example format: [{json.dumps(config)}]

Return ONLY valid JSON, no explanation."""

    return instruction


class LLMMutator:
    """LLM-powered mutator that sees the full tree to propose novel configs.

    Unlike the default mutator which only sees the immediate parent,
    this mutator queries the Store for the complete search history:
    top-K experiments, siblings, failed branches, etc. The LLM can
    then reason about unexplored regions and propose qualitatively
    different directions.

    Falls back to random perturbation if the LLM call fails.
    """

    def __init__(
        self,
        model: str = "openrouter/anthropic/claude-haiku-4-5",
        max_children: int = 3,
        param_bounds: dict | None = None,
        store: Store | None = None,
    ) -> None:
        self.model = model
        self.max_children = max_children
        self.param_bounds = param_bounds
        self.store = store
        self._tree_id: str | None = None

    def set_store(self, store: Store) -> None:
        """Set the store reference (called by TreeSearch after init)."""
        self.store = store

    def set_tree_id(self, tree_id: str) -> None:
        """Set the tree ID for querying state."""
        self._tree_id = tree_id

    def __call__(
        self,
        config: dict[str, Any],
        results: dict[str, Any],
        context: BranchContext,
    ) -> list[dict[str, Any]]:
        """Generate child configs using LLM reasoning over full tree state."""
        try:
            return self._llm_mutate(config, results, context)
        except Exception as e:
            logger.warning("LLM mutator failed, falling back to random perturbation: %s", e)
            return _perturb_config_fallback(
                config, n=self.max_children, param_bounds=self.param_bounds
            )

    def _llm_mutate(
        self,
        config: dict[str, Any],
        results: dict[str, Any],
        context: BranchContext,
    ) -> list[dict[str, Any]]:
        """Call LLM with full tree context to generate children."""
        import litellm

        prompt = _build_tree_context_prompt(
            config=config,
            results=results,
            context=context,
            store=self.store,
            tree_id=self._tree_id,
            max_children=self.max_children,
            param_bounds=self.param_bounds,
        )

        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
        )

        text = response.choices[0].message.content or ""
        parsed = _parse_json_robustly(text)

        if parsed is None:
            logger.warning("Could not parse LLM response as JSON, falling back")
            return _perturb_config_fallback(
                config, n=self.max_children, param_bounds=self.param_bounds
            )

        # Handle {"configs": [...]} wrapper
        if isinstance(parsed, dict):
            if "configs" in parsed:
                parsed = parsed["configs"]
            else:
                # Single config returned as object
                parsed = [parsed]

        if not isinstance(parsed, list):
            logger.warning("LLM returned non-list: %s", type(parsed))
            return _perturb_config_fallback(
                config, n=self.max_children, param_bounds=self.param_bounds
            )

        # Filter to valid dicts with matching keys
        children = []
        config_keys = set(config.keys())
        for item in parsed:
            if not isinstance(item, dict):
                continue
            # Ensure it has at least some of the expected keys
            if not (set(item.keys()) & config_keys):
                continue
            # Fill in missing keys from parent config
            child = dict(config)
            child.update(item)
            # Clip to bounds
            if self.param_bounds:
                child = _clip_config(child, self.param_bounds)
            children.append(child)

        if not children:
            logger.warning("LLM returned no valid configs, falling back")
            return _perturb_config_fallback(
                config, n=self.max_children, param_bounds=self.param_bounds
            )

        return children[: self.max_children]
