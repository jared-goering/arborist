"""LLM-guided strategy — uses language model reasoning to direct tree search."""

from __future__ import annotations

import json
import logging
import math
from typing import Any

from arborist.strategies.base import Strategy
from arborist.strategies.ucb import UCBStrategy

logger = logging.getLogger(__name__)


def _build_analysis_prompt(
    goal: str,
    completed: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    best_score: float,
    top_k: int = 10,
) -> str:
    """Build a prompt for the LLM to analyze search state and prioritize candidates."""
    # Sort completed by score descending
    scored = [n for n in completed if n.get("score") is not None]
    scored.sort(key=lambda n: n["score"], reverse=True)
    top = scored[:top_k]

    top_configs = []
    for n in top:
        config = n["config"]
        if isinstance(config, str):
            try:
                config = json.loads(config)
            except (json.JSONDecodeError, TypeError):
                pass
        top_configs.append({
            "score": round(n["score"], 4),
            "depth": n["depth"],
            "config": config,
        })

    candidate_summaries = []
    for c in candidates[:20]:  # Cap at 20
        config = c["config"]
        if isinstance(config, str):
            try:
                config = json.loads(config)
            except (json.JSONDecodeError, TypeError):
                pass
        candidate_summaries.append({
            "id": c["id"][:8],
            "depth": c["depth"],
            "parent_score": None,
            "config": config,
        })
        # Try to find parent score
        parent_id = c.get("parent_id")
        if parent_id:
            for comp in completed:
                if comp["id"] == parent_id and comp.get("score") is not None:
                    candidate_summaries[-1]["parent_score"] = round(comp["score"], 4)
                    break

    prompt = f"""You are an expert ML experiment optimizer guiding a tree search.

GOAL: {goal}

SEARCH STATE:
- Total completed experiments: {len(completed)}
- Best score so far: {best_score:.4f}
- Pending candidates: {len(candidates)}

TOP {len(top_configs)} RESULTS:
{json.dumps(top_configs, indent=2)}

PENDING CANDIDATES (select and rank these):
{json.dumps(candidate_summaries, indent=2)}

TASK: Analyze the patterns in successful experiments. Then rank the pending candidates by priority.

Return a JSON object with:
1. "analysis": 1-2 sentence insight about what's working
2. "rankings": array of candidate IDs (first 8 chars) in priority order (most promising first)
3. "prune": array of candidate IDs that are clearly unpromising and should be skipped

Focus on:
- Which hyperparameter regions produce the best scores
- Whether depth correlates with improvement
- Diminishing returns in certain branches

Return ONLY valid JSON, no markdown."""

    return prompt


class LLMGuidedStrategy(Strategy):
    """Uses LLM reasoning to guide node selection, with UCB fallback.

    Every `analysis_interval` completed experiments, asks the LLM to analyze
    the search tree and rerank pending nodes. Between analyses, uses UCB.
    
    The LLM also suggests which branches to prune based on pattern analysis.
    """

    def __init__(
        self,
        model: str = "openrouter/anthropic/claude-haiku-4-5",
        goal: str = "",
        analysis_interval: int = 10,
        exploration_weight: float = math.sqrt(2),
        prune_threshold: float = 0.4,
        plateau_window: int = 30,
    ) -> None:
        self.model = model
        self.goal = goal
        self.analysis_interval = analysis_interval
        self.exploration_weight = exploration_weight
        self.prune_threshold = prune_threshold
        self.plateau_window = plateau_window

        # UCB fallback for when LLM rankings aren't available
        self._ucb = UCBStrategy(exploration_weight=exploration_weight)

        # Cache LLM rankings between analyses
        self._cached_rankings: list[str] = []
        self._cached_prune: set[str] = set()
        self._last_analysis_count = 0
        self._analysis_count = 0

    def _should_reanalyze(self, n_completed: int) -> bool:
        """Check if we should query the LLM again."""
        return (n_completed - self._last_analysis_count) >= self.analysis_interval

    def _analyze(
        self,
        candidates: list[dict[str, Any]],
        completed: list[dict[str, Any]],
        goal: str = "",
    ) -> None:
        """Query LLM to analyze tree state and update rankings."""
        try:
            import litellm

            best_score = max(
                (n["score"] for n in completed if n.get("score") is not None),
                default=0,
            )

            prompt = _build_analysis_prompt(
                goal=goal,
                completed=completed,
                candidates=candidates,
                best_score=best_score,
            )

            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )

            text = response.choices[0].message.content or ""
            # Strip markdown code fences if present
            text = text.strip()
            if text.startswith("```"):
                # Remove first line (```json or ```) and last line (```)
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines)
            parsed = json.loads(text)

            if "analysis" in parsed:
                logger.info("LLM analysis: %s", parsed["analysis"])

            if "rankings" in parsed and isinstance(parsed["rankings"], list):
                self._cached_rankings = [str(r) for r in parsed["rankings"]]

            if "prune" in parsed and isinstance(parsed["prune"], list):
                self._cached_prune = {str(p) for p in parsed["prune"]}

            self._last_analysis_count = len(completed)
            self._analysis_count += 1
            logger.info(
                "LLM analysis #%d complete: %d rankings, %d prune suggestions",
                self._analysis_count,
                len(self._cached_rankings),
                len(self._cached_prune),
            )

        except Exception as e:
            logger.warning("LLM analysis failed, falling back to UCB: %s", e)

    def select(
        self,
        candidates: list[dict[str, Any]],
        completed: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        # Check if we should reanalyze
        if self._should_reanalyze(len(completed)):
            self._analyze(candidates, completed, self.goal)

        # If we have LLM rankings, use them as primary sort
        if self._cached_rankings:
            ranking_map = {
                r: i for i, r in enumerate(self._cached_rankings)
            }

            def llm_priority(node: dict[str, Any]) -> float:
                node_prefix = node["id"][:8]
                if node_prefix in self._cached_prune:
                    return float("inf")  # Push pruned to end
                if node_prefix in ranking_map:
                    return ranking_map[node_prefix]
                return len(self._cached_rankings)  # Unranked = after ranked

            return sorted(candidates, key=llm_priority)

        # Fallback: UCB
        return self._ucb_select(candidates, completed)

    def _ucb_select(
        self,
        candidates: list[dict[str, Any]],
        completed: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """UCB1 fallback selection — delegates to composed UCBStrategy."""
        return self._ucb.select(candidates, completed)

    def should_prune(
        self,
        node: dict[str, Any],
        best_score: float,
        siblings: list[dict[str, Any]],
    ) -> tuple[bool, str]:
        if node.get("score") is None:
            return False, ""

        # LLM-suggested prune
        if node["id"][:8] in self._cached_prune:
            return True, "LLM analysis identified branch as unpromising"

        # Numeric prune threshold
        if best_score > 0 and node["score"] < best_score * self.prune_threshold:
            return True, (
                f"Score {node['score']:.4f} < {self.prune_threshold:.0%} of best "
                f"({best_score:.4f})"
            )

        return False, ""
