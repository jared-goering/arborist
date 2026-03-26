"""Code generation mutator — proposes modification instructions for CodeGeneratorExecutor."""

from __future__ import annotations

import json
import logging
from typing import Any

from arborist.manager import BranchContext

logger = logging.getLogger(__name__)


class CodeGenMutator:
    """Mutator that generates modification instructions for code generation.

    Instead of mutating numeric parameters, this mutator asks an LLM to propose
    new natural-language modification instructions. The parent node's generated
    script becomes the new base for children, and mutations are additive
    (keep what worked, try additional changes).
    """

    def __init__(
        self,
        model: str = "openrouter/anthropic/claude-haiku-4-5",
        max_children: int = 2,
        base_script: str | None = None,
    ) -> None:
        self.model = model
        self.max_children = max_children
        self.base_script = base_script

    def __call__(
        self,
        config: dict[str, Any],
        results: dict[str, Any],
        context: BranchContext,
    ) -> list[dict[str, Any]]:
        """Generate child configs with new modification instructions."""
        try:
            return self._llm_propose(config, results, context)
        except Exception as e:
            logger.warning("CodeGenMutator LLM call failed: %s", e)
            return self._fallback(config)

    def _llm_propose(
        self,
        config: dict[str, Any],
        results: dict[str, Any],
        context: BranchContext,
    ) -> list[dict[str, Any]]:
        import litellm

        parent_modifications = config.get("modifications", "")
        parent_metrics = results.get("metrics", {})
        parent_diff = results.get("diff", "")

        # Use generated script as new base if it exists
        parent_script = config.get("generated_script") or config.get("base_script", "")

        prompt = self._build_prompt(
            parent_modifications=parent_modifications,
            parent_metrics=parent_metrics,
            parent_diff=parent_diff,
            context=context,
        )

        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
        )

        text = response.choices[0].message.content or ""
        proposals = self._parse_proposals(text)

        # Build child configs
        children = []
        for proposal in proposals[: self.max_children]:
            child = {
                "base_script": parent_script if parent_script else self.base_script,
                "modifications": proposal,
            }
            children.append(child)

        return children if children else self._fallback(config)

    def _build_prompt(
        self,
        parent_modifications: str,
        parent_metrics: dict[str, float],
        parent_diff: str,
        context: BranchContext,
    ) -> str:
        parts = [
            f"GOAL: {context.goal}\n",
            f"PARENT MODIFICATIONS: {parent_modifications}\n",
        ]

        if parent_metrics:
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in parent_metrics.items())
            parts.append(f"PARENT METRICS: {metrics_str}\n")

        if context.parent_score is not None:
            parts.append(f"PARENT SCORE: {context.parent_score:.4f}\n")

        if parent_diff:
            # Show a truncated diff for context
            diff_lines = parent_diff.split("\n")[:30]
            parts.append(f"PARENT DIFF (what was changed):\n{''.join(diff_lines)}\n")

        parts.append(
            f"\nPropose exactly {self.max_children} NEW modification instructions "
            f"that build on what the parent tried.\n"
            f"Each should be ADDITIVE — keep what worked and add new changes.\n"
            f"Be specific about feature engineering or architecture changes.\n\n"
            f"Return a JSON array of strings, each being a modification instruction.\n"
            f'Example: ["Add rolling std over 5 and 10 min windows", "Add frequency domain features via FFT"]\n'
        )

        return "".join(parts)

    def _parse_proposals(self, text: str) -> list[str]:
        """Parse LLM response into a list of modification instruction strings."""
        text = text.strip()

        # Strip markdown fences
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if item]
        except json.JSONDecodeError:
            pass

        # Fallback: split by newlines, look for numbered items
        proposals = []
        for line in text.split("\n"):
            line = line.strip().lstrip("0123456789.-) ")
            if line and len(line) > 10:
                proposals.append(line)

        return proposals

    def _fallback(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        """Simple fallback: keep same base with generic exploratory instructions."""
        base = config.get("generated_script") or config.get("base_script", "")
        return [
            {
                "base_script": base if base else self.base_script,
                "modifications": (
                    config.get("modifications", "")
                    + " Additionally, add interaction features between existing features."
                ),
            },
        ]
