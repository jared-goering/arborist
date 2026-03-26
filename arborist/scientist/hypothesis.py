"""Hypothesis generation — LLM-powered research hypothesis ranking."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from arborist.scientist.moves import MoveCategory, get_move
from arborist.scientist.observer import Observation

logger = logging.getLogger(__name__)


@dataclass
class Hypothesis:
    """A falsifiable research hypothesis mapping to a research move."""

    id: str
    description: str
    move_category: MoveCategory
    expected_impact: float  # 0.0 to 1.0
    confidence: float       # 0.0 to 1.0
    effort: str             # LOW, MEDIUM, HIGH
    rationale: str
    falsifiable: str        # criteria for rejection
    depends_on: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "move_category": self.move_category.value,
            "expected_impact": self.expected_impact,
            "confidence": self.confidence,
            "effort": self.effort,
            "rationale": self.rationale,
            "falsifiable": self.falsifiable,
            "depends_on": self.depends_on,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Hypothesis:
        return cls(
            id=data.get("id", f"h_{uuid.uuid4().hex[:6]}"),
            description=data["description"],
            move_category=MoveCategory(data.get("move_category", "param_tuning")),
            expected_impact=float(data.get("expected_impact", 0.5)),
            confidence=float(data.get("confidence", 0.5)),
            effort=data.get("effort", "MEDIUM"),
            rationale=data.get("rationale", ""),
            falsifiable=data.get("falsifiable", ""),
            depends_on=data.get("depends_on", []),
        )

    @property
    def priority_score(self) -> float:
        """Rank hypotheses by expected value adjusted for effort."""
        effort_multiplier = {"LOW": 1.5, "MEDIUM": 1.0, "HIGH": 0.6}
        return (
            self.expected_impact
            * self.confidence
            * effort_multiplier.get(self.effort, 1.0)
        )


class HypothesisGenerator:
    """Generates ranked research hypotheses from observations using an LLM."""

    def __init__(
        self,
        model: str = "openrouter/anthropic/claude-haiku-4-5",
        max_hypotheses: int = 5,
    ) -> None:
        self.model = model
        self.max_hypotheses = max_hypotheses

    def generate(
        self,
        observation: Observation,
        problem: str,
        journal_context: str = "",
    ) -> list[Hypothesis]:
        """Generate ranked hypotheses from an observation.

        Args:
            observation: Current structured observation.
            problem: The research problem statement.
            journal_context: Summary of past experiments from journal.

        Returns:
            List of Hypothesis objects, ranked by priority_score.
        """
        prompt = self._build_prompt(observation, problem, journal_context)

        try:
            import litellm

            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content
            hypotheses = self._parse_response(text)
        except Exception as e:
            logger.warning("LLM hypothesis generation failed: %s", e)
            hypotheses = self._fallback_hypotheses(observation)

        # Sort by priority
        hypotheses.sort(key=lambda h: h.priority_score, reverse=True)
        return hypotheses[:self.max_hypotheses]

    def _system_prompt(self) -> str:
        categories = ", ".join(c.value for c in MoveCategory)
        return (
            "You are a research scientist analyzing ML experiment results. "
            "Your job is to generate falsifiable hypotheses for improving model "
            "performance. Each hypothesis must map to a specific research move "
            f"category: {categories}.\n\n"
            "Ground every hypothesis in the actual data — reference specific "
            "class confusions, specific metrics, specific features. "
            "No vague suggestions.\n\n"
            "Respond with a JSON object: {\"hypotheses\": [...]}"
        )

    def _build_prompt(
        self,
        observation: Observation,
        problem: str,
        journal_context: str,
    ) -> str:
        parts = [f"## Research Problem\n{problem}\n"]

        parts.append("## Current Observations")
        parts.append(observation.summary())

        obs_data = observation.to_dict()

        if obs_data["class_metrics"]:
            parts.append("\n### Per-Class Metrics")
            for cls, m in obs_data["class_metrics"].items():
                parts.append(
                    f"  {cls}: P={m.get('precision', 0):.3f} "
                    f"R={m.get('recall', 0):.3f} "
                    f"F1={m.get('f1', 0):.3f} "
                    f"(n={m.get('support', '?')})"
                )

        if obs_data["confusion_matrix"]:
            parts.append("\n### Confusion Matrix")
            parts.append(json.dumps(obs_data["confusion_matrix"], indent=2))

        if obs_data["feature_importance"]:
            parts.append("\n### Top Features (by importance)")
            for feat, imp in list(obs_data["feature_importance"].items())[:10]:
                parts.append(f"  {feat}: {imp:.4f}")

        if obs_data.get("subject_variance") is not None:
            parts.append(
                f"\n### Subject Variance: {obs_data['subject_variance']:.4f}"
            )

        if journal_context:
            parts.append(f"\n## Past Experiment History\n{journal_context}")

        parts.append(
            f"\n## Instructions\n"
            f"Generate up to {self.max_hypotheses} hypotheses. "
            f"For each, provide:\n"
            f'- "description": what to try\n'
            f'- "move_category": one of {[c.value for c in MoveCategory]}\n'
            f'- "expected_impact": 0.0-1.0\n'
            f'- "confidence": 0.0-1.0\n'
            f'- "effort": LOW|MEDIUM|HIGH\n'
            f'- "rationale": why this should work (cite specific data)\n'
            f'- "falsifiable": criteria to reject this hypothesis\n'
        )

        return "\n".join(parts)

    def _parse_response(self, text: str | None) -> list[Hypothesis]:
        """Parse LLM response into Hypothesis objects."""
        if not text:
            return []

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try extracting JSON from markdown fences
            import re
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                data = json.loads(match.group(1))
            else:
                logger.warning("Could not parse hypothesis response")
                return []

        raw_list = data.get("hypotheses", [])
        if not isinstance(raw_list, list):
            return []

        hypotheses = []
        for i, raw in enumerate(raw_list):
            if not isinstance(raw, dict):
                continue
            if "description" not in raw:
                continue

            # Assign ID if missing
            if "id" not in raw:
                raw["id"] = f"h_{i:03d}"

            # Validate move_category
            cat = raw.get("move_category", "param_tuning")
            try:
                MoveCategory(cat)
            except ValueError:
                raw["move_category"] = "param_tuning"

            try:
                hypotheses.append(Hypothesis.from_dict(raw))
            except (KeyError, ValueError) as e:
                logger.warning("Skipping malformed hypothesis: %s", e)

        return hypotheses

    def _fallback_hypotheses(self, observation: Observation) -> list[Hypothesis]:
        """Generate simple hypotheses without LLM when it fails."""
        hypotheses = []

        # Always suggest param tuning as a safe default
        hypotheses.append(Hypothesis(
            id="h_fallback_001",
            description="Tune hyperparameters with broader search range",
            move_category=MoveCategory.PARAM_TUNING,
            expected_impact=0.3,
            confidence=0.7,
            effort="LOW",
            rationale="Hyperparameter tuning is a reliable baseline improvement strategy",
            falsifiable="If best score doesn't improve by >0.5% after 50 experiments, reject",
        ))

        if observation.worst_classes:
            worst = observation.worst_classes[0]
            hypotheses.append(Hypothesis(
                id="h_fallback_002",
                description=f"Engineer features targeting {worst} class discrimination",
                move_category=MoveCategory.FEATURE_ENGINEERING,
                expected_impact=0.5,
                confidence=0.5,
                effort="MEDIUM",
                rationale=f"Class {worst} has lowest F1, targeted features may help",
                falsifiable=f"If {worst} F1 doesn't improve by >2%, reject",
            ))

        if observation.data_issues:
            hypotheses.append(Hypothesis(
                id="h_fallback_003",
                description="Address data quality issues via resampling or cleaning",
                move_category=MoveCategory.DATA_CURATION,
                expected_impact=0.4,
                confidence=0.5,
                effort="MEDIUM",
                rationale=f"Data issues detected: {observation.data_issues[0]}",
                falsifiable="If overall metric doesn't improve by >1%, reject",
            ))

        return hypotheses
