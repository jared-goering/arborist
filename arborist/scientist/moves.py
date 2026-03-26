"""Research move library — catalog of typed interventions for the scientist."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class MoveCategory(str, enum.Enum):
    """Categories of research moves the scientist can make."""

    PARAM_TUNING = "param_tuning"
    FEATURE_ENGINEERING = "feature_engineering"
    ARCHITECTURE = "architecture"
    DATA_CURATION = "data_curation"
    ENSEMBLE = "ensemble"
    PIPELINE = "pipeline"
    EVALUATION = "evaluation"


@dataclass
class ResearchMove:
    """A typed intervention the scientist can propose."""

    name: str
    category: MoveCategory
    description: str
    default_strategy: str = "ucb"
    default_mutator: str = "llm"
    default_budget: int = 30
    default_max_depth: int = 4
    applicable_when: list[str] = field(default_factory=list)

    def generate_config(
        self,
        observation: Any | None = None,
        budget_override: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate an Arborist TreeSearch config for this move type.

        Args:
            observation: Current Observation (used to adapt config).
            budget_override: Override the default budget.
            **kwargs: Additional config overrides.

        Returns:
            Dict suitable for passing to TreeSearch constructor.
        """
        budget = budget_override or self.default_budget

        config: dict[str, Any] = {
            "strategy": self.default_strategy,
            "mutator": self.default_mutator,
            "max_experiments": budget,
            "max_depth": self.default_max_depth,
        }

        # Adapt config based on observation if available
        if observation is not None:
            config = self._adapt_to_observation(config, observation)

        config.update(kwargs)
        return config

    def _adapt_to_observation(
        self,
        config: dict[str, Any],
        observation: Any,
    ) -> dict[str, Any]:
        """Adjust config based on current observations."""
        # If we're seeing diminishing returns, reduce budget
        if hasattr(observation, "diminishing_returns") and observation.diminishing_returns:
            config["max_experiments"] = max(10, config["max_experiments"] // 2)

        # If plateau detected, increase exploration
        if hasattr(observation, "plateau_detected") and observation.plateau_detected:
            if config["strategy"] == "ucb":
                config["exploration_weight"] = 2.0  # more exploration

        return config


# ── Move Library ──────────────────────────────────────────────────────

PARAM_TUNING = ResearchMove(
    name="Hyperparameter Tuning",
    category=MoveCategory.PARAM_TUNING,
    description=(
        "Optimize existing hyperparameters using tree search. "
        "Best when the model architecture is fixed and quick gains remain."
    ),
    default_strategy="ucb",
    default_mutator="random",
    default_budget=50,
    default_max_depth=5,
    applicable_when=[
        "baseline exists",
        "architecture is fixed",
        "quick gains expected",
    ],
)

FEATURE_ENGINEERING = ResearchMove(
    name="Feature Engineering",
    category=MoveCategory.FEATURE_ENGINEERING,
    description=(
        "LLM proposes feature code changes against a stable scaffold. "
        "Use when error analysis shows missing signal or feature gaps."
    ),
    default_strategy="ucb",
    default_mutator="llm",
    default_budget=30,
    default_max_depth=4,
    applicable_when=[
        "error analysis shows missing signal",
        "feature importance reveals gaps",
        "domain knowledge suggests untried features",
    ],
)

ARCHITECTURE = ResearchMove(
    name="Architecture Search",
    category=MoveCategory.ARCHITECTURE,
    description=(
        "Change model type or pipeline structure. "
        "Use when the current model class is limiting performance."
    ),
    default_strategy="llm_guided",
    default_mutator="llm",
    default_budget=20,
    default_max_depth=3,
    applicable_when=[
        "current model class is limiting",
        "error patterns suggest structural issues",
        "different model families untried",
    ],
)

DATA_CURATION = ResearchMove(
    name="Data Curation",
    category=MoveCategory.DATA_CURATION,
    description=(
        "Preprocessing, augmentation, resampling, or cleaning changes. "
        "Use when data quality issues are identified."
    ),
    default_strategy="breadth_first",
    default_mutator="llm",
    default_budget=20,
    default_max_depth=3,
    applicable_when=[
        "class imbalance detected",
        "outliers or noise identified",
        "missing values present",
        "data distribution issues",
    ],
)

ENSEMBLE = ResearchMove(
    name="Ensemble Methods",
    category=MoveCategory.ENSEMBLE,
    description=(
        "Combine top-N models with complementary error patterns. "
        "Use when individual models have diverse failure modes."
    ),
    default_strategy="breadth_first",
    default_mutator="llm",
    default_budget=15,
    default_max_depth=2,
    applicable_when=[
        "multiple trained models available",
        "models have complementary errors",
        "individual model performance plateaued",
    ],
)

PIPELINE = ResearchMove(
    name="Pipeline Modification",
    category=MoveCategory.PIPELINE,
    description=(
        "Change preprocessing, normalization, or feature pipeline. "
        "Use when feature distribution issues are suspected."
    ),
    default_strategy="ucb",
    default_mutator="llm",
    default_budget=25,
    default_max_depth=4,
    applicable_when=[
        "feature distribution issues",
        "normalization mismatch",
        "preprocessing bottleneck",
    ],
)

EVALUATION = ResearchMove(
    name="Evaluation Strategy",
    category=MoveCategory.EVALUATION,
    description=(
        "Change metrics, cross-validation strategy, or evaluation methodology. "
        "Use when current evaluation may be misleading."
    ),
    default_strategy="breadth_first",
    default_mutator="llm",
    default_budget=10,
    default_max_depth=2,
    applicable_when=[
        "metric may not reflect true goal",
        "cross-val strategy may leak data",
        "evaluation methodology questionable",
    ],
)


# Registry: category -> move
MOVE_LIBRARY: dict[MoveCategory, ResearchMove] = {
    MoveCategory.PARAM_TUNING: PARAM_TUNING,
    MoveCategory.FEATURE_ENGINEERING: FEATURE_ENGINEERING,
    MoveCategory.ARCHITECTURE: ARCHITECTURE,
    MoveCategory.DATA_CURATION: DATA_CURATION,
    MoveCategory.ENSEMBLE: ENSEMBLE,
    MoveCategory.PIPELINE: PIPELINE,
    MoveCategory.EVALUATION: EVALUATION,
}


def get_move(category: MoveCategory | str) -> ResearchMove:
    """Look up a research move by category."""
    if isinstance(category, str):
        category = MoveCategory(category)
    move = MOVE_LIBRARY.get(category)
    if not move:
        raise ValueError(f"Unknown move category: {category}")
    return move
