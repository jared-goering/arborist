"""ProblemSpec — generic description of an ML problem for the scientist."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProblemSpec:
    """Generic description of an ML problem.

    Provides all the information needed for FullScriptGenerator to produce
    a complete training script from scratch, without any existing base script.
    """

    name: str
    description: str
    dataset_path: str
    target_variable: str
    metric: str  # e.g. 'f1_macro', 'rmse', 'accuracy'
    task_type: str  # 'classification', 'regression', 'time_series'
    constraints: list[str] = field(default_factory=list)
    data_description: str = ""
    forbidden_patterns: list[str] = field(default_factory=list)
    output_format: dict = field(default_factory=lambda: {
        "pattern": "val_{metric}={value:.4f}",
        "example": "val_f1=0.4200 val_accuracy=0.7500",
    })
    python_cmd: str = "python3"
    timeout: float = 300
    extra_context: str = ""

    def to_prompt(self) -> str:
        """Render this spec as a prompt string for an LLM."""
        lines = [
            f"# Problem: {self.name}",
            f"Description: {self.description}",
            f"Dataset: {self.dataset_path}",
            f"Target variable: {self.target_variable}",
            f"Task type: {self.task_type}",
            f"Primary metric: {self.metric}",
        ]
        if self.data_description:
            lines.append(f"Data description: {self.data_description}")
        if self.constraints:
            lines.append("Constraints:")
            for c in self.constraints:
                lines.append(f"  - {c}")
        if self.forbidden_patterns:
            lines.append("Forbidden patterns (DO NOT use):")
            for p in self.forbidden_patterns:
                lines.append(f"  - {p}")
        if self.extra_context:
            lines.append(f"Additional context: {self.extra_context}")
        return "\n".join(lines)
