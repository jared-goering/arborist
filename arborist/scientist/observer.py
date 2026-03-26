"""Observation engine — reads model outputs to produce structured observations."""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Observation:
    """Structured observation of current model/experiment state."""

    # Per-class metrics
    class_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    # e.g. {"Wake": {"precision": 0.8, "recall": 0.6, "f1": 0.69}, ...}

    # Confusion matrix (row=true, col=predicted)
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)

    # Classes with worst performance
    worst_classes: list[str] = field(default_factory=list)

    # Detected error patterns
    error_patterns: list[str] = field(default_factory=list)

    # Feature gaps or missing signals
    feature_gaps: list[str] = field(default_factory=list)

    # Data quality issues
    data_issues: list[str] = field(default_factory=list)

    # Current best result
    current_best: dict[str, Any] = field(default_factory=dict)

    # How many experiments have been run
    experiment_count: int = 0

    # Multi-seed evaluation results
    seed_scores: list[float] = field(default_factory=list)
    score_mean: float | None = None
    score_std: float | None = None
    score_ci_95: tuple[float, float] | None = None

    # Per-subject variance (if subject IDs available)
    subject_scores: dict[str, float] = field(default_factory=dict)
    subject_variance: float | None = None

    # Feature importance rankings
    feature_importance: dict[str, float] = field(default_factory=dict)

    # Plateau detection
    plateau_detected: bool = False
    diminishing_returns: bool = False

    # Score history for trend analysis
    score_history: list[float] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary of observations."""
        lines = []

        if self.current_best:
            metric = self.current_best.get("metric", "score")
            value = self.current_best.get("value", "?")
            lines.append(f"Current best {metric}: {value}")

        if self.experiment_count:
            lines.append(f"Experiments run: {self.experiment_count}")

        if self.score_mean is not None:
            ci = ""
            if self.score_ci_95:
                ci = f" (95% CI: [{self.score_ci_95[0]:.4f}, {self.score_ci_95[1]:.4f}])"
                lines.append(
                    f"Multi-seed: mean={self.score_mean:.4f} "
                    f"std={self.score_std:.4f}{ci}"
                )

        if self.worst_classes:
            lines.append(f"Worst classes: {', '.join(self.worst_classes)}")

        if self.error_patterns:
            lines.append("Error patterns:")
            for p in self.error_patterns[:5]:
                lines.append(f"  - {p}")

        if self.data_issues:
            lines.append("Data issues:")
            for d in self.data_issues[:5]:
                lines.append(f"  - {d}")

        if self.plateau_detected:
            lines.append("WARNING: Performance plateau detected")

        if self.diminishing_returns:
            lines.append("WARNING: Diminishing returns detected")

        return "\n".join(lines) if lines else "No observations yet."

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for LLM prompts and journal storage."""
        return {
            "class_metrics": self.class_metrics,
            "confusion_matrix": self.confusion_matrix,
            "worst_classes": self.worst_classes,
            "error_patterns": self.error_patterns,
            "feature_gaps": self.feature_gaps,
            "data_issues": self.data_issues,
            "current_best": self.current_best,
            "experiment_count": self.experiment_count,
            "score_mean": self.score_mean,
            "score_std": self.score_std,
            "score_ci_95": list(self.score_ci_95) if self.score_ci_95 else None,
            "subject_scores": self.subject_scores,
            "subject_variance": self.subject_variance,
            "feature_importance": dict(
                sorted(
                    self.feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:20]
            ) if self.feature_importance else {},
            "plateau_detected": self.plateau_detected,
            "diminishing_returns": self.diminishing_returns,
        }


class Observer:
    """Produces Observations from model outputs and experiment history."""

    def __init__(
        self,
        metric_name: str = "f1_macro",
        plateau_window: int = 10,
        plateau_threshold: float = 0.005,
    ) -> None:
        self.metric_name = metric_name
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold

    def observe(
        self,
        results: dict[str, Any] | None = None,
        score_history: list[float] | None = None,
        experiment_count: int = 0,
        best_score: float | None = None,
        best_config: dict[str, Any] | None = None,
    ) -> Observation:
        """Build an Observation from available data.

        Args:
            results: Latest experiment results dict (may contain
                classification_report, confusion_matrix, feature_importance, etc.)
            score_history: List of scores from all experiments in order.
            experiment_count: Total experiments run so far.
            best_score: Best score achieved.
            best_config: Config that produced best score.
        """
        obs = Observation(experiment_count=experiment_count)

        if best_score is not None:
            obs.current_best = {
                "metric": self.metric_name,
                "value": best_score,
                "config": best_config or {},
            }

        if results:
            self._parse_classification_report(obs, results)
            self._parse_confusion_matrix(obs, results)
            self._parse_feature_importance(obs, results)
            self._parse_subject_scores(obs, results)
            self._detect_data_issues(obs, results)

        if score_history:
            obs.score_history = list(score_history)
            self._detect_plateau(obs, score_history)

        return obs

    def observe_multi_seed(
        self,
        scores: list[float],
        results_list: list[dict[str, Any]] | None = None,
    ) -> Observation:
        """Build observation from multiple seed evaluations.

        Args:
            scores: List of scores from different random seeds.
            results_list: Optional list of full results dicts per seed.
        """
        obs = Observation()
        obs.seed_scores = list(scores)

        if scores:
            n = len(scores)
            obs.score_mean = sum(scores) / n
            if n > 1:
                variance = sum((s - obs.score_mean) ** 2 for s in scores) / (n - 1)
                obs.score_std = math.sqrt(variance)
                # 95% CI using t-distribution approximation (z=1.96 for large n)
                z = 1.96
                margin = z * obs.score_std / math.sqrt(n)
                obs.score_ci_95 = (
                    obs.score_mean - margin,
                    obs.score_mean + margin,
                )
            else:
                obs.score_std = 0.0

        # Merge results from first seed for class metrics etc.
        if results_list and results_list[0]:
            self._parse_classification_report(obs, results_list[0])
            self._parse_confusion_matrix(obs, results_list[0])

        return obs

    def _parse_classification_report(
        self,
        obs: Observation,
        results: dict[str, Any],
    ) -> None:
        """Extract per-class metrics from sklearn-style classification report."""
        report = results.get("classification_report")
        if isinstance(report, str):
            report = self._parse_report_text(report)
        if not isinstance(report, dict):
            return

        class_metrics: dict[str, dict[str, float]] = {}
        skip_keys = {"accuracy", "macro avg", "weighted avg", "micro avg"}

        for class_name, metrics in report.items():
            if class_name in skip_keys:
                continue
            if not isinstance(metrics, dict):
                continue
            class_metrics[class_name] = {
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "f1": metrics.get("f1-score", metrics.get("f1", 0.0)),
                "support": metrics.get("support", 0),
            }

        obs.class_metrics = class_metrics

        # Identify worst classes by F1
        if class_metrics:
            sorted_classes = sorted(
                class_metrics.items(),
                key=lambda x: x[1].get("f1", 0),
            )
            obs.worst_classes = [
                name for name, _ in sorted_classes[:3]
                if class_metrics[name].get("f1", 1.0) < 0.5
            ]

            # Detect error patterns
            for name, m in class_metrics.items():
                p, r = m.get("precision", 0), m.get("recall", 0)
                if p > 0 and r > 0:
                    if p > r * 1.5:
                        obs.error_patterns.append(
                            f"{name}: low recall ({r:.2f}) vs precision ({p:.2f}) "
                            f"— model misses many {name} samples"
                        )
                    elif r > p * 1.5:
                        obs.error_patterns.append(
                            f"{name}: low precision ({p:.2f}) vs recall ({r:.2f}) "
                            f"— model over-predicts {name}"
                        )

    def _parse_report_text(self, text: str) -> dict[str, Any]:
        """Parse sklearn classification_report text output to dict."""
        result: dict[str, Any] = {}
        lines = text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Match lines like: "Wake       0.85      0.70      0.77       150"
            match = re.match(
                r"^(\S+(?:\s+\S+)*?)\s+"
                r"(\d+\.?\d*)\s+"
                r"(\d+\.?\d*)\s+"
                r"(\d+\.?\d*)\s+"
                r"(\d+)",
                line,
            )
            if match:
                name = match.group(1).strip()
                result[name] = {
                    "precision": float(match.group(2)),
                    "recall": float(match.group(3)),
                    "f1-score": float(match.group(4)),
                    "support": int(match.group(5)),
                }
        return result

    def _parse_confusion_matrix(
        self,
        obs: Observation,
        results: dict[str, Any],
    ) -> None:
        """Parse confusion matrix from results."""
        cm = results.get("confusion_matrix")
        labels = results.get("class_labels", results.get("labels"))

        if cm is None:
            return

        if isinstance(cm, list) and labels and isinstance(labels, list):
            matrix: dict[str, dict[str, int]] = {}
            for i, true_label in enumerate(labels):
                row: dict[str, int] = {}
                for j, pred_label in enumerate(labels):
                    if i < len(cm) and j < len(cm[i]):
                        row[str(pred_label)] = int(cm[i][j])
                matrix[str(true_label)] = row
            obs.confusion_matrix = matrix

            # Detect confusion patterns
            for true_label, preds in matrix.items():
                total = sum(preds.values())
                if total == 0:
                    continue
                for pred_label, count in preds.items():
                    if pred_label != true_label and count > 0:
                        rate = count / total
                        if rate > 0.2:
                            obs.error_patterns.append(
                                f"{true_label} misclassified as {pred_label} "
                                f"{rate:.0%} of the time ({count}/{total})"
                            )
        elif isinstance(cm, dict):
            obs.confusion_matrix = cm

    def _parse_feature_importance(
        self,
        obs: Observation,
        results: dict[str, Any],
    ) -> None:
        """Extract feature importance rankings."""
        fi = results.get("feature_importance")
        if isinstance(fi, dict):
            obs.feature_importance = {
                str(k): float(v) for k, v in fi.items()
            }
        elif isinstance(fi, list):
            names = results.get("feature_names", [])
            if names and len(names) == len(fi):
                obs.feature_importance = {
                    str(n): float(v) for n, v in zip(names, fi)
                }

        # Detect feature gaps
        if obs.feature_importance:
            sorted_fi = sorted(
                obs.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            total_importance = sum(v for _, v in sorted_fi)
            if total_importance > 0:
                top_3_share = sum(v for _, v in sorted_fi[:3]) / total_importance
                if top_3_share > 0.8:
                    top_names = [n for n, _ in sorted_fi[:3]]
                    obs.feature_gaps.append(
                        f"Top 3 features ({', '.join(top_names)}) account for "
                        f"{top_3_share:.0%} of importance — consider adding "
                        f"diverse feature types"
                    )

    def _parse_subject_scores(
        self,
        obs: Observation,
        results: dict[str, Any],
    ) -> None:
        """Parse per-subject performance if available."""
        subj = results.get("subject_scores", results.get("per_subject"))
        if isinstance(subj, dict):
            obs.subject_scores = {str(k): float(v) for k, v in subj.items()}
            if len(obs.subject_scores) > 1:
                values = list(obs.subject_scores.values())
                mean = sum(values) / len(values)
                obs.subject_variance = sum(
                    (v - mean) ** 2 for v in values
                ) / (len(values) - 1)

                # Flag high-variance subjects
                for subj_id, score in obs.subject_scores.items():
                    if score < mean - 2 * math.sqrt(obs.subject_variance):
                        obs.data_issues.append(
                            f"Subject {subj_id} scores very low ({score:.3f}) "
                            f"vs mean ({mean:.3f})"
                        )

    def _detect_data_issues(
        self,
        obs: Observation,
        results: dict[str, Any],
    ) -> None:
        """Detect data quality issues from results."""
        # Class imbalance from support values
        if obs.class_metrics:
            supports = {
                name: m.get("support", 0)
                for name, m in obs.class_metrics.items()
            }
            total = sum(supports.values())
            if total > 0:
                for name, count in supports.items():
                    ratio = count / total
                    if ratio < 0.05:
                        obs.data_issues.append(
                            f"Class '{name}' is severely underrepresented "
                            f"({ratio:.1%} of samples)"
                        )
                    elif ratio > 0.5:
                        obs.data_issues.append(
                            f"Class '{name}' dominates the dataset "
                            f"({ratio:.1%} of samples)"
                        )

    def _detect_plateau(
        self,
        obs: Observation,
        score_history: list[float],
    ) -> None:
        """Detect if score improvements have plateaued."""
        if len(score_history) < self.plateau_window * 2:
            return

        recent = score_history[-self.plateau_window:]
        older = score_history[-self.plateau_window * 2:-self.plateau_window]

        recent_best = max(recent)
        older_best = max(older)

        improvement = recent_best - older_best

        if improvement < self.plateau_threshold:
            obs.plateau_detected = True

        # Diminishing returns: check if rate of improvement is declining
        if len(score_history) >= self.plateau_window * 3:
            even_older = score_history[-self.plateau_window * 3:-self.plateau_window * 2]
            early_improvement = older_best - max(even_older)
            if early_improvement > 0 and improvement < early_improvement * 0.25:
                obs.diminishing_returns = True
