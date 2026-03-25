"""Abstract evaluator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Evaluator(ABC):
    """Base class for experiment evaluators."""

    @abstractmethod
    def evaluate(self, results: dict[str, Any], config: dict[str, Any]) -> float:
        """Score an experiment's results.

        Args:
            results: The structured output from the executor.
            config: The experiment configuration.

        Returns:
            A numeric score (higher is better).
        """
        ...
