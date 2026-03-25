"""Abstract executor interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from arborist.manager import BranchContext


class Executor(ABC):
    """Base class for experiment executors."""

    @abstractmethod
    def run(self, config: dict[str, Any], context: BranchContext) -> dict[str, Any]:
        """Run one experiment.

        Args:
            config: Experiment parameters.
            context: Branch context (parent info, goal, depth, etc.).

        Returns:
            Structured results dict.
        """
        ...
