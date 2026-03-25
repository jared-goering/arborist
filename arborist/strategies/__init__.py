"""Search strategies for tree exploration."""

from arborist.strategies.base import Strategy
from arborist.strategies.best_first import BestFirstStrategy
from arborist.strategies.breadth_first import BreadthFirstStrategy
from arborist.strategies.llm_guided import LLMGuidedStrategy
from arborist.strategies.ucb import UCBStrategy

STRATEGIES: dict[str, type[Strategy]] = {
    "ucb": UCBStrategy,
    "best_first": BestFirstStrategy,
    "breadth_first": BreadthFirstStrategy,
    "llm_guided": LLMGuidedStrategy,
}

__all__ = [
    "Strategy",
    "UCBStrategy",
    "BestFirstStrategy",
    "BreadthFirstStrategy",
    "LLMGuidedStrategy",
    "STRATEGIES",
]
