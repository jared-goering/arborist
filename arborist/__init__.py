"""Arborist — Tree search engine for automated ML experimentation."""

from arborist.evaluators import Evaluator, NumericEvaluator
from arborist.executors import Executor, PythonExecutor, ShellExecutor
from arborist.manager import BranchContext, TreeManager
from arborist.store import Store
from arborist.strategies import (
    STRATEGIES,
    BestFirstStrategy,
    BreadthFirstStrategy,
    LLMGuidedStrategy,
    Strategy,
    UCBStrategy,
)
from arborist.mutators import LLMMutator, RandomMutator
from arborist.synthesis import SearchResults, generate_report
from arborist.tree import TreeSearch

__version__ = "0.1.0"

__all__ = [
    "TreeSearch",
    "SearchResults",
    "Store",
    "TreeManager",
    "BranchContext",
    "Strategy",
    "UCBStrategy",
    "BestFirstStrategy",
    "BreadthFirstStrategy",
    "LLMGuidedStrategy",
    "STRATEGIES",
    "Executor",
    "PythonExecutor",
    "ShellExecutor",
    "Evaluator",
    "NumericEvaluator",
    "LLMMutator",
    "RandomMutator",
    "generate_report",
    "__version__",
]
