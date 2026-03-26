"""Arborist — Agentic tree search engine for parallelized experiment orchestration."""

from arborist.evaluators import Evaluator, NumericEvaluator
from arborist.executors import (
    CodeGeneratorExecutor,
    Executor,
    PythonExecutor,
    ScriptScaffold,
    ShellExecutor,
)
from arborist.manager import BranchContext, TreeManager
from arborist.store import Store
from arborist.strategies import (
    STRATEGIES,
    BestFirstStrategy,
    BreadthFirstStrategy,
    Strategy,
    UCBStrategy,
)
from arborist.mutators import CodeGenMutator, LLMMutator, RandomMutator
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
    "STRATEGIES",
    "Executor",
    "PythonExecutor",
    "ShellExecutor",
    "CodeGeneratorExecutor",
    "ScriptScaffold",
    "Evaluator",
    "NumericEvaluator",
    "CodeGenMutator",
    "LLMMutator",
    "RandomMutator",
    "generate_report",
    "__version__",
]
