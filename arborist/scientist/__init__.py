"""Scientist layer — autonomous research agent for Arborist."""

from arborist.scientist.hypothesis import Hypothesis, HypothesisGenerator
from arborist.scientist.journal import Journal, JournalEntry
from arborist.scientist.moves import (
    MOVE_LIBRARY,
    MoveCategory,
    ResearchMove,
    get_move,
)
from arborist.scientist.observer import Observation, Observer
from arborist.scientist.scientist import RoundResult, Scientist, SessionResult

__all__ = [
    "Scientist",
    "SessionResult",
    "RoundResult",
    "Observer",
    "Observation",
    "HypothesisGenerator",
    "Hypothesis",
    "Journal",
    "JournalEntry",
    "ResearchMove",
    "MoveCategory",
    "MOVE_LIBRARY",
    "get_move",
]
