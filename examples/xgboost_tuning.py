#!/usr/bin/env python3
"""
XGBoost Hyperparameter Tuning with Arborist
============================================
Uses tree search to optimize XGBoost hyperparameters on scikit-learn's
wine dataset. Demonstrates branching, pruning, and convergence.

Usage:
  pip install xgboost scikit-learn
  python3 examples/xgboost_tuning.py
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from arborist import TreeSearch

# Load a standard dataset
X, y = load_wine(return_X_y=True)


def train_and_evaluate(config: dict) -> dict:
    """Train XGBoost with given hyperparameters and return CV F1 score."""
    clf = XGBClassifier(
        n_estimators=int(config.get("n_estimators", 100)),
        max_depth=int(config.get("max_depth", 6)),
        learning_rate=config.get("learning_rate", 0.1),
        subsample=config.get("subsample", 0.8),
        colsample_bytree=config.get("colsample_bytree", 0.8),
        min_child_weight=config.get("min_child_weight", 1),
        reg_alpha=config.get("reg_alpha", 0.0),
        reg_lambda=config.get("reg_lambda", 1.0),
        random_state=42,
        verbosity=0,
    )
    scores = cross_val_score(clf, X, y, cv=5, scoring="f1_macro")
    return {"f1": float(np.mean(scores)), "std": float(np.std(scores))}


# Seed configurations covering different regions of hyperparameter space
seeds = [
    {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.3, "subsample": 1.0},
    {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.7},
    {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1, "subsample": 0.8},
]

search = TreeSearch(
    goal="Maximize F1 macro for wine classification via XGBoost hyperparameter tuning",
    executor=train_and_evaluate,
    score=lambda r: r["f1"],
    seed_configs=seeds,
    strategy="hybrid",       # explore (UCB+LLM) then exploit (greedy hill-climb)
    max_experiments=30,
    max_depth=4,
    concurrency=3,
    param_bounds={
        "n_estimators": (50, 500),
        "max_depth": (2, 10),
        "learning_rate": (0.01, 0.5),
        "subsample": (0.5, 1.0),
        "colsample_bytree": (0.3, 1.0),
        "min_child_weight": (1, 10),
        "reg_alpha": (0.0, 5.0),
        "reg_lambda": (0.0, 5.0),
    },
)

results = search.run()
print(f"\nBest F1: {results.best_score:.4f}")
print(f"Config:  {results.best_config}")
print(f"Experiments: {results.total_experiments}")
print(f"Duration: {results.duration_seconds:.1f}s")
