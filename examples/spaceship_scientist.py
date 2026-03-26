#!/usr/bin/env python3
"""
Arborist Scientist x Spaceship Titanic
=======================================
Full script generation mode: no existing training script needed.
Arborist generates complete training pipelines from scratch using ProblemSpec.

Usage:
  python3 examples/spaceship_scientist.py
  python3 examples/spaceship_scientist.py --max-rounds 5 --budget 80
  python3 examples/spaceship_scientist.py --codegen-model openrouter/anthropic/claude-sonnet-4-6
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from arborist.scientist import Scientist, ProblemSpec

DATA_DIR = Path(__file__).parent.parent / "data" / "spaceship-titanic"
SYSTEM_PYTHON = "/opt/homebrew/bin/python3"

# Quick baseline: logistic regression with minimal preprocessing
BASELINE_SCRIPT = '''
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, accuracy_score
import sys, os

data_dir = os.environ.get("DATA_DIR", "data/spaceship-titanic")
df = pd.read_csv(f"{data_dir}/train.csv")

# Minimal preprocessing
df["Transported"] = df["Transported"].astype(int)

# Extract cabin deck/num/side
df[["Deck", "CabinNum", "Side"]] = df["Cabin"].str.split("/", expand=True)
df["CabinNum"] = pd.to_numeric(df["CabinNum"], errors="coerce")

# Encode categoricals
for col in ["HomePlanet", "Destination", "Deck", "Side"]:
    df[col] = LabelEncoder().fit_transform(df[col].fillna("Unknown"))

df["CryoSleep"] = df["CryoSleep"].map({True: 1, False: 0, "True": 1, "False": 0}).fillna(0).astype(int)
df["VIP"] = df["VIP"].map({True: 1, False: 0, "True": 1, "False": 0}).fillna(0).astype(int)

spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
for col in spend_cols:
    df[col] = df[col].fillna(0)
df["TotalSpend"] = df[spend_cols].sum(axis=1)
df["Age"] = df["Age"].fillna(df["Age"].median())

features = ["HomePlanet", "CryoSleep", "Destination", "Age", "VIP",
            "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
            "Deck", "CabinNum", "Side", "TotalSpend"]

X = df[features].fillna(0).values
y = df["Transported"].values

# 5-fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1s, accs = [], []
for train_idx, val_idx in skf.split(X, y):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_tr, y[train_idx])
    preds = model.predict(X_val)
    f1s.append(f1_score(y[val_idx], preds))
    accs.append(accuracy_score(y[val_idx], preds))

print(f"METRICS val_loss=0.0000 val_f1={np.mean(f1s):.4f} val_accuracy={np.mean(accs):.4f} val_kappa=0.0000")
'''


def run_baseline() -> dict:
    """Run the baseline script and return parsed metrics."""
    baseline_path = DATA_DIR / "_baseline.py"
    baseline_path.write_text(BASELINE_SCRIPT)

    result = subprocess.run(
        [SYSTEM_PYTHON, str(baseline_path)],
        capture_output=True, text=True, timeout=60,
        env={**__import__("os").environ, "DATA_DIR": str(DATA_DIR)},
    )

    metrics = {}
    m = re.search(r"val_f1=([\d.]+)\s+val_accuracy=([\d.]+)", result.stdout)
    if m:
        metrics["f1"] = float(m.group(1))
        metrics["accuracy"] = float(m.group(2))
    else:
        print(f"Baseline stderr: {result.stderr[:500]}")
        metrics["f1"] = 0.0
        metrics["accuracy"] = 0.0

    return metrics


def score_experiment(results: dict) -> float:
    """Extract F1 from experiment results."""
    return results.get("f1", results.get("val_f1", 0.0))


def on_round_complete(round_result):
    imp = round_result.improvement
    imp_str = f" ({imp:+.4f})" if imp is not None else ""
    print(f"\n{'='*60}")
    print(f"Round {round_result.round_number} complete: {round_result.outcome}{imp_str}")
    print(f"Hypothesis: {round_result.hypothesis.description}")
    print(f"Experiments used: {round_result.experiments_used}")
    for finding in round_result.findings:
        print(f"  Finding: {finding}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Arborist Scientist on Spaceship Titanic")
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--model", type=str, default="openrouter/google/gemini-2.5-flash")
    parser.add_argument("--mutator-model", type=str, default="openrouter/google/gemini-2.5-flash")
    parser.add_argument("--codegen-model", type=str, default="openrouter/anthropic/claude-sonnet-4-6")
    parser.add_argument("--db-path", type=str, default="./spaceship_scientist.db")
    parser.add_argument("--journal-dir", type=str, default="./spaceship_journals")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    print("=" * 60)
    print("Arborist Scientist - Spaceship Titanic (Full Script Gen)")
    print("=" * 60)

    # Run baseline
    print("Running baseline (logistic regression + minimal features)...")
    baseline = run_baseline()
    baseline_f1 = baseline["f1"]
    print(f"Baseline F1: {baseline_f1:.4f}")
    print(f"Baseline Accuracy: {baseline.get('accuracy', 0):.4f}")
    print()

    spec = ProblemSpec(
        name="spaceship_titanic",
        description=(
            f"Binary classification: predict whether a passenger was Transported to an alternate dimension. "
            f"Dataset: Spaceship Titanic (Kaggle), 8693 training samples, 13 features. "
            f"Features: PassengerId (group_num), HomePlanet (Earth/Europa/Mars), CryoSleep (bool), "
            f"Cabin (deck/num/side format), Destination (3 planets), Age (float), VIP (bool), "
            f"5 spending columns (RoomService, FoodCourt, ShoppingMall, Spa, VRDeck), Name. "
            f"Target: Transported (bool). ~50/50 class balance. "
            f"Key feature engineering opportunities: cabin parsing (deck/side strongly predictive), "
            f"group features from PassengerId (family traveling together), spending patterns "
            f"(CryoSleep passengers have zero spend), age binning, interaction features. "
            f"Current baseline: F1={baseline_f1:.4f} (logistic regression, basic features). "
            f"Top Kaggle scores: ~0.81 accuracy. Use 5-fold stratified CV."
        ),
        dataset_path=str(DATA_DIR),
        target_variable="Transported",
        metric="f1",
        task_type="classification",
        constraints=[
            "Use 5-fold stratified cross-validation",
            "Report mean F1 and accuracy across folds",
            "No test set leakage (test.csv is held out for Kaggle submission)",
            "Only use packages available: numpy, pandas, scikit-learn, xgboost, scipy. Do NOT use lightgbm, catboost, pytorch, or tensorflow.",
        ],
        data_description=(
            "CSV with columns: PassengerId, HomePlanet, CryoSleep, Cabin, Destination, "
            "Age, VIP, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck, Name, Transported. "
            "Missing values present in most columns. Cabin format: Deck/Num/Side. "
            "PassengerId format: GroupNum_PersonNum (people in same group are traveling together)."
        ),
        forbidden_patterns=[
            "Do not use test.csv for training or validation",
            "Do not use Name as a direct feature (data leakage risk)",
        ],
        python_cmd=SYSTEM_PYTHON,
        timeout=120,
        extra_context=(
            f"Data directory: {DATA_DIR}\n"
            f"Load with: pd.read_csv('{DATA_DIR}/train.csv')\n"
            "Print metrics as: METRICS val_f1=X.XXXX val_accuracy=X.XXXX\n"
            "The METRICS line must be on stdout for metric parsing."
        ),
    )

    scientist = Scientist(
        problem=spec.description,
        problem_spec=spec,
        executor=None,  # Full script gen mode - no executor needed
        score=score_experiment,
        baseline_config={},
        metric_name="f1",
        max_rounds=args.max_rounds,
        total_budget=args.budget,
        model=args.model,
        mutator_model=args.mutator_model,
        codegen_model=args.codegen_model,
        db_path=args.db_path,
        journal_dir=args.journal_dir,
        on_round_complete=on_round_complete,
        verbose=args.verbose,
    )

    print(f"Session ID: {scientist.session_id}")
    print(f"Codegen model: {args.codegen_model}")
    print(f"Hypothesis/mutation model: {args.model}")
    print(f"Starting research loop...\n")

    result = scientist.run()

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(result.summary())
    print(f"\nBaseline F1: {baseline_f1:.4f}")
    if result.best_score is not None:
        delta = result.best_score - baseline_f1
        print(f"Best F1:     {result.best_score:.4f} ({delta:+.4f} vs baseline)")
    print(f"Total experiments: {result.total_experiments}")
    print(f"Duration: {result.duration_seconds:.1f}s")

    report_path = Path(args.journal_dir) / f"report_{scientist.session_id}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "session_id": scientist.session_id,
        "dataset": "spaceship_titanic",
        "baseline_f1": baseline_f1,
        "best_f1": result.best_score,
        "best_config": result.best_config,
        "total_experiments": result.total_experiments,
        "duration_seconds": result.duration_seconds,
        "rounds": [
            {
                "round": r.round_number,
                "hypothesis": r.hypothesis.description,
                "outcome": r.outcome,
                "score_before": r.score_before,
                "score_after": r.score_after,
                "improvement": r.improvement,
                "experiments_used": r.experiments_used,
                "findings": r.findings,
            }
            for r in result.rounds
        ],
    }
    report_path.write_text(json.dumps(report_data, indent=2))
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
