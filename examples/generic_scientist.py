#!/usr/bin/env python3
"""
Generic Scientist Example
=========================
Demonstrates using Arborist on a new problem WITHOUT any existing script.
The FullScriptGenerator creates complete training scripts from a ProblemSpec.

Usage:
  python3 examples/generic_scientist.py
  python3 examples/generic_scientist.py --max-rounds 5 --budget 50
  python3 examples/generic_scientist.py --codegen-model openrouter/anthropic/claude-sonnet-4-6
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from arborist.scientist import Scientist
from arborist.scientist.problem_spec import ProblemSpec


def score_experiment(results: dict) -> float:
    """Extract the primary metric from experiment results."""
    return results.get("val_f1_macro", results.get("val_f1", 0.0))


def on_round_complete(round_result):
    """Callback after each scientist round."""
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
    parser = argparse.ArgumentParser(
        description="Arborist Generic Scientist — generate scripts from scratch"
    )
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--model", type=str, default="openrouter/google/gemini-2.5-flash")
    parser.add_argument("--mutator-model", type=str, default="openrouter/google/gemini-2.5-flash")
    parser.add_argument("--codegen-model", type=str, default="openrouter/anthropic/claude-sonnet-4-6")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to dataset CSV file")
    parser.add_argument("--target", type=str, required=True,
                        help="Name of the target variable column")
    parser.add_argument("--metric", type=str, default="f1_macro",
                        help="Primary evaluation metric")
    parser.add_argument("--task-type", type=str, default="classification",
                        choices=["classification", "regression", "time_series"])
    parser.add_argument("--db-path", type=str, default="./generic_scientist.db")
    parser.add_argument("--journal-dir", type=str, default="./generic_journals")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path).resolve()
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        sys.exit(1)

    # Build ProblemSpec from CLI args
    spec = ProblemSpec(
        name=dataset_path.stem,
        description=f"Build a {args.task_type} model for {dataset_path.stem}",
        dataset_path=str(dataset_path),
        target_variable=args.target,
        metric=args.metric,
        task_type=args.task_type,
        data_description=f"CSV file at {dataset_path}",
    )

    print("=" * 60)
    print("Arborist Generic Scientist — Full Script Generation")
    print("=" * 60)
    print(f"Dataset: {dataset_path}")
    print(f"Target: {args.target}")
    print(f"Task: {args.task_type}")
    print(f"Metric: {args.metric}")
    print(f"Budget: {args.budget} experiments across {args.max_rounds} rounds")
    print(f"Codegen model: {args.codegen_model}")
    print()

    scientist = Scientist(
        problem=spec.description,
        problem_spec=spec,
        executor=lambda config: {},  # Not used for full-gen mode
        score=score_experiment,
        metric_name=args.metric,
        max_rounds=args.max_rounds,
        total_budget=args.budget,
        model=args.model,
        mutator_model=args.mutator_model,
        codegen_model=args.codegen_model,
        db_path=args.db_path,
        journal_dir=args.journal_dir,
        memory_url=None,
        on_round_complete=on_round_complete,
        verbose=args.verbose,
    )

    print(f"Session ID: {scientist.session_id}")
    print("Starting research loop...\n")

    result = scientist.run()

    # Final report
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(result.summary())
    if result.best_score is not None:
        print(f"\nBest {args.metric}: {result.best_score:.4f}")
    print(f"Total experiments: {result.total_experiments}")
    print(f"Duration: {result.duration_seconds:.1f}s")

    # Save report
    report_path = Path(args.journal_dir) / f"report_{scientist.session_id}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "session_id": scientist.session_id,
        "problem": result.problem,
        "problem_spec": {
            "name": spec.name,
            "dataset_path": spec.dataset_path,
            "target_variable": spec.target_variable,
            "metric": spec.metric,
            "task_type": spec.task_type,
        },
        "best_score": result.best_score,
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
