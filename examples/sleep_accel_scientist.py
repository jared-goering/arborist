#!/usr/bin/env python3
"""
Arborist Scientist × SleepAccel Integration
============================================
Runs the autonomous research agent on the SleepAccel sleep staging dataset.
The scientist observes model performance, forms hypotheses about what to try,
designs experiments, and runs them via Arborist tree search.

Usage:
  python3 examples/sleep_accel_scientist.py
  python3 examples/sleep_accel_scientist.py --max-rounds 5 --budget 100
  python3 examples/sleep_accel_scientist.py --human-in-the-loop
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from arborist.scientist import Scientist, Observer, Observation

TRAIN_SCRIPT = Path.home() / "Projects" / "sleep-ai" / "train_xgb.py"
SYSTEM_PYTHON = "/opt/homebrew/bin/python3"

# Known baseline from 199 linear autoresearch experiments
LINEAR_BASELINE_F1 = 0.3632
LINEAR_BASELINE_CONFIG = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "min_samples_leaf": 10,
    "subsample": 0.8,
}

PARAM_BOUNDS = {
    "n_estimators": (50, 1500, "int"),
    "max_depth": (2, 15, "int"),
    "learning_rate": (0.005, 0.3, "float"),
    "min_samples_leaf": (1, 50, "int"),
    "subsample": (0.4, 1.0, "float"),
}

SLEEP_STAGES = ["Wake", "N1", "N2", "N3", "REM"]


def parse_train_output(stdout: str) -> dict:
    """Parse the full output of train_xgb.py into structured results.

    Extracts: classification_report, confusion_matrix, feature_importance,
    class_labels, f1, accuracy, kappa, training_time, feature_count, etc.
    """
    results = {"raw_output": stdout}

    # Parse METRICS line: val_loss=0.0000 val_f1=0.2769 val_accuracy=0.3514 val_kappa=0.0818
    metrics_match = re.search(
        r"val_f1=([\d.]+)\s+val_accuracy=([\d.]+)\s+val_kappa=([\d.]+)",
        stdout,
    )
    if metrics_match:
        results["f1"] = float(metrics_match.group(1))
        results["accuracy"] = float(metrics_match.group(2))
        results["kappa"] = float(metrics_match.group(3))

    # Parse training time
    time_match = re.search(r"Training time:\s+([\d.]+)s", stdout)
    if time_match:
        results["training_time"] = float(time_match.group(1))

    # Parse feature count
    feat_match = re.search(r"(\d+) total\s*$", stdout, re.MULTILINE)
    if feat_match:
        results["feature_count"] = int(feat_match.group(1))

    # Parse classification report into dict
    # Format:
    #               precision    recall  f1-score   support
    #         Wake       0.35      0.38      0.36       399
    report = {}
    report_pattern = re.compile(
        r"^\s+(\w+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$",
        re.MULTILINE,
    )
    for m in report_pattern.finditer(stdout):
        name = m.group(1)
        report[name] = {
            "precision": float(m.group(2)),
            "recall": float(m.group(3)),
            "f1-score": float(m.group(4)),
            "support": int(m.group(5)),
        }
    if report:
        results["classification_report"] = report

    # Parse confusion matrix
    # Format:
    #           Wake    N1    N2    N3   REM
    #     Wake   150     3    77    51   118
    cm_section = re.search(r"Confusion Matrix:\s*\n(.+?)(?:\n\n|\nTop)", stdout, re.DOTALL)
    if cm_section:
        cm_lines = cm_section.group(1).strip().split("\n")
        # First line is header with class names
        if len(cm_lines) > 1:
            matrix = []
            labels = []
            for line in cm_lines[1:]:  # skip header
                parts = line.split()
                if len(parts) >= 6:  # label + 5 values
                    labels.append(parts[0])
                    matrix.append([int(x) for x in parts[1:6]])
            if matrix:
                results["confusion_matrix"] = matrix
                results["class_labels"] = labels

    # Parse feature importance
    # Format:
    #   1. epoch_position: 0.0592
    fi = {}
    fi_pattern = re.compile(r"^\s+\d+\.\s+(\S+):\s+([\d.]+)\s*$", re.MULTILINE)
    for m in fi_pattern.finditer(stdout):
        fi[m.group(1)] = float(m.group(2))
    if fi:
        results["feature_importance"] = fi

    # Parse class distribution
    # Format:
    #   Wake: 2633 (10.6%)
    class_dist = {}
    dist_pattern = re.compile(r"^\s+(\w+):\s+(\d+)\s+\(([\d.]+)%\)\s*$", re.MULTILINE)
    for m in dist_pattern.finditer(stdout):
        class_dist[m.group(1)] = {
            "count": int(m.group(2)),
            "percent": float(m.group(3)),
        }
    if class_dist:
        results["class_distribution"] = class_dist

    # Parse train/val split
    split_match = re.search(r"Train:\s+(\d+)\s+\|\s+Val:\s+(\d+)", stdout)
    if split_match:
        results["train_size"] = int(split_match.group(1))
        results["val_size"] = int(split_match.group(2))

    return results


def run_experiment(config: dict) -> dict:
    """Execute train_xgb.py with given config and return structured results."""
    cmd = [
        SYSTEM_PYTHON, "-u", str(TRAIN_SCRIPT),
        "--n-estimators", str(int(config.get("n_estimators", 300))),
        "--max-depth", str(int(config.get("max_depth", 6))),
        "--learning-rate", str(config.get("learning_rate", 0.1)),
        "--min-samples-leaf", str(int(config.get("min_samples_leaf", 10))),
        "--subsample", str(config.get("subsample", 0.8)),
    ]

    start = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            cwd=str(TRAIN_SCRIPT.parent),
        )
        elapsed = time.time() - start

        if result.returncode != 0:
            return {
                "error": result.stderr[:500],
                "f1": 0.0,
                "wall_time": elapsed,
            }

        parsed = parse_train_output(result.stdout)
        parsed["wall_time"] = elapsed
        return parsed

    except subprocess.TimeoutExpired:
        return {"error": "Timeout after 120s", "f1": 0.0, "wall_time": 120.0}
    except Exception as e:
        return {"error": str(e), "f1": 0.0, "wall_time": time.time() - start}


def score_experiment(results: dict) -> float:
    """Extract the F1 score from experiment results.

    Normal experiments (run_experiment) store the score as 'f1'.
    Code-gen experiments (CodeGeneratorExecutor) store it as 'val_f1'
    because _parse_metrics preserves the key name from stdout.
    """
    return results.get("f1", results.get("val_f1", 0.0))


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
    parser = argparse.ArgumentParser(description="Arborist Scientist on SleepAccel")
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--model", type=str, default="openrouter/anthropic/claude-sonnet-4-6")
    parser.add_argument("--mutator-model", type=str, default="openrouter/anthropic/claude-haiku-4-5")
    parser.add_argument("--codegen-model", type=str, default=None,
                        help="Model for code generation (defaults to --model)")
    parser.add_argument("--human-in-the-loop", action="store_true")
    parser.add_argument("--db-path", type=str, default="./sleep_scientist.db")
    parser.add_argument("--journal-dir", type=str, default="./sleep_journals")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    print("=" * 60)
    print("Arborist Scientist — SleepAccel Sleep Staging")
    print("=" * 60)
    print(f"Dataset: SleepAccel (31 subjects, 5-class sleep staging)")
    print(f"Baseline: F1={LINEAR_BASELINE_F1} (199 linear experiments)")
    print(f"Budget: {args.budget} experiments across {args.max_rounds} rounds")
    print(f"Scientist model: {args.model}")
    print(f"Mutator model: {args.mutator_model}")
    print()

    # Verify train script exists
    if not TRAIN_SCRIPT.exists():
        print(f"ERROR: {TRAIN_SCRIPT} not found")
        sys.exit(1)

    # Run baseline first to get structured observation
    print("Running baseline experiment...")
    baseline_results = run_experiment(LINEAR_BASELINE_CONFIG)
    baseline_f1 = baseline_results.get("f1", 0.0)
    print(f"Baseline F1: {baseline_f1:.4f}")
    print()

    def parse_code_gen_results(results: dict) -> dict:
        """Parse stdout from code-gen experiments into structured data.

        Code-gen experiments return raw stdout in results['stdout'].
        This parses it the same way we parse normal experiment output,
        so the Observer gets classification_report, confusion_matrix, etc.
        """
        stdout = results.get("stdout", "")
        if not stdout:
            return results
        parsed = parse_train_output(stdout)
        return parsed

    scientist = Scientist(
        problem=(
            "Improve 5-class sleep staging (Wake/N1/N2/N3/REM) from wrist accelerometry + heart rate. "
            f"Dataset: SleepAccel, 31 subjects, ~25K epochs, 58 features. "
            f"Current best F1: {baseline_f1:.4f}. "
            f"Key challenges: N1 is nearly undetectable (F1~0.01), severe class imbalance "
            f"(N2=48% vs N1=7%), only 31 subjects limits generalization. "
            f"Hyperparameters: n_estimators, max_depth, learning_rate, min_samples_leaf, subsample. "
            f"Model: XGBoost with sample weighting for class imbalance."
        ),
        executor=run_experiment,
        score=score_experiment,
        baseline_config=LINEAR_BASELINE_CONFIG,
        metric_name="f1_macro",
        max_rounds=args.max_rounds,
        total_budget=args.budget,
        model=args.model,
        mutator_model=args.mutator_model,
        codegen_model=args.codegen_model,
        db_path=args.db_path,
        journal_dir=args.journal_dir,
        memory_url="http://localhost:8642",
        human_in_the_loop=args.human_in_the_loop,
        on_round_complete=on_round_complete,
        results_parser=parse_code_gen_results,
        verbose=args.verbose,
        base_script=str(TRAIN_SCRIPT),
    )

    print(f"Session ID: {scientist.session_id}")
    print(f"Starting research loop...\n")

    result = scientist.run()

    # Final report
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(result.summary())
    print(f"\nBaseline F1: {LINEAR_BASELINE_F1:.4f}")
    if result.best_score is not None:
        delta = result.best_score - LINEAR_BASELINE_F1
        print(f"Best F1:     {result.best_score:.4f} ({delta:+.4f} vs linear baseline)")
    print(f"Total experiments: {result.total_experiments}")
    print(f"Duration: {result.duration_seconds:.1f}s")

    # Save report
    report_path = Path(args.journal_dir) / f"report_{scientist.session_id}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "session_id": scientist.session_id,
        "problem": result.problem,
        "baseline_f1": LINEAR_BASELINE_F1,
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
