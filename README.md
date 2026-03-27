# 🌳 Arborist

[![Tests](https://github.com/jared-goering/arborist/actions/workflows/test.yml/badge.svg)](https://github.com/jared-goering/arborist/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/arborist-ai)](https://pypi.org/project/arborist-ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Tree search for ML experiments.** Define a goal. Arborist branches, evaluates, prunes, and converges. Like MCTS, but for hyperparameter tuning and feature engineering.

```
pip install arborist-ai
```

## Why tree search?

Linear hyperparameter sweeps (grid, random, Bayesian) explore one path at a time. When they hit a local optimum, they're stuck.

Arborist treats your experiment space as a **tree**. It branches into multiple directions simultaneously, evaluates results, prunes dead ends, and doubles down on what works. The same way AlphaGo explores game states, but for ML experiments.

**Results from our benchmarks:**

| Dataset | Strategy | F1 Score | Experiments | Wall Time |
|---------|----------|----------|-------------|-----------|
| Forest Cover Type | Linear + LLM | 0.8659 | 50 | 2454s |
| Forest Cover Type | Tree + LLM (UCB) | 0.8683 | 50 | 1793s |
| Forest Cover Type | **Hybrid (explore + exploit)** | **0.8750** | 50 | **1186s** |

The hybrid strategy found a better solution **2x faster** by exploring broadly with UCB, then hill-climbing from the best region.

## Quickstart

```python
from arborist import TreeSearch

search = TreeSearch(
    goal="Find optimal x",
    executor=lambda config: {"score": -(config["x"] - 3) ** 2 + 10},
    score=lambda r: r["score"],
    seed_configs=[{"x": 0}, {"x": 1}, {"x": 5}],
    strategy="ucb",
    max_experiments=50,
)

results = search.run()
print(f"Best: {results.best['score']:.4f}")
print(f"Config: {results.best['config']}")
```

## How it works

```
Seed configs
    │
    ├── Execute experiments (parallel)
    ├── Score results
    ├── Expand promising nodes (LLM or custom mutator)
    ├── Prune dead ends
    └── Repeat until budget/target/plateau
```

Everything persists to SQLite. Kill the process, restart later, pick up where you left off.

## Features

- **Tree search strategies**: UCB1 (explore/exploit balance), best-first (greedy), breadth-first (systematic), hybrid (adaptive phase switching)
- **LLM-guided mutations**: Uses any model via litellm to analyze results and suggest new configs. Falls back to random perturbation if no LLM available.
- **Parallel execution**: Run multiple experiments concurrently with configurable concurrency limits
- **SQLite persistence**: Every node, config, and result stored. Resume any search. Query history.
- **Custom everything**: Bring your own executor, evaluator, mutator, or strategy
- **Shell executor**: Point it at any training script. No code changes needed.
- **CLI included**: `arborist run`, `arborist status`, `arborist report`
- **Budget controls**: Cap by experiment count, wall time, dollar cost, or target score

## Strategies

| Strategy | How it works | Best for |
|----------|-------------|----------|
| `ucb` | UCB1 bandit algorithm. Balances exploration of untried branches with exploitation of high scorers. | General use. Unknown search spaces. |
| `best_first` | Always expands the highest-scoring node. Pure exploitation. | When you already know a good region. |
| `breadth_first` | Level-by-level. Every node at depth N before any at N+1. | Systematic coverage. Small spaces. |
| `hybrid` | Starts with UCB exploration, detects plateau, switches to greedy hill-climb from the best node. Cycles back to explore if exploit stalls. | **Best overall.** Finds good regions fast, then squeezes out gains. |
| `llm_guided` | LLM picks which node to expand based on full tree context. | When you want the model to drive strategy. |

## Real-world example: XGBoost tuning

```python
from arborist import TreeSearch, ShellExecutor, NumericEvaluator

search = TreeSearch(
    goal="Maximize macro F1 on multi-class classification",
    executor=ShellExecutor(
        command="python3 train.py --config {config_path}",
        timeout=300,
    ),
    evaluator=NumericEvaluator(field="f1"),
    seed_configs=[
        {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05},
        {"n_estimators": 500, "max_depth": 8, "learning_rate": 0.01},
    ],
    strategy="hybrid",
    concurrency=4,
    max_experiments=100,
    max_depth=5,
    plateau_window=15,
    db_path="./experiments.db",
    verbose=True,
)

results = search.run()
print(results.report())
```

Your training script just needs to print JSON with the metric:
```python
# train.py
import json, sys
config = json.load(open(sys.argv[2]))
# ... train your model ...
print(json.dumps({"f1": 0.847, "accuracy": 0.912}))
```

## API Reference

### TreeSearch

```python
search = TreeSearch(
    goal="...",                         # What you're optimizing
    executor=my_fn,                     # callable(config) -> dict
    score=lambda r: r["f1"],            # callable(results) -> float

    # Search control
    strategy="hybrid",                  # ucb, best_first, breadth_first, hybrid, llm_guided
    mutator=my_mutator,                 # Custom mutation function (optional, defaults to LLM)
    concurrency=5,                      # Max parallel experiments
    max_experiments=200,                # Total experiment budget
    max_depth=6,                        # Max tree depth

    # Termination
    target_score=0.95,                  # Stop when reached
    plateau_window=20,                  # Stop if no improvement for N experiments
    budget_usd=10.0,                    # LLM cost cap

    # Storage
    db_path="./arborist.db",            # SQLite path (auto-created)

    # Callbacks
    on_node_complete=callback,          # Called after each experiment
    verbose=True,
)

results = search.run()
```

### Results

```python
results.best              # Best node: config, score, full results
results.top_k(5)          # Top 5 nodes
results.insights          # Cross-branch pattern analysis
results.report()          # Markdown summary
results.tree_id           # Unique ID for resuming
```

### Resume a search

```python
search = TreeSearch.resume(
    tree_id="abc123",
    db_path="./arborist.db",
    executor=my_fn,
    score=my_score,
)
results = search.run()  # Picks up where it left off
```

### Custom mutator

```python
def my_mutator(config, results, context):
    """Generate child configs from a parent experiment."""
    return [
        {**config, "lr": config["lr"] * 0.5},
        {**config, "lr": config["lr"] * 2.0},
        {**config, "n_estimators": config["n_estimators"] + 100},
    ]
```

### Custom executor

```python
from arborist import Executor, BranchContext

class MyExecutor(Executor):
    def run(self, config: dict, context: BranchContext) -> dict:
        # context.goal, context.depth, context.parent_config, etc.
        model = train(**config)
        return {"f1": model.f1, "accuracy": model.accuracy}
```

## CLI

```bash
arborist run --config search.yaml       # Run from YAML config
arborist status --db ./arborist.db      # Check progress
arborist report --tree-id ID            # Generate markdown report
arborist list --db ./arborist.db        # List all searches
arborist node NODE_ID                   # Inspect a specific node
arborist prune NODE_ID --reason "..."   # Manually kill a branch
```

### YAML config

```yaml
goal: "Maximize F1 for multi-class classification"
strategy: hybrid
concurrency: 4
max_experiments: 100
max_depth: 5
db_path: ./arborist.db

executor:
  type: shell
  command: "python3 train.py --config {config_path}"
  timeout: 300

evaluator:
  type: numeric
  field: f1

seed_configs:
  - n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
  - n_estimators: 200
    max_depth: 4
    learning_rate: 0.05

termination:
  target_score: 0.95
  plateau_window: 15
```

## Design

1. **Local-first.** SQLite storage, no cloud, no accounts, no telemetry.
2. **LLM-agnostic.** Any provider via litellm (OpenAI, Anthropic, Google, Ollama, etc).
3. **Composable.** Swap out any component: executor, evaluator, mutator, strategy.
4. **Resumable.** Full state in SQLite. Kill and restart without losing work.
5. **Observable.** Verbose logging, per-node callbacks, CLI status checks.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT
