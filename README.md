# Arborist

Agentic tree search engine for parallelized experiment orchestration. Define a goal, let it branch, evaluate, prune, and converge.

**Local-first. LLM-agnostic. Resumable. Composable.**

```
pip install arborist-ai
```

## Quickstart

```python
from arborist import TreeSearch

# Define your experiment
def my_experiment(config):
    x = config["x"]
    return {"score": -(x - 3) ** 2 + 10}

# Run a search
search = TreeSearch(
    goal="Find optimal x",
    executor=my_experiment,
    score=lambda r: r["score"],
    seed_configs=[{"x": 0}, {"x": 1}, {"x": 5}],
    strategy="ucb",
    max_experiments=50,
    max_depth=4,
)

results = search.run()
print(f"Best: {results.best['score']:.4f}")
print(f"Config: {results.best['config']}")
print(results.report())
```

## How It Works

Arborist explores a search space as a tree:

1. **Seed** — Start with initial configurations
2. **Execute** — Run experiments in parallel
3. **Evaluate** — Score each result
4. **Expand** — Generate child configs from promising results (via LLM or custom mutator)
5. **Prune** — Cut low-performing branches
6. **Repeat** — Until termination criteria are met

Everything persists to SQLite. Kill and restart anytime.

## API Reference

### TreeSearch

```python
search = TreeSearch(
    goal="Maximize F1 for multi-class classification",
    executor=my_fn,                     # callable(config) -> dict
    score=lambda r: r["f1"],            # callable(results) -> float
    seed_configs=[                      # Initial experiments
        {"lr": 0.01, "features": ["feat_1", "feat_2"]},
        {"lr": 0.001, "features": ["feat_1", "feat_3"]},
    ],

    # Optional
    strategy="ucb",                     # ucb, best_first, breadth_first
    mutator=my_mutator,                 # callable(config, results, context) -> list[dict]
    concurrency=5,                      # Max parallel branches
    max_experiments=200,                # Total experiment budget
    max_depth=6,                        # Max tree depth
    budget_usd=10.0,                    # Cost cap
    target_score=0.95,                  # Stop when reached
    plateau_window=20,                  # Stop if no improvement
    db_path="./arborist.db",            # SQLite path
    on_node_complete=callback,          # Called after each experiment
    verbose=True,
)

results = search.run()
```

### Results

```python
results.best              # Best node (config + score + results)
results.top_k(5)          # Top 5 nodes
results.insights          # Cross-branch insights
results.report()          # Markdown report
results.tree_id           # For resume
```

### Resume

```python
search = TreeSearch.resume(
    tree_id="abc123",
    db_path="./arborist.db",
    executor=my_fn,
    score=my_score,
)
results = search.run()
```

### Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `ucb` | UCB1 — balances exploration and exploitation | General use, unknown search spaces |
| `best_first` | Always expands highest-scoring node | Exploitation-heavy, known good regions |
| `breadth_first` | Level-by-level exploration | Systematic coverage, small search spaces |

### Custom Executor

```python
from arborist import Executor, BranchContext

class MyExecutor(Executor):
    def run(self, config: dict, context: BranchContext) -> dict:
        # context.goal, context.depth, context.parent_config, etc.
        return {"metric": train_model(**config)}
```

### Shell Executor

```python
from arborist import ShellExecutor

executor = ShellExecutor(
    command="python3 train.py --config {config_path}",
    timeout=300,
)
```

The shell executor substitutes `{key}` placeholders from the config dict. Use `{config_path}` to write the full config to a temp JSON file.

### Custom Mutator

```python
def my_mutator(config, results, context):
    """Generate child configs from a completed experiment."""
    return [
        {**config, "lr": config["lr"] * 0.5},
        {**config, "lr": config["lr"] * 2.0},
    ]
```

If no mutator is provided, Arborist uses an LLM-based mutator (via litellm) that analyzes results and suggests new configs. Falls back to random perturbation if no LLM is available.

### Custom Evaluator

```python
from arborist import NumericEvaluator

# By field name
evaluator = NumericEvaluator(field="metrics.f1")

# By function
evaluator = NumericEvaluator(fn=lambda r: r["precision"] * r["recall"])
```

## CLI

```bash
# Run from YAML config
arborist run --config search.yaml

# Check status
arborist status [--tree-id ID] [--db ./arborist.db]

# Generate report
arborist report [--tree-id ID] [--format markdown|json] [--output report.md]

# List all trees
arborist list [--db ./arborist.db]

# Show node details
arborist node NODE_ID [--db ./arborist.db]

# Manually prune a branch
arborist prune NODE_ID --reason "Manual prune" [--db ./arborist.db]
```

### YAML Config

```yaml
goal: "Maximize F1 for multi-class classification"
strategy: ucb
concurrency: 5
max_experiments: 200
max_depth: 6
db_path: ./arborist.db

executor:
  type: shell
  command: "python3 train.py --config {config_path}"
  timeout: 300

evaluator:
  type: numeric
  field: f1

seed_configs:
  - lr: 0.01
    features: [feat_1, feat_2, feat_3]
  - lr: 0.001
    features: [feat_1, feat_4, feat_5]

termination:
  target_score: 0.95
  plateau_window: 20
  budget_usd: 10.0
```

## Design Principles

1. **Local-first.** SQLite, no cloud, no accounts.
2. **LLM-agnostic.** litellm means any provider works.
3. **Composable.** Bring your own executor, evaluator, mutator, strategy.
4. **Resumable.** Everything persists to SQLite. Kill and restart anytime.
5. **Observable.** Verbose logging, callbacks, CLI status.
6. **Minimal dependencies.** Just litellm, click, and pyyaml.

## License

MIT
