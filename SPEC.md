# Arborist — Agentic Tree Search Engine

## Overview
Open-source Python package for parallelized experiment orchestration. Define a goal, let it branch, evaluate, prune, and converge. Local-first, LLM-agnostic, SQLite storage, zero cloud dependencies.

`pip install arborist`

## Package Structure

```
arborist/
├── __init__.py          # Public API exports
├── tree.py              # TreeSearch class — main entry point
├── manager.py           # Tree manager, DAG state, node lifecycle
├── store.py             # SQLite persistence (trees, nodes, insights)
├── strategies/
│   ├── __init__.py
│   ├── base.py          # Abstract strategy interface
│   ├── ucb.py           # Upper Confidence Bound (MCTS-style)
│   ├── best_first.py    # Always expand highest-scoring node
│   ├── breadth_first.py # Level-by-level exploration
│   └── llm_guided.py    # LLM picks which branches to expand (uses litellm)
├── executors/
│   ├── __init__.py
│   ├── base.py          # Abstract executor interface
│   ├── python.py        # Run Python callables
│   └── shell.py         # Run shell commands
├── evaluators/
│   ├── __init__.py
│   ├── base.py          # Abstract evaluator interface
│   ├── numeric.py       # Score by numeric metric
│   └── llm.py           # LLM judges quality (uses litellm)
├── synthesis.py         # Cross-branch insight extraction, report generation
├── memory.py            # Optional Ultramemory integration (HTTP to localhost:8642)
├── cli.py               # CLI entry point (click or argparse)
└── server.py            # Optional web UI server (Phase 6, skip for now)
```

## Data Model (SQLite)

```sql
CREATE TABLE trees (
    id TEXT PRIMARY KEY,
    goal TEXT NOT NULL,
    strategy TEXT NOT NULL DEFAULT 'ucb',
    status TEXT NOT NULL DEFAULT 'running',  -- running, paused, completed, failed
    config TEXT,          -- JSON: max_depth, max_nodes, concurrency, budget_usd, termination criteria
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE nodes (
    id TEXT PRIMARY KEY,
    tree_id TEXT NOT NULL REFERENCES trees(id),
    parent_id TEXT REFERENCES nodes(id),  -- NULL for root/seed nodes
    depth INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending, running, completed, failed, pruned
    hypothesis TEXT,      -- What this branch is testing (human-readable)
    config TEXT NOT NULL,  -- JSON: experiment parameters
    results TEXT,         -- JSON: structured output from executor
    score REAL,           -- Numeric score from evaluator
    cost_usd REAL,        -- Cost of this experiment (API calls, compute)
    duration_ms INTEGER,  -- Wall-clock time
    pruned INTEGER NOT NULL DEFAULT 0,
    prune_reason TEXT,
    error TEXT,           -- Error message if failed
    created_at TEXT NOT NULL,
    completed_at TEXT,
    FOREIGN KEY (tree_id) REFERENCES trees(id)
);

CREATE TABLE insights (
    id TEXT PRIMARY KEY,
    tree_id TEXT NOT NULL REFERENCES trees(id),
    source_node_ids TEXT NOT NULL,  -- JSON array of node IDs
    type TEXT NOT NULL,    -- discovery, contradiction, convergence, dead_end, plateau
    content TEXT NOT NULL,
    confidence REAL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (tree_id) REFERENCES trees(id)
);

-- Indexes
CREATE INDEX idx_nodes_tree ON nodes(tree_id);
CREATE INDEX idx_nodes_parent ON nodes(parent_id);
CREATE INDEX idx_nodes_status ON nodes(status);
CREATE INDEX idx_nodes_score ON nodes(score);
CREATE INDEX idx_insights_tree ON insights(tree_id);
```

## Core Classes

### TreeSearch (main entry point)

```python
from arborist import TreeSearch

search = TreeSearch(
    goal="Maximize F1 for sleep staging from wearable data",
    executor=my_experiment_fn,     # callable(config) -> dict
    score=lambda r: r["f1"],       # callable(results) -> float
    seed_configs=[                 # Initial experiments to try
        {"lr": 0.01, "features": ["accel_mean"]},
        {"lr": 0.001, "features": ["accel_mean", "hr_mean"]},
    ],
    # Optional:
    strategy="ucb",                # ucb, best_first, breadth_first, llm_guided
    mutator=None,                  # callable(config, results) -> list[dict] (generate children)
    concurrency=5,                 # Max parallel branches
    max_experiments=200,           # Total experiment budget
    max_depth=6,                   # Max tree depth
    budget_usd=None,               # Optional cost cap
    db_path="./arborist.db",       # SQLite path
    memory_url=None,               # Optional Ultramemory URL
    on_node_complete=None,         # Callback: callable(node)
    on_insight=None,               # Callback: callable(insight)
    verbose=True,
)

# Blocking
results = search.run()
results.best            # Best node (config + score + results)
results.top_k(5)        # Top 5 nodes
results.insights        # List of cross-branch insights
results.report()        # Markdown report
results.tree_id         # For resume

# Resume
search = TreeSearch.resume(tree_id="abc123", db_path="./arborist.db", executor=my_fn, score=my_score)
results = search.run()
```

### Strategy Interface

```python
class Strategy:
    def select(self, candidates: list[Node], completed: list[Node]) -> list[Node]:
        """Select which nodes to expand next. Return ordered list."""
        ...
    
    def should_prune(self, node: Node, best_score: float, siblings: list[Node]) -> tuple[bool, str]:
        """Should this node be pruned? Return (prune, reason)."""
        ...
    
    def should_terminate(self, tree: Tree, nodes: list[Node]) -> tuple[bool, str]:
        """Should the entire search stop? Return (stop, reason)."""
        ...
```

### UCB Strategy
- Score = exploitation (node score) + exploration (sqrt(ln(total_visits) / node_visits))
- Exploration weight configurable (default: sqrt(2))
- Prune nodes scoring < 50% of best after min_samples (default: 3)
- Terminate on: max_experiments reached, budget exceeded, plateau detected (no improvement in last N experiments)

### Executor Interface

```python
class Executor:
    def run(self, config: dict, context: BranchContext) -> dict:
        """Run one experiment. Return structured results dict."""
        ...
```

`BranchContext` contains: parent config, parent results, parent score, depth, sibling scores, goal text.

### Python Executor
- Takes a callable: `fn(config: dict) -> dict`
- Runs in thread pool (concurrent.futures.ThreadPoolExecutor)
- Catches exceptions, records as failed nodes
- Optional timeout per experiment

### Shell Executor
- Takes a command template: `"python3 train.py --lr {lr} --features {features}"`
- Runs as subprocess
- Parses JSON from stdout as results
- Timeout support

### Mutator (generating child configs)

If no mutator is provided, use a default LLM mutator:
- Send parent config + results + sibling configs/results to LLM
- Ask it to suggest 2-4 child configs that explore promising directions
- Parse JSON response
- Fallback: random perturbation of numeric params, add/remove from list params

If mutator is provided, use it: `mutator(config, results, context) -> list[dict]`

### Evaluator

Simple case: `score=lambda r: r["f1"]` (just extract a number from results).

LLM evaluator (for non-numeric goals):
- Send goal + results to LLM
- Ask for 0-1 score + reasoning
- Uses litellm for provider flexibility

### Synthesis Engine

After search completes (or on-demand):
1. Collect all completed nodes
2. Group by branch lineage
3. Identify: best configs, worst configs, key decision points
4. Extract insights: "Feature X consistently improved scores", "Learning rate below 0.0001 never helped"
5. Generate markdown report with: summary, best result, top 5, insights, tree structure, cost breakdown

### Ultramemory Integration (optional)

When `memory_url` is set:
- On node complete: POST to /api/ingest with experiment facts
- Before spawning node: GET /api/search to check if similar config was tried in past searches
- On insight: POST to /api/ingest with insight as a memory
- Uses UPDATE relations: "Hypothesis A (tree 1) was extended by Hypothesis B (tree 2)"

### CLI

```bash
# Run from YAML config
arborist run --config search.yaml

# Check status of running/completed search  
arborist status [--tree-id ID] [--db ./arborist.db]

# Resume paused/interrupted search
arborist resume --tree-id ID [--db ./arborist.db]

# Generate report
arborist report [--tree-id ID] [--format markdown|json] [--output report.md]

# List all trees
arborist list [--db ./arborist.db]

# Show specific node details
arborist node NODE_ID [--db ./arborist.db]

# Prune a branch manually
arborist prune NODE_ID --reason "Manual prune" [--db ./arborist.db]
```

### YAML Config Format

```yaml
goal: "Maximize F1 for sleep staging"
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
    features: [accel_mean, accel_std]
  - lr: 0.001
    features: [accel_mean, hr_mean]

# Optional
memory:
  url: http://localhost:8642
  enabled: true

termination:
  target_score: 0.95
  plateau_window: 20    # Stop if no improvement in last 20 experiments
  budget_usd: 10.0
```

## Build Phases

### Phase 1: Core (THIS BUILD)
- [ ] SQLite store (create tables, CRUD for trees/nodes/insights)
- [ ] Tree manager (create tree, add nodes, update state, query)
- [ ] UCB strategy (select, prune, terminate)
- [ ] Best-first strategy
- [ ] Breadth-first strategy
- [ ] Python executor (thread pool, timeout, error handling)
- [ ] Shell executor (subprocess, JSON parsing, timeout)
- [ ] Default LLM mutator (generate child configs via litellm)
- [ ] Numeric evaluator
- [ ] TreeSearch orchestrator (ties everything together, run loop)
- [ ] Synthesis engine (basic report generation)
- [ ] CLI (run, status, resume, report, list)
- [ ] YAML config loader
- [ ] pyproject.toml (package config, dependencies)
- [ ] Tests (at least 10 passing tests covering core functionality)
- [ ] README.md (install, quickstart, API reference, examples)

### Phase 2: Intelligence (NEXT BUILD)
- [ ] LLM-guided strategy
- [ ] LLM evaluator
- [ ] Ultramemory integration
- [ ] Advanced synthesis (cross-branch insights)

### Phase 3: Ship
- [ ] PyPI publish
- [ ] GitHub Actions CI
- [ ] Examples directory

## Dependencies
- litellm (LLM abstraction, same as Ultramemory)
- click (CLI)
- pyyaml (config)
- No other required deps. Keep it minimal.

## Design Principles
1. **Local-first.** SQLite, no cloud, no accounts.
2. **LLM-agnostic.** litellm means any provider works.
3. **Composable.** Bring your own executor, evaluator, mutator, strategy.
4. **Resumable.** Everything persists to SQLite. Kill and restart anytime.
5. **Observable.** Verbose logging, callbacks, CLI status, future web UI.
6. **Minimal dependencies.** Core should work with just stdlib + sqlite3 + litellm.

## Testing

Write tests that use a simple mock experiment (e.g., maximize a quadratic function) to verify:
1. Tree creation and node management
2. UCB strategy selects correct nodes
3. Pruning works (low-scoring branches get cut)
4. Termination conditions trigger correctly
5. Resume works (create tree, add nodes, resume and continue)
6. Python executor runs callables correctly
7. Shell executor runs commands and parses JSON output
8. Report generation produces valid markdown
9. CLI commands work end-to-end
10. Concurrent execution (multiple branches in parallel)

Use pytest. Mock LLM calls where needed (don't require API keys for tests).
