# Contributing to Arborist

Thanks for your interest in contributing.

## Setup

```bash
git clone https://github.com/jared-goering/arborist.git
cd arborist
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Running tests

```bash
pytest
```

All 77 tests should pass in under 3 seconds.

## Making changes

1. Fork the repo and create a branch from `main`
2. Write tests for any new functionality
3. Make sure all tests pass
4. Keep commits focused and descriptive
5. Open a PR against `main`

## Code style

- Type hints on all public functions
- Docstrings on classes and public methods
- No external dependencies beyond litellm, click, pyyaml

## Reporting bugs

Open an issue with:
- What you expected
- What happened
- Minimal reproduction steps
- Python version and OS

## Adding a strategy

Subclass `Strategy` in `arborist/strategies/`:

```python
from arborist.strategies.base import Strategy

class MyStrategy(Strategy):
    def select(self, store, tree_id, context):
        # Return the node ID to expand next
        ...

    def should_prune(self, node, store, tree_id, context):
        # Return True to prune this branch
        ...
```

Register it in `arborist/strategies/__init__.py`.

## Adding an executor

Subclass `Executor` in `arborist/executors/`:

```python
from arborist.executors.base import Executor

class MyExecutor(Executor):
    def run(self, config, context):
        # Run the experiment, return results dict
        ...
```

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
