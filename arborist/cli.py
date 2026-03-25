"""CLI entry point for arborist."""

from __future__ import annotations

import json
import sys

import click
import yaml

from arborist.evaluators.numeric import NumericEvaluator
from arborist.executors.shell import ShellExecutor
from arborist.store import Store
from arborist.synthesis import generate_report
from arborist.tree import TreeSearch


def _load_yaml_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@click.group()
@click.version_option(version="0.1.0", prog_name="arborist")
def cli() -> None:
    """Arborist — Agentic tree search engine."""
    pass


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="YAML config file")
def run(config: str) -> None:
    """Run a tree search from a YAML config file."""
    cfg = _load_yaml_config(config)

    # Build executor
    executor_cfg = cfg.get("executor", {})
    executor_type = executor_cfg.get("type", "shell")
    if executor_type == "shell":
        executor = ShellExecutor(
            command=executor_cfg["command"],
            timeout=executor_cfg.get("timeout", 300),
        )
    else:
        click.echo(f"Error: executor type '{executor_type}' not supported in CLI mode.", err=True)
        click.echo("Use the Python API for custom executors.", err=True)
        sys.exit(1)

    # Build evaluator
    evaluator_cfg = cfg.get("evaluator", {})
    evaluator_type = evaluator_cfg.get("type", "numeric")
    if evaluator_type == "numeric":
        score = NumericEvaluator(field=evaluator_cfg.get("field", "score"))
    else:
        click.echo(f"Error: evaluator type '{evaluator_type}' not supported in CLI mode.", err=True)
        sys.exit(1)

    # Termination config
    term = cfg.get("termination", {})

    search = TreeSearch(
        goal=cfg["goal"],
        executor=executor,
        score=score,
        seed_configs=cfg.get("seed_configs", []),
        strategy=cfg.get("strategy", "ucb"),
        concurrency=cfg.get("concurrency", 5),
        max_experiments=cfg.get("max_experiments", 200),
        max_depth=cfg.get("max_depth", 6),
        budget_usd=term.get("budget_usd"),
        target_score=term.get("target_score"),
        plateau_window=term.get("plateau_window", 20),
        db_path=cfg.get("db_path", "./arborist.db"),
        memory_url=cfg.get("memory", {}).get("url") if cfg.get("memory", {}).get("enabled") else None,
        verbose=cfg.get("verbose", True),
    )

    click.echo(f"Starting search: {cfg['goal']}")
    results = search.run()

    best = results.best
    if best:
        click.echo(f"\nBest result: score={best['score']:.4f}")
        click.echo(f"Config: {json.dumps(best['config'], indent=2)}")
    else:
        click.echo("\nNo results found.")

    click.echo(f"\nTree ID: {results.tree_id}")
    click.echo(f"Run 'arborist report --tree-id {results.tree_id}' for full report")


@cli.command()
@click.option("--tree-id", "-t", help="Tree ID (latest if omitted)")
@click.option("--db", default="./arborist.db", help="Database path")
def status(tree_id: str | None, db: str) -> None:
    """Show status of a search tree."""
    store = Store(db)

    if tree_id:
        tree = store.get_tree(tree_id)
    else:
        trees = store.list_trees()
        tree = trees[0] if trees else None

    if not tree:
        click.echo("No trees found.")
        return

    nodes = store.get_tree_nodes(tree["id"])
    completed = [n for n in nodes if n["status"] == "completed"]
    pending = [n for n in nodes if n["status"] == "pending"]
    running = [n for n in nodes if n["status"] == "running"]
    failed = [n for n in nodes if n["status"] == "failed"]
    pruned = [n for n in nodes if n["pruned"]]
    best = store.get_best_node(tree["id"])

    click.echo(f"Tree: {tree['id']}")
    click.echo(f"Goal: {tree['goal']}")
    click.echo(f"Strategy: {tree['strategy']}")
    click.echo(f"Status: {tree['status']}")
    click.echo(f"Created: {tree['created_at']}")
    click.echo(f"")
    click.echo(f"Nodes: {len(nodes)} total")
    click.echo(f"  Completed: {len(completed)}")
    click.echo(f"  Pending:   {len(pending)}")
    click.echo(f"  Running:   {len(running)}")
    click.echo(f"  Failed:    {len(failed)}")
    click.echo(f"  Pruned:    {len(pruned)}")

    if best:
        click.echo(f"\nBest score: {best['score']:.4f} (node {best['id']})")

    total_cost = sum(n.get("cost_usd") or 0 for n in nodes)
    if total_cost > 0:
        click.echo(f"Total cost: ${total_cost:.4f}")


@cli.command()
@click.option("--tree-id", "-t", required=True, help="Tree ID to resume")
@click.option("--db", default="./arborist.db", help="Database path")
def resume(tree_id: str, db: str) -> None:
    """Resume a paused or interrupted search.

    Note: In CLI mode, resume only works with shell executors.
    Provide the original YAML config to restore the executor.
    """
    store = Store(db)
    tree = store.get_tree(tree_id)
    if not tree:
        click.echo(f"Tree {tree_id} not found.", err=True)
        sys.exit(1)

    if tree["status"] == "completed":
        click.echo(f"Tree {tree_id} is already completed.")
        return

    click.echo(f"To resume tree {tree_id}, use the Python API:")
    click.echo(f"")
    click.echo(f"  from arborist import TreeSearch")
    click.echo(f"  search = TreeSearch.resume('{tree_id}', db_path='{db}', executor=my_fn, score=my_score)")
    click.echo(f"  results = search.run()")
    click.echo(f"")
    click.echo(f"CLI resume requires the original executor which can't be serialized.")


@cli.command()
@click.option("--tree-id", "-t", help="Tree ID (latest if omitted)")
@click.option("--db", default="./arborist.db", help="Database path")
@click.option("--format", "-f", "fmt", type=click.Choice(["markdown", "json"]), default="markdown")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
def report(tree_id: str | None, db: str, fmt: str, output: str | None) -> None:
    """Generate a search report."""
    store = Store(db)

    if tree_id:
        tree = store.get_tree(tree_id)
    else:
        trees = store.list_trees()
        tree = trees[0] if trees else None

    if not tree:
        click.echo("No trees found.", err=True)
        sys.exit(1)

    if fmt == "markdown":
        content = generate_report(tree["id"], store)
    else:
        # JSON format
        from arborist.synthesis import SearchResults
        results = SearchResults(tree["id"], store)
        data = {
            "tree_id": tree["id"],
            "goal": tree["goal"],
            "status": tree["status"],
            "best": results.best,
            "top_5": results.top_k(5),
            "insights": results.insights,
        }
        content = json.dumps(data, indent=2, default=str)

    if output:
        with open(output, "w") as f:
            f.write(content)
        click.echo(f"Report written to {output}")
    else:
        click.echo(content)


@cli.command("list")
@click.option("--db", default="./arborist.db", help="Database path")
def list_trees(db: str) -> None:
    """List all search trees."""
    store = Store(db)
    trees = store.list_trees()

    if not trees:
        click.echo("No trees found.")
        return

    click.echo(f"{'ID':<14} {'Status':<12} {'Strategy':<14} {'Goal'}")
    click.echo("-" * 70)
    for tree in trees:
        goal = tree["goal"]
        if len(goal) > 40:
            goal = goal[:37] + "..."
        click.echo(f"{tree['id']:<14} {tree['status']:<12} {tree['strategy']:<14} {goal}")


@cli.command()
@click.argument("node_id")
@click.option("--db", default="./arborist.db", help="Database path")
def node(node_id: str, db: str) -> None:
    """Show details of a specific node."""
    store = Store(db)
    n = store.get_node(node_id)

    if not n:
        click.echo(f"Node {node_id} not found.", err=True)
        sys.exit(1)

    click.echo(f"Node: {n['id']}")
    click.echo(f"Tree: {n['tree_id']}")
    click.echo(f"Parent: {n['parent_id'] or '(root)'}")
    click.echo(f"Depth: {n['depth']}")
    click.echo(f"Status: {n['status']}")

    if n["hypothesis"]:
        click.echo(f"Hypothesis: {n['hypothesis']}")

    if n["score"] is not None:
        click.echo(f"Score: {n['score']:.4f}")

    config = n["config"]
    if isinstance(config, str):
        config = json.loads(config)
    click.echo(f"\nConfig:")
    click.echo(json.dumps(config, indent=2))

    if n["results"]:
        results = n["results"]
        if isinstance(results, str):
            results = json.loads(results)
        click.echo(f"\nResults:")
        click.echo(json.dumps(results, indent=2))

    if n["error"]:
        click.echo(f"\nError: {n['error']}")

    if n["pruned"]:
        click.echo(f"\nPruned: {n['prune_reason']}")

    if n["duration_ms"]:
        click.echo(f"\nDuration: {n['duration_ms']}ms")
    if n["cost_usd"]:
        click.echo(f"Cost: ${n['cost_usd']:.4f}")

    # Show children
    children = store.get_children(n["id"])
    if children:
        click.echo(f"\nChildren ({len(children)}):")
        for c in children:
            score_str = f" score={c['score']:.4f}" if c["score"] is not None else ""
            click.echo(f"  {c['id']} [{c['status']}]{score_str}")


@cli.command()
@click.argument("node_id")
@click.option("--reason", "-r", required=True, help="Reason for pruning")
@click.option("--db", default="./arborist.db", help="Database path")
def prune(node_id: str, reason: str, db: str) -> None:
    """Manually prune a node and its descendants."""
    store = Store(db)
    n = store.get_node(node_id)

    if not n:
        click.echo(f"Node {node_id} not found.", err=True)
        sys.exit(1)

    store.update_node(node_id, status="pruned", pruned=1, prune_reason=reason)
    click.echo(f"Pruned node {node_id}: {reason}")

    # Also prune descendants
    count = _prune_descendants(store, node_id, reason)
    if count > 0:
        click.echo(f"Also pruned {count} descendant(s)")


def _prune_descendants(store: Store, node_id: str, reason: str) -> int:
    children = store.get_children(node_id)
    count = 0
    for child in children:
        if not child["pruned"]:
            store.update_node(
                child["id"], status="pruned", pruned=1,
                prune_reason=f"Parent pruned: {reason}",
            )
            count += 1
        count += _prune_descendants(store, child["id"], reason)
    return count
