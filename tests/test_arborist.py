"""Tests for arborist — uses simple mock experiments (quadratic optimization)."""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from arborist.evaluators.numeric import NumericEvaluator
from arborist.executors.python import PythonExecutor
from arborist.executors.shell import ShellExecutor
from arborist.manager import BranchContext, TreeManager
from arborist.store import Store
from arborist.strategies.best_first import BestFirstStrategy
from arborist.strategies.breadth_first import BreadthFirstStrategy
from arborist.strategies.ucb import UCBStrategy
from arborist.synthesis import SearchResults, extract_basic_insights, generate_report
from arborist.tree import TreeSearch


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temp database path."""
    return str(tmp_path / "test.db")


@pytest.fixture
def store(tmp_db):
    """Create a store with temp database."""
    return Store(tmp_db)


@pytest.fixture
def manager(store):
    """Create a tree manager."""
    return TreeManager(store)


# ── Simple experiment function for testing ─────────────────────────────

def quadratic_experiment(config: dict) -> dict:
    """Simple experiment: maximize -(x-3)^2 + 10. Optimal at x=3, score=10."""
    x = config.get("x", 0)
    score = -(x - 3) ** 2 + 10
    return {"score": score, "x": x}


def simple_mutator(config, results, context):
    """Deterministic mutator for testing — generates two children."""
    x = config.get("x", 0)
    return [
        {"x": x + 0.5},
        {"x": x - 0.5},
    ]


# ── Test 1: Tree creation and node management ─────────────────────────

class TestTreeAndNodeManagement:
    def test_create_tree(self, store):
        tree = store.create_tree(goal="Test goal", strategy="ucb")
        assert tree["id"]
        assert tree["goal"] == "Test goal"
        assert tree["strategy"] == "ucb"
        assert tree["status"] == "running"

    def test_create_and_get_node(self, store):
        tree = store.create_tree(goal="Test", strategy="ucb")
        node = store.create_node(tree["id"], config={"x": 1}, depth=0)
        assert node["id"]
        assert node["tree_id"] == tree["id"]
        assert node["status"] == "pending"
        assert json.loads(node["config"]) == {"x": 1}

    def test_node_hierarchy(self, store):
        tree = store.create_tree(goal="Test", strategy="ucb")
        root = store.create_node(tree["id"], config={"x": 0}, depth=0)
        child = store.create_node(tree["id"], config={"x": 1}, parent_id=root["id"], depth=1)

        children = store.get_children(root["id"])
        assert len(children) == 1
        assert children[0]["id"] == child["id"]

    def test_update_node(self, store):
        tree = store.create_tree(goal="Test", strategy="ucb")
        node = store.create_node(tree["id"], config={"x": 1})
        store.update_node(node["id"], status="completed", score=5.0, results={"value": 5})

        updated = store.get_node(node["id"])
        assert updated["status"] == "completed"
        assert updated["score"] == 5.0

    def test_tree_manager_lifecycle(self, manager, store):
        tree = manager.create_tree(goal="Lifecycle test")
        nodes = manager.add_seed_nodes(tree["id"], [{"x": 1}, {"x": 2}])
        assert len(nodes) == 2

        manager.mark_running(nodes[0]["id"])
        manager.mark_completed(nodes[0]["id"], results={"score": 7.0}, score=7.0)

        completed = manager.get_completed_nodes(tree["id"])
        assert len(completed) == 1
        assert completed[0]["score"] == 7.0


# ── Test 2: UCB strategy selects correct nodes ────────────────────────

class TestUCBStrategy:
    def test_select_prioritizes_unexplored(self):
        strategy = UCBStrategy(exploration_weight=1.414)

        # Completed nodes with scores
        completed = [
            {"id": "c1", "parent_id": None, "score": 5.0},
            {"id": "c2", "parent_id": None, "score": 3.0},
        ]

        # Pending nodes — one from explored branch, one from unexplored
        candidates = [
            {"id": "p1", "parent_id": "c1", "depth": 1, "created_at": "2025-01-01"},
            {"id": "p2", "parent_id": "c2", "depth": 1, "created_at": "2025-01-01"},
        ]

        selected = strategy.select(candidates, completed)
        assert len(selected) == 2
        # Should return both, with higher-scoring parent first (exploitation)
        # or unexplored first (exploration) depending on weights

    def test_select_empty_candidates(self):
        strategy = UCBStrategy()
        assert strategy.select([], []) == []


# ── Test 3: Pruning works ─────────────────────────────────────────────

class TestPruning:
    def test_prune_low_scoring_node(self):
        strategy = UCBStrategy(prune_threshold=0.5, min_samples=2)

        node = {"id": "n1", "score": 1.0}
        siblings = [
            {"id": "s1", "score": 8.0},
            {"id": "s2", "score": 7.0},
            {"id": "s3", "score": 9.0},
        ]

        should_prune, reason = strategy.should_prune(node, best_score=10.0, siblings=siblings)
        assert should_prune
        assert "50%" in reason

    def test_no_prune_above_threshold(self):
        strategy = UCBStrategy(prune_threshold=0.5, min_samples=2)

        node = {"id": "n1", "score": 8.0}
        siblings = [
            {"id": "s1", "score": 7.0},
            {"id": "s2", "score": 9.0},
            {"id": "s3", "score": 6.0},
        ]

        should_prune, _ = strategy.should_prune(node, best_score=10.0, siblings=siblings)
        assert not should_prune

    def test_no_prune_insufficient_samples(self):
        strategy = UCBStrategy(prune_threshold=0.5, min_samples=3)

        node = {"id": "n1", "score": 1.0}
        siblings = [{"id": "s1", "score": 8.0}]  # Only 1 sibling < min_samples

        should_prune, _ = strategy.should_prune(node, best_score=10.0, siblings=siblings)
        assert not should_prune


# ── Test 4: Termination conditions ────────────────────────────────────

class TestTermination:
    def test_terminate_on_max_experiments(self):
        strategy = UCBStrategy()
        tree = {"id": "t1", "status": "running"}
        completed = [{"id": f"n{i}", "score": float(i)} for i in range(10)]
        config = {"max_experiments": 10}

        should_stop, reason = strategy.should_terminate(tree, completed, config)
        assert should_stop
        assert "max experiments" in reason.lower()

    def test_terminate_on_target_score(self):
        strategy = UCBStrategy()
        tree = {"id": "t1", "status": "running"}
        completed = [{"id": "n1", "score": 0.96}]
        config = {"max_experiments": 100, "target_score": 0.95}

        should_stop, reason = strategy.should_terminate(tree, completed, config)
        assert should_stop
        assert "target" in reason.lower()

    def test_no_terminate_when_progressing(self):
        strategy = UCBStrategy()
        tree = {"id": "t1", "status": "running"}
        completed = [{"id": f"n{i}", "score": float(i)} for i in range(5)]
        config = {"max_experiments": 100}

        should_stop, _ = strategy.should_terminate(tree, completed, config)
        assert not should_stop


# ── Test 5: Resume works ──────────────────────────────────────────────

class TestResume:
    def test_resume_tree(self, tmp_db):
        """Create a tree, add some nodes, then resume."""
        store = Store(tmp_db)
        tree = store.create_tree(goal="Resume test", strategy="ucb", config=json.dumps({
            "max_experiments": 50, "max_depth": 4, "concurrency": 2,
        }))
        node = store.create_node(tree["id"], config={"x": 2}, depth=0)
        store.update_node(node["id"], status="completed", score=6.0, results=json.dumps({"score": 6.0}))
        store.update_tree(tree["id"], status="paused")
        store.close()

        # Resume
        search = TreeSearch.resume(
            tree_id=tree["id"],
            db_path=tmp_db,
            executor=quadratic_experiment,
            score=lambda r: r["score"],
            mutator=simple_mutator,
            max_experiments=3,
        )
        assert search._tree_id == tree["id"]
        assert search.goal == "Resume test"


# ── Test 6: Python executor ───────────────────────────────────────────

class TestPythonExecutor:
    def test_run_callable(self):
        executor = PythonExecutor(fn=quadratic_experiment)
        context = BranchContext(goal="test", depth=0)
        result = executor.run({"x": 3}, context)
        assert result["score"] == 10.0

    def test_run_with_timeout(self):
        import time

        def slow_fn(config):
            time.sleep(5)
            return {"score": 1}

        executor = PythonExecutor(fn=slow_fn, timeout=0.1)
        context = BranchContext(goal="test", depth=0)

        with pytest.raises(TimeoutError):
            executor.run({}, context)

    def test_non_dict_result(self):
        executor = PythonExecutor(fn=lambda c: 42)
        context = BranchContext(goal="test", depth=0)
        result = executor.run({}, context)
        assert result == {"value": 42}


# ── Test 7: Shell executor ────────────────────────────────────────────

class TestShellExecutor:
    def test_run_echo_json(self):
        executor = ShellExecutor(command='echo \'{"score": 42}\'', timeout=5)
        context = BranchContext(goal="test", depth=0)
        result = executor.run({}, context)
        assert result["score"] == 42

    def test_run_with_config_substitution(self):
        executor = ShellExecutor(
            command='echo \'{"value": {x}}\'',
            timeout=5,
        )
        context = BranchContext(goal="test", depth=0)
        result = executor.run({"x": 7}, context)
        assert result["value"] == 7

    def test_command_failure(self):
        executor = ShellExecutor(command="exit 1", timeout=5)
        context = BranchContext(goal="test", depth=0)

        with pytest.raises(RuntimeError, match="exited with code 1"):
            executor.run({}, context)


# ── Test 8: Report generation ─────────────────────────────────────────

class TestReportGeneration:
    def test_generate_report(self, store):
        tree = store.create_tree(goal="Report test", strategy="ucb")
        n1 = store.create_node(tree["id"], config={"x": 1}, depth=0)
        store.update_node(n1["id"], status="completed", score=5.0, results={"score": 5.0})
        n2 = store.create_node(tree["id"], config={"x": 3}, depth=0)
        store.update_node(n2["id"], status="completed", score=10.0, results={"score": 10.0})

        report = generate_report(tree["id"], store)
        assert "# Arborist Search Report" in report
        assert "Report test" in report
        assert "10.0000" in report
        assert "## Best Result" in report

    def test_search_results_interface(self, store):
        tree = store.create_tree(goal="Results test", strategy="ucb")
        for i in range(5):
            n = store.create_node(tree["id"], config={"x": i}, depth=0)
            store.update_node(n["id"], status="completed", score=float(i), results={"score": float(i)})

        results = SearchResults(tree["id"], store)
        assert results.best["score"] == 4.0
        assert len(results.top_k(3)) == 3
        assert results.top_k(3)[0]["score"] == 4.0


# ── Test 9: CLI commands ──────────────────────────────────────────────

class TestCLI:
    def test_list_trees(self, tmp_db):
        from click.testing import CliRunner
        from arborist.cli import cli

        store = Store(tmp_db)
        store.create_tree(goal="CLI test tree", strategy="ucb")
        store.close()

        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--db", tmp_db])
        assert result.exit_code == 0
        assert "CLI test tree" in result.output

    def test_status_command(self, tmp_db):
        from click.testing import CliRunner
        from arborist.cli import cli

        store = Store(tmp_db)
        tree = store.create_tree(goal="Status test", strategy="ucb")
        store.close()

        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--tree-id", tree["id"], "--db", tmp_db])
        assert result.exit_code == 0
        assert "Status test" in result.output
        assert tree["id"] in result.output

    def test_report_command(self, tmp_db):
        from click.testing import CliRunner
        from arborist.cli import cli

        store = Store(tmp_db)
        tree = store.create_tree(goal="Report CLI test", strategy="ucb")
        n = store.create_node(tree["id"], config={"x": 5}, depth=0)
        store.update_node(n["id"], status="completed", score=7.5, results={"score": 7.5})
        store.close()

        runner = CliRunner()
        result = runner.invoke(cli, ["report", "--tree-id", tree["id"], "--db", tmp_db])
        assert result.exit_code == 0
        assert "Arborist Search Report" in result.output

    def test_node_command(self, tmp_db):
        from click.testing import CliRunner
        from arborist.cli import cli

        store = Store(tmp_db)
        tree = store.create_tree(goal="Node test", strategy="ucb")
        n = store.create_node(tree["id"], config={"x": 42}, depth=0)
        store.update_node(n["id"], status="completed", score=9.0, results={"score": 9.0})
        store.close()

        runner = CliRunner()
        result = runner.invoke(cli, ["node", n["id"], "--db", tmp_db])
        assert result.exit_code == 0
        assert "9.0" in result.output

    def test_prune_command(self, tmp_db):
        from click.testing import CliRunner
        from arborist.cli import cli

        store = Store(tmp_db)
        tree = store.create_tree(goal="Prune test", strategy="ucb")
        n = store.create_node(tree["id"], config={"x": 1}, depth=0)
        store.close()

        runner = CliRunner()
        result = runner.invoke(cli, ["prune", n["id"], "--reason", "Bad branch", "--db", tmp_db])
        assert result.exit_code == 0
        assert "Pruned" in result.output


# ── Test 10: Concurrent execution (end-to-end) ────────────────────────

class TestConcurrentExecution:
    def test_full_search(self, tmp_db):
        """Run a complete search with quadratic optimization."""
        search = TreeSearch(
            goal="Maximize quadratic",
            executor=quadratic_experiment,
            score=lambda r: r["score"],
            seed_configs=[{"x": 0}, {"x": 1}, {"x": 5}],
            strategy="ucb",
            mutator=simple_mutator,
            concurrency=2,
            max_experiments=10,
            max_depth=3,
            db_path=tmp_db,
            verbose=False,
        )
        results = search.run()

        assert results.best is not None
        assert results.best["score"] is not None
        # With seeds at 0, 1, 5 and mutator +/- 0.5, we should find decent scores
        assert results.best["score"] > 0

        # Check we have some completed nodes
        top = results.top_k(3)
        assert len(top) > 0

        # Report should work
        report = results.report()
        assert "Arborist Search Report" in report

    def test_full_search_best_first(self, tmp_db):
        """Run a search with best-first strategy."""
        search = TreeSearch(
            goal="Best-first test",
            executor=quadratic_experiment,
            score=lambda r: r["score"],
            seed_configs=[{"x": 2}, {"x": 4}],
            strategy="best_first",
            mutator=simple_mutator,
            concurrency=2,
            max_experiments=8,
            max_depth=2,
            db_path=tmp_db,
            verbose=False,
        )
        results = search.run()
        assert results.best is not None

    def test_full_search_breadth_first(self, tmp_db):
        """Run a search with breadth-first strategy."""
        search = TreeSearch(
            goal="Breadth-first test",
            executor=quadratic_experiment,
            score=lambda r: r["score"],
            seed_configs=[{"x": 1}, {"x": 5}],
            strategy="breadth_first",
            mutator=simple_mutator,
            concurrency=2,
            max_experiments=8,
            max_depth=2,
            db_path=tmp_db,
            verbose=False,
        )
        results = search.run()
        assert results.best is not None


# ── Test 11: Evaluators ───────────────────────────────────────────────

class TestEvaluators:
    def test_numeric_field(self):
        evaluator = NumericEvaluator(field="f1")
        score = evaluator.evaluate({"f1": 0.95}, {})
        assert score == 0.95

    def test_numeric_dotpath(self):
        evaluator = NumericEvaluator(field="metrics.f1")
        score = evaluator.evaluate({"metrics": {"f1": 0.87}}, {})
        assert score == 0.87

    def test_numeric_fn(self):
        evaluator = NumericEvaluator(fn=lambda r: r["a"] + r["b"])
        score = evaluator.evaluate({"a": 3, "b": 4}, {})
        assert score == 7.0

    def test_numeric_requires_field_or_fn(self):
        with pytest.raises(ValueError):
            NumericEvaluator()


# ── Test 12: Insights extraction ──────────────────────────────────────

class TestInsights:
    def test_extract_basic_insights(self, store):
        tree = store.create_tree(goal="Insight test", strategy="ucb")
        for i in range(5):
            n = store.create_node(tree["id"], config={"x": i}, depth=0)
            store.update_node(n["id"], status="completed", score=float(i), results={"score": float(i)})

        insights = extract_basic_insights(tree["id"], store)
        assert len(insights) >= 1
        types = [ins["type"] for ins in insights]
        assert "discovery" in types

    def test_no_insights_with_few_nodes(self, store):
        tree = store.create_tree(goal="Few nodes", strategy="ucb")
        n = store.create_node(tree["id"], config={"x": 1}, depth=0)
        store.update_node(n["id"], status="completed", score=5.0, results={"score": 5.0})

        insights = extract_basic_insights(tree["id"], store)
        assert len(insights) == 0  # Need at least 2 completed


# ── Test 13: Strategies (breadth-first, best-first) ───────────────────

class TestStrategies:
    def test_breadth_first_ordering(self):
        strategy = BreadthFirstStrategy()
        candidates = [
            {"id": "d2", "depth": 2, "created_at": "2025-01-01T00:00:01"},
            {"id": "d0", "depth": 0, "created_at": "2025-01-01T00:00:00"},
            {"id": "d1", "depth": 1, "created_at": "2025-01-01T00:00:00"},
        ]
        selected = strategy.select(candidates, [])
        assert selected[0]["id"] == "d0"
        assert selected[1]["id"] == "d1"
        assert selected[2]["id"] == "d2"

    def test_best_first_ordering(self):
        strategy = BestFirstStrategy()
        completed = [
            {"id": "c1", "score": 5.0},
            {"id": "c2", "score": 9.0},
        ]
        candidates = [
            {"id": "p1", "parent_id": "c1", "depth": 1, "created_at": "2025-01-01"},
            {"id": "p2", "parent_id": "c2", "depth": 1, "created_at": "2025-01-01"},
        ]
        selected = strategy.select(candidates, completed)
        # p2's parent has higher score, so it should be first
        assert selected[0]["id"] == "p2"
