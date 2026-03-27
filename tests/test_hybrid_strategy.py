"""Tests for HybridStrategy — two-phase explore/exploit search."""

from __future__ import annotations

import json
import tempfile

import pytest

from arborist.strategies.hybrid import HybridStrategy
from arborist.tree import TreeSearch


# ── Helpers ───────────────────────────────────────────────────────────

def make_node(
    id: str,
    score: float | None = None,
    parent_id: str | None = None,
    depth: int = 0,
    status: str = "completed",
) -> dict:
    return {
        "id": id,
        "score": score,
        "parent_id": parent_id,
        "depth": depth,
        "status": status,
        "config": json.dumps({"x": 1}),
        "created_at": "2025-01-01T00:00:00",
    }


def make_candidate(
    id: str,
    parent_id: str | None = None,
    depth: int = 1,
) -> dict:
    return {
        "id": id,
        "parent_id": parent_id,
        "depth": depth,
        "config": json.dumps({"x": 1}),
        "created_at": "2025-01-01T00:00:00",
    }


# ── Test: Initialization ─────────────────────────────────────────────

class TestHybridInit:
    def test_defaults(self):
        s = HybridStrategy()
        assert s.explore_plateau == 8
        assert s.exploit_plateau == 5
        assert s.prune_threshold == 0.5
        assert s.plateau_window == 50
        assert s.phase == "explore"

    def test_custom_params(self):
        s = HybridStrategy(
            explore_plateau=12,
            exploit_plateau=3,
            exploration_weight=1.0,
            prune_threshold=0.3,
            plateau_window=30,
        )
        assert s.explore_plateau == 12
        assert s.exploit_plateau == 3
        assert s.prune_threshold == 0.3

    def test_starts_in_explore(self):
        s = HybridStrategy()
        assert s.phase == "explore"
        assert s._exploit_root_id is None
        assert len(s._exploited_roots) == 0


# ── Test: Explore phase uses UCB ──────────────────────────────────────

class TestExplorePhase:
    def test_explore_delegates_to_ucb(self):
        s = HybridStrategy()
        completed = [
            make_node("c1", score=5.0),
            make_node("c2", score=3.0),
        ]
        candidates = [
            make_candidate("p1", parent_id="c1"),
            make_candidate("p2", parent_id="c2"),
        ]
        selected = s.select(candidates, completed)
        assert len(selected) == 2
        # Should return both candidates (UCB ordering)
        ids = {n["id"] for n in selected}
        assert ids == {"p1", "p2"}

    def test_explore_empty_candidates(self):
        s = HybridStrategy()
        assert s.select([], []) == []


# ── Test: Phase transitions ──────────────────────────────────────────

class TestPhaseTransitions:
    def test_explore_to_exploit_on_plateau(self):
        s = HybridStrategy(explore_plateau=3)
        # Simulate completions with no improvement
        completed = [make_node("c0", score=10.0)]
        candidates = [make_candidate("p1", parent_id="c0")]

        # First call establishes baseline
        s.select(candidates, completed)
        assert s.phase == "explore"

        # Add non-improving nodes to trigger plateau
        for i in range(1, 5):
            completed.append(make_node(f"c{i}", score=5.0))

        s.select(candidates, completed)
        assert s.phase == "exploit"
        assert s._exploit_root_id == "c0"  # Best node
        assert "c0" in s._exploited_roots

    def test_exploit_to_explore_on_plateau(self):
        s = HybridStrategy(explore_plateau=3, exploit_plateau=2)

        # Build up to exploit phase
        completed = [make_node("c0", score=10.0)]
        candidates = [make_candidate("p1", parent_id="c0")]
        s.select(candidates, completed)

        for i in range(1, 5):
            completed.append(make_node(f"c{i}", score=5.0))
        s.select(candidates, completed)
        assert s.phase == "exploit"

        # Now trigger exploit plateau
        for i in range(5, 8):
            completed.append(make_node(f"c{i}", score=4.0, parent_id="c0"))
        s.select(candidates, completed)
        assert s.phase == "explore"

    def test_exploited_roots_tracked(self):
        s = HybridStrategy(explore_plateau=2, exploit_plateau=2)

        # First cycle
        completed = [make_node("best1", score=10.0)]
        candidates = [make_candidate("p1", parent_id="best1")]
        s.select(candidates, completed)

        # Plateau → exploit
        for i in range(3):
            completed.append(make_node(f"flat{i}", score=5.0))
        s.select(candidates, completed)
        assert s.phase == "exploit"
        assert "best1" in s._exploited_roots

        # Exploit plateau → back to explore
        for i in range(3):
            completed.append(make_node(f"expl{i}", score=4.0, parent_id="best1"))
        s.select(candidates, completed)
        assert s.phase == "explore"

        # Add a second best node
        completed.append(make_node("best2", score=9.0))

        # Plateau again → should exploit best2 (best1 already exploited)
        for i in range(3):
            completed.append(make_node(f"flat2_{i}", score=3.0))
        s.select(candidates, completed)
        assert s.phase == "exploit"
        assert s._exploit_root_id == "best2"
        assert "best1" in s._exploited_roots
        assert "best2" in s._exploited_roots

    def test_stay_explore_when_all_exploited(self):
        s = HybridStrategy(explore_plateau=2)

        completed = [make_node("only", score=10.0)]
        candidates = [make_candidate("p1", parent_id="only")]
        s.select(candidates, completed)

        # Mark ALL nodes as exploited manually (including ones we'll add)
        s._exploited_roots.add("only")
        for i in range(3):
            s._exploited_roots.add(f"f{i}")

        # Trigger plateau — should stay in explore since all nodes are exploited
        for i in range(3):
            completed.append(make_node(f"f{i}", score=5.0))
        s.select(candidates, completed)
        assert s.phase == "explore"


# ── Test: Exploit phase filtering ─────────────────────────────────────

class TestExploitFiltering:
    def test_exploit_filters_to_subtree(self):
        s = HybridStrategy(explore_plateau=2)

        # Setup: force into exploit phase
        completed = [
            make_node("root", score=10.0),
            make_node("other", score=8.0),
        ]
        candidates = [
            make_candidate("child_root", parent_id="root"),
            make_candidate("child_other", parent_id="other"),
        ]
        s.select(candidates, completed)

        # Plateau → exploit on "root"
        for i in range(3):
            completed.append(make_node(f"f{i}", score=5.0))
        selected = s.select(candidates, completed)

        assert s.phase == "exploit"
        # Should only include child_root (descendant of exploit root)
        assert len(selected) == 1
        assert selected[0]["id"] == "child_root"

    def test_exploit_includes_grandchildren(self):
        s = HybridStrategy(explore_plateau=2)

        completed = [
            make_node("root", score=10.0),
            make_node("child", score=9.0, parent_id="root"),
        ]
        candidates = [
            make_candidate("grandchild", parent_id="child", depth=2),
            make_candidate("unrelated", parent_id="other_root"),
        ]
        s.select(candidates, completed)

        # Trigger exploit
        for i in range(3):
            completed.append(make_node(f"f{i}", score=5.0))
        selected = s.select(candidates, completed)

        assert s.phase == "exploit"
        assert len(selected) == 1
        assert selected[0]["id"] == "grandchild"

    def test_exploit_falls_back_to_ucb_when_no_subtree_candidates(self):
        s = HybridStrategy(explore_plateau=2)

        completed = [make_node("root", score=10.0)]
        # No candidates in root's subtree
        candidates = [make_candidate("unrelated", parent_id="other")]
        s.select(candidates, completed)

        # Trigger exploit
        for i in range(3):
            completed.append(make_node(f"f{i}", score=5.0))
        selected = s.select(candidates, completed)

        assert s.phase == "exploit"
        # Falls back to UCB since no exploit candidates
        assert len(selected) == 1
        assert selected[0]["id"] == "unrelated"

    def test_exploit_greedy_ordering(self):
        s = HybridStrategy(explore_plateau=2)

        completed = [
            make_node("root", score=10.0),
            make_node("child_a", score=7.0, parent_id="root"),
            make_node("child_b", score=9.0, parent_id="root"),
        ]
        candidates = [
            make_candidate("gc_a", parent_id="child_a", depth=2),
            make_candidate("gc_b", parent_id="child_b", depth=2),
        ]
        s.select(candidates, completed)

        # Trigger exploit
        for i in range(3):
            completed.append(make_node(f"f{i}", score=5.0))
        selected = s.select(candidates, completed)

        assert s.phase == "exploit"
        assert len(selected) == 2
        # gc_b's parent (child_b) has higher score → should be first
        assert selected[0]["id"] == "gc_b"
        assert selected[1]["id"] == "gc_a"


# ── Test: Pruning (inherited from Strategy base) ──────────────────────

class TestHybridPruning:
    def test_prune_below_threshold(self):
        s = HybridStrategy(prune_threshold=0.5)
        node = {"id": "n1", "score": 2.0}
        should_prune, reason = s.should_prune(node, best_score=10.0, siblings=[])
        assert should_prune
        assert "50%" in reason

    def test_no_prune_above_threshold(self):
        s = HybridStrategy(prune_threshold=0.5)
        node = {"id": "n1", "score": 8.0}
        should_prune, _ = s.should_prune(node, best_score=10.0, siblings=[])
        assert not should_prune

    def test_no_prune_none_score(self):
        s = HybridStrategy()
        node = {"id": "n1", "score": None}
        should_prune, _ = s.should_prune(node, best_score=10.0, siblings=[])
        assert not should_prune


# ── Test: Termination (inherited from Strategy base) ──────────────────

class TestHybridTermination:
    def test_terminates_on_max_experiments(self):
        s = HybridStrategy()
        tree = {"id": "t1", "status": "running"}
        completed = [{"id": f"n{i}", "score": float(i)} for i in range(10)]
        config = {"max_experiments": 10}
        should_stop, reason = s.should_terminate(tree, completed, config)
        assert should_stop
        assert "max experiments" in reason.lower()


# ── Test: Improvement resets plateau counter ──────────────────────────

class TestImprovementResets:
    def test_improvement_resets_counter(self):
        s = HybridStrategy(explore_plateau=5)

        completed = [make_node("c0", score=5.0)]
        candidates = [make_candidate("p1")]
        s.select(candidates, completed)

        # 3 non-improving
        for i in range(1, 4):
            completed.append(make_node(f"c{i}", score=3.0))
        s.select(candidates, completed)
        assert s.phase == "explore"

        # Improvement resets counter
        completed.append(make_node("c4", score=15.0))
        s.select(candidates, completed)
        assert s.phase == "explore"
        assert s._no_improve_count == 0

        # Add 4 non-improving — not enough to trigger plateau (need 5)
        for i in range(5, 9):
            completed.append(make_node(f"c{i}", score=3.0))
        s.select(candidates, completed)
        assert s.phase == "explore"  # 4 < threshold of 5

        # One more tips it over
        completed.append(make_node("c9", score=3.0))
        s.select(candidates, completed)
        assert s.phase == "exploit"


# ── Test: Strategy registered ─────────────────────────────────────────

class TestRegistration:
    def test_hybrid_in_strategies_dict(self):
        from arborist.strategies import STRATEGIES
        assert "hybrid" in STRATEGIES
        assert STRATEGIES["hybrid"] is HybridStrategy


# ── Test: End-to-end with TreeSearch ──────────────────────────────────

def quadratic_experiment(config: dict) -> dict:
    x = config.get("x", 0)
    return {"score": -(x - 3) ** 2 + 10, "x": x}


def simple_mutator(config, results, context):
    x = config.get("x", 0)
    return [{"x": x + 0.5}, {"x": x - 0.5}]


class TestHybridEndToEnd:
    def test_full_search(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        search = TreeSearch(
            goal="Maximize quadratic",
            executor=quadratic_experiment,
            score=lambda r: r["score"],
            seed_configs=[{"x": 0}, {"x": 1}, {"x": 5}],
            strategy=HybridStrategy(
                explore_plateau=4,
                exploit_plateau=3,
            ),
            mutator=simple_mutator,
            concurrency=2,
            max_experiments=15,
            max_depth=4,
            db_path=db_path,
            verbose=False,
        )
        results = search.run()

        assert results.best is not None
        assert results.best["score"] is not None
        assert results.best["score"] > 0

    def test_strategy_string_lookup(self, tmp_path):
        """Verify 'hybrid' string resolves to HybridStrategy."""
        db_path = str(tmp_path / "test.db")
        search = TreeSearch(
            goal="Test hybrid string",
            executor=quadratic_experiment,
            score=lambda r: r["score"],
            seed_configs=[{"x": 2}],
            strategy="hybrid",
            mutator=simple_mutator,
            concurrency=1,
            max_experiments=5,
            max_depth=2,
            db_path=db_path,
            verbose=False,
        )
        results = search.run()
        assert results.best is not None
