"""TreeSearch — main orchestrator that ties everything together."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import Any, Callable

from arborist.evaluators import NumericEvaluator
from arborist.evaluators.base import Evaluator
from arborist.executors.base import Executor
from arborist.executors.python import PythonExecutor
from arborist.manager import BranchContext, TreeManager
from arborist.mutators import RandomMutator
from arborist.store import Store
from arborist.strategies import STRATEGIES, Strategy
from arborist.synthesis import SearchResults, extract_basic_insights

logger = logging.getLogger(__name__)


class TreeSearch:
    """Main entry point for running agentic tree searches.

    Orchestrates strategies, executors, evaluators, and mutators to
    explore a search space and find optimal configurations.
    """

    def __init__(
        self,
        goal: str,
        executor: Callable[[dict[str, Any]], dict[str, Any]] | Executor,
        score: Callable[[dict[str, Any]], float] | Evaluator | str | None = None,
        seed_configs: list[dict[str, Any]] | None = None,
        strategy: str | Strategy = "ucb",
        mutator: Callable[..., list[dict[str, Any]]] | None = None,
        concurrency: int = 5,
        max_experiments: int = 200,
        max_depth: int = 6,
        budget_usd: float | None = None,
        target_score: float | None = None,
        plateau_window: int = 20,
        db_path: str = "./arborist.db",
        on_node_complete: Callable[[dict[str, Any]], None] | None = None,
        on_insight: Callable[[dict[str, Any]], None] | None = None,
        verbose: bool = True,
        # Internal: for resume
        _tree_id: str | None = None,
    ) -> None:
        self.goal = goal
        self.seed_configs = seed_configs or []
        self.concurrency = concurrency
        self.max_depth = max_depth
        self.verbose = verbose
        self.on_node_complete = on_node_complete
        self.on_insight = on_insight

        # Set up logging
        if verbose:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            )

        # Store and manager
        self.store = Store(db_path)
        self.manager = TreeManager(self.store)

        # Executor
        if callable(executor) and not isinstance(executor, Executor):
            self._executor = PythonExecutor(fn=executor)
        else:
            self._executor = executor  # type: ignore[assignment]

        # Evaluator
        if score is None:
            self._evaluator = NumericEvaluator(field="score")
        elif isinstance(score, str):
            self._evaluator = NumericEvaluator(field=score)
        elif callable(score) and not isinstance(score, Evaluator):
            self._evaluator = NumericEvaluator(fn=score)
        else:
            self._evaluator = score  # type: ignore[assignment]

        # Strategy
        if isinstance(strategy, str):
            strategy_cls = STRATEGIES.get(strategy)
            if not strategy_cls:
                raise ValueError(
                    f"Unknown strategy '{strategy}'. "
                    f"Available: {list(STRATEGIES.keys())}"
                )
            # Pass goal to strategies that accept it (e.g., LLMGuidedStrategy)
            import inspect
            sig = inspect.signature(strategy_cls.__init__)
            if "goal" in sig.parameters:
                self._strategy = strategy_cls(goal=goal)
            else:
                self._strategy = strategy_cls()
        else:
            self._strategy = strategy

        # Mutator
        self._mutator = mutator or RandomMutator()

        # Wire store into mutator if it supports it (e.g., LLMMutator)
        if hasattr(self._mutator, "set_store"):
            self._mutator.set_store(self.store)

        # Config for termination checks
        self._config = {
            "max_experiments": max_experiments,
            "max_depth": max_depth,
            "budget_usd": budget_usd,
            "target_score": target_score,
            "plateau_window": plateau_window,
        }

        # Tree ID for resume
        self._tree_id = _tree_id

    @classmethod
    def resume(
        cls,
        tree_id: str,
        db_path: str = "./arborist.db",
        executor: Callable[[dict[str, Any]], dict[str, Any]] | Executor | None = None,
        score: Callable[[dict[str, Any]], float] | Evaluator | str | None = None,
        **kwargs: Any,
    ) -> TreeSearch:
        """Resume a paused or interrupted search."""
        store = Store(db_path)
        tree = store.get_tree(tree_id)
        if not tree:
            raise ValueError(f"Tree {tree_id} not found in {db_path}")

        raw_config = tree["config"]
        if raw_config:
            tree_config = json.loads(raw_config)
            # Handle double-encoded JSON (string inside string)
            if isinstance(tree_config, str):
                tree_config = json.loads(tree_config)
        else:
            tree_config = {}

        if executor is None:
            raise ValueError("Must provide executor to resume a search")

        # kwargs override stored config
        merged = {
            "max_experiments": tree_config.get("max_experiments", 200),
            "max_depth": tree_config.get("max_depth", 6),
            "concurrency": tree_config.get("concurrency", 5),
        }
        merged.update(kwargs)

        return cls(
            goal=tree["goal"],
            executor=executor,
            score=score,
            strategy=tree["strategy"],
            db_path=db_path,
            _tree_id=tree_id,
            **merged,
        )

    def run(self) -> SearchResults:
        """Run the search to completion. Blocking."""
        # Create or load tree
        if self._tree_id:
            tree = self.store.get_tree(self._tree_id)
            if not tree:
                raise ValueError(f"Tree {self._tree_id} not found")
            tree_id = self._tree_id
            self.store.update_tree(tree_id, status="running")
            logger.info("Resuming tree %s: %s", tree_id, tree["goal"])

            # Wire tree_id into mutator if it supports it
            if hasattr(self._mutator, "set_tree_id"):
                self._mutator.set_tree_id(tree_id)
        else:
            tree = self.manager.create_tree(
                goal=self.goal,
                strategy=(
                    self._strategy.__class__.__name__
                    if isinstance(self._strategy, Strategy)
                    else str(self._strategy)
                ),
                config=self._config,
            )
            tree_id = tree["id"]
            logger.info("Created tree %s: %s", tree_id, self.goal)

            # Wire tree_id into mutator if it supports it
            if hasattr(self._mutator, "set_tree_id"):
                self._mutator.set_tree_id(tree_id)

            # Add seed nodes
            if self.seed_configs:
                self.manager.add_seed_nodes(tree_id, self.seed_configs)
                logger.info("Added %d seed nodes", len(self.seed_configs))

        # Main loop
        try:
            self._run_loop(tree_id)
        except KeyboardInterrupt:
            logger.info("Search interrupted, pausing tree %s", tree_id)
            self.manager.pause_tree(tree_id)
        except Exception as e:
            logger.error("Search failed: %s", e)
            self.manager.fail_tree(tree_id)
            raise

        # Extract insights
        try:
            insights = extract_basic_insights(tree_id, self.store)
            for insight in insights:
                if self.on_insight:
                    self.on_insight(insight)

        except Exception as e:
            logger.warning("Failed to extract insights: %s", e)

        # Mark complete
        tree = self.store.get_tree(tree_id)
        if tree and tree["status"] == "running":
            self.manager.complete_tree(tree_id)

        return SearchResults(tree_id, self.store)

    def _run_loop(self, tree_id: str) -> None:
        """Core search loop: select, execute, evaluate, expand."""
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            while True:
                tree = self.store.get_tree(tree_id)
                if not tree or tree["status"] != "running":
                    break

                completed = self.manager.get_completed_nodes(tree_id)

                # Check termination
                should_stop, reason = self._strategy.should_terminate(
                    tree, completed, self._config
                )
                if should_stop:
                    logger.info("Terminating: %s", reason)
                    break

                # Get pending nodes
                pending = self.manager.get_pending_nodes(tree_id)
                if not pending:
                    # No pending and nothing running — check if we should generate more
                    running = self.store.get_tree_nodes(tree_id, status="running")
                    if not running:
                        # Try to expand from completed nodes
                        if not self._expand_from_completed(tree_id, completed):
                            logger.info("No more nodes to explore")
                            break
                        pending = self.manager.get_pending_nodes(tree_id)
                        if not pending:
                            break

                # Strategy selects which to run
                selected = self._strategy.select(pending, completed)
                batch = selected[: self.concurrency]

                if not batch:
                    break

                # Submit batch
                futures: dict[Future, dict[str, Any]] = {}
                for node in batch:
                    self.manager.mark_running(node["id"])
                    config = json.loads(node["config"]) if isinstance(node["config"], str) else node["config"]
                    context = self.manager.get_branch_context(node, self.goal)
                    future = pool.submit(self._execute_node, node, config, context)
                    futures[future] = node

                # Collect results
                for future in as_completed(futures):
                    node = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        logger.error("Node %s execution error: %s", node["id"], e)

    def _execute_node(
        self,
        node: dict[str, Any],
        config: dict[str, Any],
        context: BranchContext,
    ) -> None:
        """Execute a single node: run, evaluate, prune check, expand."""
        node_id = node["id"]
        tree_id = node["tree_id"]
        start_time = time.monotonic()

        try:
            # Inject node_id so executors can write unique filenames
            config["node_id"] = node_id
            results = self._executor.run(config, context)
            duration_ms = int((time.monotonic() - start_time) * 1000)

            # Evaluate
            score = self._evaluator.evaluate(results, config)

            # Mark completed
            self.manager.mark_completed(
                node_id,
                results=results,
                score=score,
                duration_ms=duration_ms,
            )

            if self.verbose:
                logger.info(
                    "Node %s completed: score=%.4f depth=%d duration=%dms",
                    node_id, score, node["depth"], duration_ms,
                )

            # Callback
            updated_node = self.store.get_node(node_id)
            if self.on_node_complete and updated_node:
                self.on_node_complete(updated_node)

            # Prune check
            best_score = self.manager.get_best_score(tree_id) or 0
            siblings = self.store.get_siblings(node_id)
            should_prune, reason = self._strategy.should_prune(
                {"id": node_id, "score": score, **node},
                best_score,
                siblings,
            )
            if should_prune:
                self.manager.prune_node(node_id, reason)
                return

            # Expand: generate children if not at max depth
            if node["depth"] < self.max_depth:
                self._expand_node(tree_id, node_id, config, results, context)

        except Exception as e:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            self.manager.mark_failed(node_id, str(e))
            logger.error("Node %s failed after %dms: %s", node_id, duration_ms, e)

    def _expand_node(
        self,
        tree_id: str,
        node_id: str,
        config: dict[str, Any],
        results: dict[str, Any],
        context: BranchContext,
    ) -> None:
        """Generate child configs from a completed node."""
        try:
            children = self._mutator(config, results, context)
            if children:
                self.manager.add_child_nodes(node_id, children)
                logger.debug("Expanded node %s with %d children", node_id, len(children))
        except Exception as e:
            logger.warning("Mutator failed for node %s: %s", node_id, e)

    def _expand_from_completed(
        self,
        tree_id: str,
        completed: list[dict[str, Any]],
    ) -> bool:
        """Try to expand from the best completed leaf nodes."""
        # Find leaf nodes (no children) that haven't been pruned
        leaf_nodes = []
        for node in completed:
            if node["pruned"]:
                continue
            if node["depth"] >= self.max_depth:
                continue
            children = self.store.get_children(node["id"])
            if not children:
                leaf_nodes.append(node)

        if not leaf_nodes:
            return False

        # Expand from the best leaf
        leaf_nodes.sort(key=lambda n: n["score"] or 0, reverse=True)
        best_leaf = leaf_nodes[0]
        config = json.loads(best_leaf["config"]) if isinstance(best_leaf["config"], str) else best_leaf["config"]
        results = json.loads(best_leaf["results"]) if isinstance(best_leaf["results"], str) else (best_leaf["results"] or {})
        context = self.manager.get_branch_context(best_leaf, self.goal)
        self._expand_node(tree_id, best_leaf["id"], config, results, context)
        return True
