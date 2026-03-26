"""Scientist — main orchestrator for the observe-hypothesize-execute-analyze loop."""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from arborist.scientist.hypothesis import Hypothesis, HypothesisGenerator
from arborist.scientist.journal import Journal
from arborist.scientist.moves import MoveCategory, get_move
from arborist.scientist.observer import Observation, Observer
from arborist.store import Store
from arborist.synthesis import SearchResults

logger = logging.getLogger(__name__)


@dataclass
class RoundResult:
    """Result of a single scientist round."""

    round_number: int
    hypothesis: Hypothesis
    outcome: str  # CONFIRMED, REJECTED, INCONCLUSIVE
    score_before: float | None
    score_after: float | None
    tree_id: str | None = None
    findings: list[str] = field(default_factory=list)
    experiments_used: int = 0
    duration_seconds: float = 0.0

    @property
    def improvement(self) -> float | None:
        if self.score_before is not None and self.score_after is not None:
            return self.score_after - self.score_before
        return None


@dataclass
class SessionResult:
    """Result of an entire scientist session."""

    session_id: str
    problem: str
    rounds: list[RoundResult] = field(default_factory=list)
    total_experiments: int = 0
    best_score: float | None = None
    best_config: dict[str, Any] | None = None
    duration_seconds: float = 0.0

    def summary(self) -> str:
        lines = [
            f"# Scientist Session: {self.session_id}",
            f"Problem: {self.problem}",
            f"Rounds: {len(self.rounds)}",
            f"Total experiments: {self.total_experiments}",
        ]
        if self.best_score is not None:
            lines.append(f"Best score: {self.best_score:.4f}")
        lines.append(f"Duration: {self.duration_seconds:.1f}s")
        lines.append("")

        for r in self.rounds:
            outcome_symbol = {
                "CONFIRMED": "+", "REJECTED": "-", "INCONCLUSIVE": "?"
            }.get(r.outcome, "?")
            imp = f" ({r.improvement:+.4f})" if r.improvement is not None else ""
            lines.append(
                f"  [{outcome_symbol}] Round {r.round_number}: "
                f"{r.hypothesis.description[:60]}{imp}"
            )

        return "\n".join(lines)


class Scientist:
    """Autonomous research agent that iterates through observe-hypothesize-execute-analyze.

    Sits above Arborist's TreeSearch — uses it as the execution engine
    for testing hypotheses.
    """

    def __init__(
        self,
        problem: str,
        executor: Callable[[dict[str, Any]], dict[str, Any]] | Any,
        score: Callable[[dict[str, Any]], float] | Any | str | None = None,
        baseline_config: dict[str, Any] | None = None,
        metric_name: str = "f1_macro",
        max_rounds: int = 10,
        total_budget: int = 200,
        model: str = "openrouter/anthropic/claude-sonnet-4-5",
        mutator_model: str = "openrouter/anthropic/claude-haiku-4-5",
        db_path: str = "./scientist.db",
        journal_dir: str = "./scientist_journals",
        memory_url: str | None = "http://localhost:8642",
        human_in_the_loop: bool = False,
        on_round_complete: Callable[[RoundResult], None] | None = None,
        verbose: bool = True,
        base_script: str | None = None,
        code_gen_output_dir: str = "./experiments/generated",
    ) -> None:
        self.problem = problem
        self.executor = executor
        self.score = score
        self.baseline_config = baseline_config or {}
        self.metric_name = metric_name
        self.max_rounds = max_rounds
        self.total_budget = total_budget
        self.model = model
        self.mutator_model = mutator_model
        self.db_path = db_path
        self.human_in_the_loop = human_in_the_loop
        self.on_round_complete = on_round_complete
        self.verbose = verbose
        self.base_script = base_script
        self.code_gen_output_dir = code_gen_output_dir

        # Session tracking
        self.session_id = uuid.uuid4().hex[:12]
        self._budget_used = 0
        self._best_score: float | None = None
        self._best_config: dict[str, Any] | None = None
        self._score_history: list[float] = []
        self._all_results: dict[str, Any] = {}

        # Components
        self._observer = Observer(
            metric_name=metric_name,
            plateau_window=10,
        )
        self._hypothesis_gen = HypothesisGenerator(
            model=model,
            max_hypotheses=5,
        )
        self._journal = Journal(
            session_id=self.session_id,
            journal_dir=journal_dir,
            memory_url=memory_url,
        )

        if verbose:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            )

    @property
    def budget_remaining(self) -> int:
        return max(0, self.total_budget - self._budget_used)

    def run(self) -> SessionResult:
        """Run the full scientist loop. Blocking."""
        session_start = time.monotonic()
        result = SessionResult(
            session_id=self.session_id,
            problem=self.problem,
        )

        logger.info(
            "Starting scientist session %s: %s (budget=%d, max_rounds=%d)",
            self.session_id, self.problem, self.total_budget, self.max_rounds,
        )

        for round_num in range(1, self.max_rounds + 1):
            if self.budget_remaining <= 0:
                logger.info("Budget exhausted after %d rounds", round_num - 1)
                break

            logger.info(
                "=== Round %d/%d (budget remaining: %d) ===",
                round_num, self.max_rounds, self.budget_remaining,
            )

            try:
                round_result = self._run_round(round_num)
            except KeyboardInterrupt:
                logger.info("Session interrupted at round %d", round_num)
                break
            except Exception as e:
                logger.error("Round %d failed: %s", round_num, e)
                continue

            result.rounds.append(round_result)
            result.total_experiments += round_result.experiments_used

            if self.on_round_complete:
                self.on_round_complete(round_result)

            # Check for early stopping
            if self._should_stop_early(result):
                logger.info("Early stopping: diminishing returns")
                break

        result.best_score = self._best_score
        result.best_config = self._best_config
        result.duration_seconds = time.monotonic() - session_start

        logger.info("Session complete: %s", result.summary())
        return result

    def _run_round(self, round_num: int) -> RoundResult:
        """Execute one observe-hypothesize-execute-analyze cycle."""
        round_start = time.monotonic()
        score_before = self._best_score

        # 1. OBSERVE
        observation = self._observe()
        logger.info("Observation: %s", observation.summary())

        # 2. HYPOTHESIZE
        journal_context = self._journal.get_context_for_llm()
        hypotheses = self._hypothesis_gen.generate(
            observation=observation,
            problem=self.problem,
            journal_context=journal_context,
        )

        if not hypotheses:
            logger.warning("No hypotheses generated, ending session")
            return RoundResult(
                round_number=round_num,
                hypothesis=Hypothesis(
                    id="none", description="No hypotheses generated",
                    move_category=MoveCategory.PARAM_TUNING,
                    expected_impact=0, confidence=0,
                    effort="LOW", rationale="", falsifiable="",
                ),
                outcome="INCONCLUSIVE",
                score_before=score_before,
                score_after=self._best_score,
            )

        # Pick top hypothesis
        hypothesis = hypotheses[0]
        logger.info(
            "Testing hypothesis %s: %s (priority=%.2f)",
            hypothesis.id, hypothesis.description, hypothesis.priority_score,
        )

        # Human-in-the-loop checkpoint
        if self.human_in_the_loop:
            self._human_checkpoint(hypothesis, hypotheses)

        # Record hypothesis
        self._journal.record_hypothesis(
            hypothesis_id=hypothesis.id,
            description=hypothesis.description,
            move_category=hypothesis.move_category.value,
            rationale=hypothesis.rationale,
        )

        # 3. DESIGN experiment
        move = get_move(hypothesis.move_category)
        round_budget = min(
            move.default_budget,
            self.budget_remaining,
        )
        search_config = move.generate_config(
            observation=observation,
            budget_override=round_budget,
        )

        # 4. EXECUTE via TreeSearch
        # Use CodeGeneratorExecutor for code-generation moves
        use_code_gen = (
            move.executor_type == "code_generator" and self.base_script is not None
        )

        tree_id, experiments_used = self._execute_search(
            hypothesis=hypothesis,
            search_config=search_config,
            use_code_generator=use_code_gen,
        )
        self._budget_used += experiments_used

        # 5. ANALYZE
        outcome, findings = self._analyze(
            hypothesis=hypothesis,
            tree_id=tree_id,
            score_before=score_before,
        )

        # Record result
        self._journal.record_result(
            hypothesis_id=hypothesis.id,
            outcome=outcome,
            score_before=score_before,
            score_after=self._best_score,
            findings=findings,
            best_config=self._best_config,
        )

        if outcome == "REJECTED":
            self._journal.record_dead_end(
                hypothesis.id,
                f"{hypothesis.description} — did not improve score",
            )

        round_result = RoundResult(
            round_number=round_num,
            hypothesis=hypothesis,
            outcome=outcome,
            score_before=score_before,
            score_after=self._best_score,
            tree_id=tree_id,
            findings=findings,
            experiments_used=experiments_used,
            duration_seconds=time.monotonic() - round_start,
        )

        logger.info(
            "Round %d complete: %s (improvement: %s)",
            round_num,
            outcome,
            f"{round_result.improvement:+.4f}" if round_result.improvement is not None else "N/A",
        )

        return round_result

    def _observe(self) -> Observation:
        """Phase 1: Produce structured observation of current state."""
        # Get latest results and score history from store
        store = Store(self.db_path)
        try:
            best_node = None
            best_results = None

            # Look across all trees for this session's best
            trees = store.list_trees()
            for tree in trees:
                node = store.get_best_node(tree["id"])
                if node and (best_node is None or (node["score"] or 0) > (best_node["score"] or 0)):
                    best_node = node

            if best_node and best_node.get("results"):
                results_raw = best_node["results"]
                if isinstance(results_raw, str):
                    best_results = json.loads(results_raw)
                else:
                    best_results = results_raw

            return self._observer.observe(
                results=best_results,
                score_history=self._score_history,
                experiment_count=self._budget_used,
                best_score=self._best_score,
                best_config=self._best_config,
            )
        finally:
            store.close()

    def _execute_search(
        self,
        hypothesis: Hypothesis,
        search_config: dict[str, Any],
        use_code_generator: bool = False,
    ) -> tuple[str | None, int]:
        """Phase 4: Run a TreeSearch for the given hypothesis.

        Args:
            hypothesis: The hypothesis being tested.
            search_config: Config from move.generate_config().
            use_code_generator: If True, use CodeGeneratorExecutor and
                code-gen seed configs instead of numeric param tuning.

        Returns (tree_id, experiments_used).
        """
        from arborist.tree import TreeSearch

        # Configure mutator based on move type
        mutator_type = search_config.pop("mutator", "llm")
        mutator = self._get_mutator(mutator_type)

        strategy = search_config.pop("strategy", "ucb")
        max_experiments = search_config.pop("max_experiments", 30)
        max_depth = search_config.pop("max_depth", 4)

        # Remove non-TreeSearch keys
        extra_keys = {"exploration_weight"}
        for k in extra_keys:
            search_config.pop(k, None)

        # Determine executor and seed configs
        if use_code_generator:
            from arborist.executors.code_generator import CodeGeneratorExecutor

            executor = CodeGeneratorExecutor(
                model=self.model,
                output_dir=self.code_gen_output_dir,
            )
            # Seed config for code gen: base_script + hypothesis description
            seed_configs = [{
                "base_script": self.base_script,
                "modifications": hypothesis.description,
            }]
            # Use code-gen-aware mutator
            mutator = self._get_code_gen_mutator()
        else:
            executor = self.executor
            seed_configs = [self.baseline_config] if self.baseline_config else [{}]
            if self._best_config and self._best_config != self.baseline_config:
                seed_configs.append(self._best_config)

        try:
            search = TreeSearch(
                goal=f"[{hypothesis.id}] {hypothesis.description}",
                executor=executor,
                score=self.score,
                seed_configs=seed_configs,
                strategy=strategy,
                mutator=mutator,
                concurrency=3,
                max_experiments=max_experiments,
                max_depth=max_depth,
                db_path=self.db_path,
                verbose=self.verbose,
            )

            results = search.run()

            # Update global best
            best = results.best
            if best and best.get("score") is not None:
                score = best["score"]
                self._score_history.append(score)
                if self._best_score is None or score > self._best_score:
                    self._best_score = score
                    self._best_config = best.get("config")

            # Count experiments
            store = Store(self.db_path)
            try:
                experiments_used = store.count_nodes(results.tree_id)
            finally:
                store.close()

            return results.tree_id, experiments_used

        except Exception as e:
            logger.error("TreeSearch failed for hypothesis %s: %s", hypothesis.id, e)
            return None, 0

    def _get_code_gen_mutator(self) -> Any:
        """Return a mutator that generates modification instructions for code gen.

        Instead of mutating numeric params, this mutator asks the LLM to propose
        new modification instructions. The parent node's generated_script becomes
        the new base for children, and mutations are additive.
        """
        from arborist.mutators import CodeGenMutator
        return CodeGenMutator(
            model=self.mutator_model,
            base_script=self.base_script,
        )

    def _get_mutator(self, mutator_type: str) -> Any:
        """Get the appropriate mutator for the experiment type."""
        if mutator_type == "llm":
            from arborist.mutators import LLMMutator
            return LLMMutator(model=self.mutator_model)
        else:
            from arborist.mutators import RandomMutator
            return RandomMutator()

    def _analyze(
        self,
        hypothesis: Hypothesis,
        tree_id: str | None,
        score_before: float | None,
    ) -> tuple[str, list[str]]:
        """Phase 5: Analyze results and determine outcome.

        Returns (outcome, findings).
        """
        findings: list[str] = []
        score_after = self._best_score

        if tree_id is None:
            return "INCONCLUSIVE", ["TreeSearch failed to execute"]

        # Get search results
        store = Store(self.db_path)
        try:
            best_node = store.get_best_node(tree_id)
            top_nodes = store.get_top_nodes(tree_id, k=5)
            total = store.count_nodes(tree_id, status="completed")
            failed = store.count_nodes(tree_id, status="failed")

            if best_node:
                findings.append(
                    f"Best score in this search: {best_node['score']:.4f}"
                )

            if failed > 0:
                findings.append(
                    f"{failed}/{total + failed} experiments failed"
                )

            # Determine outcome
            if score_before is None:
                # First round — any result is a baseline
                outcome = "CONFIRMED" if best_node else "INCONCLUSIVE"
                findings.append("Established baseline")
            elif score_after is not None and score_before is not None:
                improvement = score_after - score_before
                if improvement > 0.01:  # >1% improvement
                    outcome = "CONFIRMED"
                    findings.append(f"Improvement: {improvement:+.4f}")
                elif improvement > -0.005:  # within noise
                    outcome = "INCONCLUSIVE"
                    findings.append(f"Marginal change: {improvement:+.4f}")
                else:
                    outcome = "REJECTED"
                    findings.append(f"Regression: {improvement:+.4f}")
            else:
                outcome = "INCONCLUSIVE"

        finally:
            store.close()

        return outcome, findings

    def _should_stop_early(self, result: SessionResult) -> bool:
        """Check if we should stop the session early."""
        if len(result.rounds) < 3:
            return False

        # Stop if last 3 rounds all rejected or inconclusive
        recent = result.rounds[-3:]
        if all(r.outcome in ("REJECTED", "INCONCLUSIVE") for r in recent):
            return True

        return False

    def _human_checkpoint(
        self,
        selected: Hypothesis,
        all_hypotheses: list[Hypothesis],
    ) -> None:
        """Pause for human approval before executing."""
        print(f"\n{'='*60}")
        print(f"SCIENTIST CHECKPOINT - Round")
        print(f"{'='*60}")
        print(f"\nSelected hypothesis: {selected.description}")
        print(f"  Category: {selected.move_category.value}")
        print(f"  Expected impact: {selected.expected_impact:.1%}")
        print(f"  Confidence: {selected.confidence:.1%}")
        print(f"  Effort: {selected.effort}")
        print(f"  Rationale: {selected.rationale}")
        print(f"\nAlternatives:")
        for h in all_hypotheses[1:]:
            print(f"  - {h.description} (priority={h.priority_score:.2f})")
        print(f"\nPress Enter to continue, or Ctrl+C to stop...")
        input()
