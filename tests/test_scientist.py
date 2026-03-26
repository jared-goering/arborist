"""Tests for the arborist.scientist layer."""

from __future__ import annotations

import json
import math
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from arborist.scientist.moves import (
    MOVE_LIBRARY,
    MoveCategory,
    ResearchMove,
    get_move,
)
from arborist.scientist.observer import Observation, Observer
from arborist.scientist.hypothesis import Hypothesis, HypothesisGenerator
from arborist.scientist.journal import Journal, JournalEntry
from arborist.scientist.scientist import Scientist, RoundResult, SessionResult


# ── Moves ─────────────────────────────────────────────────────────────


class TestMoveCategory:
    def test_all_categories_in_library(self):
        for cat in MoveCategory:
            assert cat in MOVE_LIBRARY

    def test_get_move_by_enum(self):
        move = get_move(MoveCategory.PARAM_TUNING)
        assert move.name == "Hyperparameter Tuning"
        assert move.category == MoveCategory.PARAM_TUNING

    def test_get_move_by_string(self):
        move = get_move("feature_engineering")
        assert move.category == MoveCategory.FEATURE_ENGINEERING

    def test_get_move_invalid(self):
        with pytest.raises(ValueError):
            get_move("nonexistent_move")


class TestResearchMove:
    def test_generate_config_defaults(self):
        move = get_move(MoveCategory.PARAM_TUNING)
        config = move.generate_config()
        assert config["strategy"] == "ucb"
        assert config["mutator"] == "random"
        assert config["max_experiments"] == 50
        assert config["max_depth"] == 5

    def test_generate_config_budget_override(self):
        move = get_move(MoveCategory.ARCHITECTURE)
        config = move.generate_config(budget_override=100)
        assert config["max_experiments"] == 100

    def test_generate_config_with_kwargs(self):
        move = get_move(MoveCategory.FEATURE_ENGINEERING)
        config = move.generate_config(concurrency=10)
        assert config["concurrency"] == 10

    def test_generate_config_adapts_to_observation(self):
        move = get_move(MoveCategory.PARAM_TUNING)
        obs = Observation(diminishing_returns=True)
        config = move.generate_config(observation=obs)
        # Should reduce budget when diminishing returns
        assert config["max_experiments"] <= 50

    def test_generate_config_plateau_increases_exploration(self):
        move = get_move(MoveCategory.PARAM_TUNING)
        obs = Observation(plateau_detected=True)
        config = move.generate_config(observation=obs)
        assert config.get("exploration_weight", 1.0) >= 1.0

    def test_each_move_has_description(self):
        for cat, move in MOVE_LIBRARY.items():
            assert move.description
            assert move.name
            assert len(move.applicable_when) > 0


# ── Observer ──────────────────────────────────────────────────────────


class TestObserver:
    def test_basic_observe(self):
        observer = Observer(metric_name="f1_macro")
        obs = observer.observe(
            results={},
            score_history=[0.3, 0.32, 0.35],
            experiment_count=10,
            best_score=0.35,
            best_config={"lr": 0.01},
        )
        assert obs.experiment_count == 10
        assert obs.current_best["value"] == 0.35
        assert obs.current_best["metric"] == "f1_macro"

    def test_observe_empty(self):
        observer = Observer()
        obs = observer.observe()
        assert obs.experiment_count == 0
        assert obs.current_best == {}


class TestObserverClassificationReport:
    def test_parse_dict_report(self):
        observer = Observer()
        results = {
            "classification_report": {
                "Wake": {"precision": 0.8, "recall": 0.6, "f1-score": 0.69, "support": 100},
                "N1": {"precision": 0.3, "recall": 0.2, "f1-score": 0.24, "support": 50},
                "N2": {"precision": 0.7, "recall": 0.8, "f1-score": 0.75, "support": 200},
                "accuracy": 0.65,
                "macro avg": {"precision": 0.6, "recall": 0.53, "f1-score": 0.56, "support": 350},
            }
        }
        obs = observer.observe(results=results)
        assert "Wake" in obs.class_metrics
        assert "N1" in obs.class_metrics
        assert obs.class_metrics["N1"]["f1"] == 0.24
        assert "N1" in obs.worst_classes  # F1 < 0.5

    def test_parse_text_report(self):
        observer = Observer()
        text = (
            "              precision    recall  f1-score   support\n"
            "\n"
            "        Wake       0.80      0.60      0.69       100\n"
            "          N1       0.30      0.20      0.24        50\n"
            "          N2       0.70      0.80      0.75       200\n"
        )
        results = {"classification_report": text}
        obs = observer.observe(results=results)
        assert "Wake" in obs.class_metrics
        assert abs(obs.class_metrics["Wake"]["precision"] - 0.80) < 0.01

    def test_error_pattern_detection(self):
        observer = Observer()
        results = {
            "classification_report": {
                "Wake": {"precision": 0.9, "recall": 0.3, "f1-score": 0.45, "support": 100},
            }
        }
        obs = observer.observe(results=results)
        assert any("low recall" in p for p in obs.error_patterns)


class TestObserverConfusionMatrix:
    def test_parse_confusion_matrix(self):
        observer = Observer()
        results = {
            "confusion_matrix": [[80, 10, 10], [20, 30, 0], [5, 5, 190]],
            "labels": ["Wake", "N1", "N2"],
        }
        obs = observer.observe(results=results)
        assert obs.confusion_matrix["Wake"]["Wake"] == 80
        assert obs.confusion_matrix["Wake"]["N1"] == 10

    def test_confusion_error_patterns(self):
        observer = Observer()
        results = {
            "confusion_matrix": [[50, 30, 20], [10, 80, 10], [5, 5, 90]],
            "labels": ["Wake", "N1", "N2"],
        }
        obs = observer.observe(results=results)
        # Wake misclassified as N1 30% of the time
        assert any("Wake" in p and "N1" in p for p in obs.error_patterns)


class TestObserverMultiSeed:
    def test_multi_seed_stats(self):
        observer = Observer()
        scores = [0.35, 0.37, 0.34, 0.36, 0.38]
        obs = observer.observe_multi_seed(scores)
        assert obs.score_mean is not None
        assert abs(obs.score_mean - 0.36) < 0.01
        assert obs.score_std is not None
        assert obs.score_std > 0
        assert obs.score_ci_95 is not None
        assert obs.score_ci_95[0] < obs.score_mean < obs.score_ci_95[1]

    def test_single_seed(self):
        observer = Observer()
        obs = observer.observe_multi_seed([0.42])
        assert obs.score_mean == 0.42
        assert obs.score_std == 0.0
        assert obs.score_ci_95 is None

    def test_empty_seeds(self):
        observer = Observer()
        obs = observer.observe_multi_seed([])
        assert obs.score_mean is None


class TestObserverPlateau:
    def test_plateau_detection(self):
        observer = Observer(plateau_window=5, plateau_threshold=0.01)
        # Scores that plateau
        history = [0.1, 0.2, 0.3, 0.35, 0.36, 0.36, 0.361, 0.362, 0.361, 0.362]
        obs = observer.observe(score_history=history)
        assert obs.plateau_detected

    def test_no_plateau_when_improving(self):
        observer = Observer(plateau_window=3, plateau_threshold=0.01)
        history = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        obs = observer.observe(score_history=history)
        assert not obs.plateau_detected

    def test_diminishing_returns(self):
        observer = Observer(plateau_window=3, plateau_threshold=0.01)
        # Big improvement, then small
        history = [0.1, 0.2, 0.3, 0.35, 0.36, 0.36, 0.361, 0.362, 0.362]
        obs = observer.observe(score_history=history)
        assert obs.diminishing_returns


class TestObserverFeatureImportance:
    def test_dict_feature_importance(self):
        observer = Observer()
        results = {
            "feature_importance": {"hr_mean": 0.5, "accel_std": 0.3, "age": 0.1, "other": 0.1}
        }
        obs = observer.observe(results=results)
        assert obs.feature_importance["hr_mean"] == 0.5

    def test_list_feature_importance(self):
        observer = Observer()
        results = {
            "feature_importance": [0.5, 0.3, 0.1, 0.1],
            "feature_names": ["hr_mean", "accel_std", "age", "other"],
        }
        obs = observer.observe(results=results)
        assert obs.feature_importance["hr_mean"] == 0.5

    def test_concentrated_importance_gap(self):
        observer = Observer()
        results = {
            "feature_importance": {"f1": 0.7, "f2": 0.15, "f3": 0.1, "f4": 0.03, "f5": 0.02}
        }
        obs = observer.observe(results=results)
        assert len(obs.feature_gaps) > 0


class TestObserverSubjectScores:
    def test_subject_variance(self):
        observer = Observer()
        results = {
            "subject_scores": {
                "S01": 0.8, "S02": 0.1, "S03": 0.8,
                "S04": 0.8, "S05": 0.8, "S06": 0.8,
                "S07": 0.8, "S08": 0.8,
            }
        }
        obs = observer.observe(results=results)
        assert obs.subject_variance is not None
        assert obs.subject_variance > 0
        # S02 is far below mean with enough subjects for tight std
        assert any("S02" in d for d in obs.data_issues)


class TestObserverDataIssues:
    def test_class_imbalance_detection(self):
        observer = Observer()
        results = {
            "classification_report": {
                "A": {"precision": 0.9, "recall": 0.9, "f1-score": 0.9, "support": 900},
                "B": {"precision": 0.3, "recall": 0.3, "f1-score": 0.3, "support": 30},
                "C": {"precision": 0.5, "recall": 0.5, "f1-score": 0.5, "support": 70},
            }
        }
        obs = observer.observe(results=results)
        assert any("underrepresented" in d for d in obs.data_issues)
        assert any("dominates" in d for d in obs.data_issues)


class TestObservationSerialization:
    def test_to_dict(self):
        obs = Observation(
            experiment_count=10,
            current_best={"metric": "f1", "value": 0.5},
            worst_classes=["N1"],
            plateau_detected=True,
        )
        d = obs.to_dict()
        assert d["experiment_count"] == 10
        assert d["plateau_detected"] is True
        assert d["worst_classes"] == ["N1"]

    def test_summary(self):
        obs = Observation(
            current_best={"metric": "f1", "value": 0.5},
            experiment_count=10,
            worst_classes=["N1", "REM"],
            error_patterns=["N1 misclassified as Wake"],
            plateau_detected=True,
        )
        s = obs.summary()
        assert "f1" in s
        assert "N1" in s
        assert "plateau" in s.lower()


# ── Hypothesis ────────────────────────────────────────────────────────


class TestHypothesis:
    def test_from_dict(self):
        data = {
            "id": "h_001",
            "description": "Add spectral entropy features",
            "move_category": "feature_engineering",
            "expected_impact": 0.6,
            "confidence": 0.7,
            "effort": "MEDIUM",
            "rationale": "Wake/N1 confusion suggests missing frequency features",
            "falsifiable": "If Wake F1 doesn't improve by >2%, reject",
        }
        h = Hypothesis.from_dict(data)
        assert h.id == "h_001"
        assert h.move_category == MoveCategory.FEATURE_ENGINEERING
        assert h.expected_impact == 0.6

    def test_to_dict_roundtrip(self):
        h = Hypothesis(
            id="h_001",
            description="Test",
            move_category=MoveCategory.PARAM_TUNING,
            expected_impact=0.5,
            confidence=0.5,
            effort="LOW",
            rationale="reason",
            falsifiable="criteria",
        )
        d = h.to_dict()
        h2 = Hypothesis.from_dict(d)
        assert h2.id == h.id
        assert h2.move_category == h.move_category

    def test_priority_score(self):
        h_low = Hypothesis(
            id="h1", description="", move_category=MoveCategory.PARAM_TUNING,
            expected_impact=0.3, confidence=0.3, effort="HIGH",
            rationale="", falsifiable="",
        )
        h_high = Hypothesis(
            id="h2", description="", move_category=MoveCategory.PARAM_TUNING,
            expected_impact=0.8, confidence=0.9, effort="LOW",
            rationale="", falsifiable="",
        )
        assert h_high.priority_score > h_low.priority_score

    def test_effort_affects_priority(self):
        base = dict(
            id="h", description="", move_category=MoveCategory.PARAM_TUNING,
            expected_impact=0.5, confidence=0.5, rationale="", falsifiable="",
        )
        h_low = Hypothesis(**base, effort="LOW")
        h_high = Hypothesis(**base, effort="HIGH")
        assert h_low.priority_score > h_high.priority_score


class TestHypothesisGenerator:
    def test_generate_with_mocked_llm(self):
        gen = HypothesisGenerator(model="test-model")
        obs = Observation(
            worst_classes=["N1"],
            current_best={"metric": "f1", "value": 0.35},
            experiment_count=10,
        )

        mock_response = {
            "hypotheses": [
                {
                    "description": "Add frequency features for N1",
                    "move_category": "feature_engineering",
                    "expected_impact": 0.6,
                    "confidence": 0.7,
                    "effort": "MEDIUM",
                    "rationale": "N1 has lowest F1",
                    "falsifiable": "If N1 F1 < 0.3, reject",
                },
                {
                    "description": "Tune learning rate",
                    "move_category": "param_tuning",
                    "expected_impact": 0.3,
                    "confidence": 0.8,
                    "effort": "LOW",
                    "rationale": "Quick win",
                    "falsifiable": "If F1 < 0.36, reject",
                },
            ]
        }

        mock_msg = MagicMock()
        mock_msg.content = json.dumps(mock_response)
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]

        with patch("litellm.completion", return_value=mock_resp):
            hypotheses = gen.generate(obs, problem="Improve sleep staging")

        assert len(hypotheses) == 2
        assert all(isinstance(h, Hypothesis) for h in hypotheses)
        # Should be sorted by priority
        assert hypotheses[0].priority_score >= hypotheses[1].priority_score

    def test_fallback_on_llm_failure(self):
        gen = HypothesisGenerator(model="test-model")
        obs = Observation(
            worst_classes=["N1"],
            data_issues=["Class N1 underrepresented"],
        )

        with patch("litellm.completion", side_effect=Exception("API error")):
            hypotheses = gen.generate(obs, problem="Test problem")

        # Should get fallback hypotheses
        assert len(hypotheses) > 0
        assert any(h.move_category == MoveCategory.PARAM_TUNING for h in hypotheses)

    def test_parse_malformed_response(self):
        gen = HypothesisGenerator()
        # Missing required fields
        result = gen._parse_response('{"hypotheses": [{"description": "test"}]}')
        assert len(result) == 1  # Should still parse with defaults

    def test_parse_empty_response(self):
        gen = HypothesisGenerator()
        result = gen._parse_response("")
        assert result == []

    def test_parse_none_response(self):
        gen = HypothesisGenerator()
        result = gen._parse_response(None)
        assert result == []


# ── Journal ───────────────────────────────────────────────────────────


class TestJournal:
    @pytest.fixture
    def tmp_journal_dir(self, tmp_path):
        return str(tmp_path / "journals")

    def test_record_and_retrieve(self, tmp_journal_dir):
        journal = Journal(
            session_id="test_001",
            journal_dir=tmp_journal_dir,
            memory_url=None,
        )
        journal.record_hypothesis(
            hypothesis_id="h_001",
            description="Add features",
            move_category="feature_engineering",
            rationale="Missing signal",
        )
        journal.record_result(
            hypothesis_id="h_001",
            outcome="CONFIRMED",
            score_before=0.35,
            score_after=0.38,
            findings=["Spectral entropy helped"],
        )

        assert len(journal.entries) == 2
        outcomes = journal.get_hypothesis_outcomes()
        assert outcomes["h_001"] == "CONFIRMED"

    def test_dead_ends(self, tmp_journal_dir):
        journal = Journal(
            session_id="test_002",
            journal_dir=tmp_journal_dir,
            memory_url=None,
        )
        journal.record_dead_end("h_bad", "HR features don't help for REM")
        dead_ends = journal.get_dead_ends()
        assert len(dead_ends) == 1
        assert "HR features" in dead_ends[0]

    def test_confirmed_findings(self, tmp_journal_dir):
        journal = Journal(
            session_id="test_003",
            journal_dir=tmp_journal_dir,
            memory_url=None,
        )
        journal.record_hypothesis("h_001", "Test", "param_tuning", "reason")
        journal.record_result("h_001", "CONFIRMED", findings=["LR=0.01 is optimal"])
        journal.record_hypothesis("h_002", "Test2", "param_tuning", "reason2")
        journal.record_result("h_002", "REJECTED", findings=["Didn't help"])

        findings = journal.get_confirmed_findings()
        assert "LR=0.01 is optimal" in findings
        assert "Didn't help" not in findings

    def test_persistence(self, tmp_journal_dir):
        # Write
        j1 = Journal(session_id="test_persist", journal_dir=tmp_journal_dir, memory_url=None)
        j1.record_finding("Important insight", tags=["key"])
        assert len(j1.entries) == 1

        # Read back in new instance
        j2 = Journal(session_id="test_persist", journal_dir=tmp_journal_dir, memory_url=None)
        assert len(j2.entries) == 1
        assert j2.entries[0].content == "Important insight"

    def test_context_for_llm(self, tmp_journal_dir):
        journal = Journal(session_id="test_ctx", journal_dir=tmp_journal_dir, memory_url=None)
        journal.record_hypothesis("h_001", "Test hypothesis", "param_tuning", "reason")
        journal.record_result("h_001", "CONFIRMED", score_before=0.3, score_after=0.35)
        journal.record_dead_end("h_002", "Bad idea")

        ctx = journal.get_context_for_llm()
        assert "test_ctx" in ctx
        assert "CONFIRMED" in ctx
        assert "Dead Ends" in ctx

    def test_empty_journal_context(self, tmp_journal_dir):
        journal = Journal(session_id="empty", journal_dir=tmp_journal_dir, memory_url=None)
        ctx = journal.get_context_for_llm()
        assert "No previous experiments" in ctx


class TestJournalEntry:
    def test_roundtrip(self):
        entry = JournalEntry(
            timestamp="2024-01-01T00:00:00",
            entry_type="hypothesis",
            hypothesis_id="h_001",
            content="Test content",
            outcome="CONFIRMED",
            score_before=0.3,
            score_after=0.35,
            tags=["tag1"],
            metadata={"key": "value"},
        )
        d = entry.to_dict()
        e2 = JournalEntry.from_dict(d)
        assert e2.hypothesis_id == "h_001"
        assert e2.outcome == "CONFIRMED"
        assert e2.score_before == 0.3


# ── Scientist Main Loop ──────────────────────────────────────────────


class TestScientist:
    @pytest.fixture
    def tmp_dir(self, tmp_path):
        return str(tmp_path)

    def _make_executor(self):
        """Simple executor that returns a score based on 'x' param."""
        def executor(config, context=None):
            x = config.get("x", 0)
            score = -(x - 3) ** 2 + 10
            return {"score": score}
        return executor

    def test_scientist_init(self, tmp_dir):
        sci = Scientist(
            problem="Test problem",
            executor=self._make_executor(),
            score="score",
            db_path=os.path.join(tmp_dir, "test.db"),
            journal_dir=os.path.join(tmp_dir, "journals"),
            memory_url=None,
            max_rounds=2,
            total_budget=20,
            verbose=False,
        )
        assert sci.session_id
        assert sci.budget_remaining == 20

    def test_scientist_run_with_mocked_hypotheses(self, tmp_dir):
        """Test the full loop with mocked LLM calls."""
        executor = self._make_executor()

        sci = Scientist(
            problem="Maximize f(x) = -(x-3)^2 + 10",
            executor=executor,
            score="score",
            baseline_config={"x": 0},
            metric_name="score",
            max_rounds=2,
            total_budget=30,
            db_path=os.path.join(tmp_dir, "test.db"),
            journal_dir=os.path.join(tmp_dir, "journals"),
            memory_url=None,
            verbose=False,
        )

        # Mock the hypothesis generator to return a simple hypothesis
        mock_hypothesis = Hypothesis(
            id="h_001",
            description="Tune x parameter",
            move_category=MoveCategory.PARAM_TUNING,
            expected_impact=0.5,
            confidence=0.8,
            effort="LOW",
            rationale="x is the key parameter",
            falsifiable="If score doesn't improve, reject",
        )
        sci._hypothesis_gen.generate = MagicMock(return_value=[mock_hypothesis])

        result = sci.run()

        assert isinstance(result, SessionResult)
        assert result.session_id == sci.session_id
        assert len(result.rounds) > 0
        assert result.total_experiments > 0
        assert result.best_score is not None

    def test_budget_tracking(self, tmp_dir):
        sci = Scientist(
            problem="Test",
            executor=self._make_executor(),
            score="score",
            baseline_config={"x": 1},
            max_rounds=5,
            total_budget=10,
            db_path=os.path.join(tmp_dir, "budget.db"),
            journal_dir=os.path.join(tmp_dir, "journals"),
            memory_url=None,
            verbose=False,
        )

        mock_h = Hypothesis(
            id="h_001", description="Test", move_category=MoveCategory.PARAM_TUNING,
            expected_impact=0.5, confidence=0.5, effort="LOW",
            rationale="", falsifiable="",
        )
        sci._hypothesis_gen.generate = MagicMock(return_value=[mock_h])

        result = sci.run()
        # Should stop due to budget (PARAM_TUNING caps at min(50, budget_remaining))
        # With total_budget=10, first round gets 10 experiments max
        # but tree search creates seed + children, so allow some overhead
        assert result.total_experiments <= 30

    def test_early_stopping(self, tmp_dir):
        sci = Scientist(
            problem="Test",
            executor=self._make_executor(),
            score="score",
            baseline_config={"x": 3},  # already optimal
            max_rounds=10,
            total_budget=200,
            db_path=os.path.join(tmp_dir, "early.db"),
            journal_dir=os.path.join(tmp_dir, "journals"),
            memory_url=None,
            verbose=False,
        )

        # Always return inconclusive hypothesis
        mock_h = Hypothesis(
            id="h_inc", description="Inconclusive test",
            move_category=MoveCategory.PARAM_TUNING,
            expected_impact=0.1, confidence=0.3, effort="LOW",
            rationale="", falsifiable="",
        )
        sci._hypothesis_gen.generate = MagicMock(return_value=[mock_h])

        result = sci.run()
        # Should stop early due to consecutive non-improvements
        assert len(result.rounds) < 10

    def test_session_result_summary(self):
        result = SessionResult(
            session_id="test",
            problem="Test problem",
            best_score=0.85,
            duration_seconds=120.0,
        )
        result.rounds.append(RoundResult(
            round_number=1,
            hypothesis=Hypothesis(
                id="h1", description="Tune params",
                move_category=MoveCategory.PARAM_TUNING,
                expected_impact=0.5, confidence=0.5, effort="LOW",
                rationale="", falsifiable="",
            ),
            outcome="CONFIRMED",
            score_before=0.8,
            score_after=0.85,
            experiments_used=20,
        ))

        summary = result.summary()
        assert "test" in summary
        assert "0.85" in summary
        assert "CONFIRMED" not in summary  # uses [+] symbol
        assert "[+]" in summary

    def test_round_result_improvement(self):
        r = RoundResult(
            round_number=1,
            hypothesis=Hypothesis(
                id="h1", description="", move_category=MoveCategory.PARAM_TUNING,
                expected_impact=0, confidence=0, effort="LOW",
                rationale="", falsifiable="",
            ),
            outcome="CONFIRMED",
            score_before=0.3,
            score_after=0.35,
        )
        assert r.improvement == pytest.approx(0.05)

    def test_round_result_no_improvement(self):
        r = RoundResult(
            round_number=1,
            hypothesis=Hypothesis(
                id="h1", description="", move_category=MoveCategory.PARAM_TUNING,
                expected_impact=0, confidence=0, effort="LOW",
                rationale="", falsifiable="",
            ),
            outcome="INCONCLUSIVE",
            score_before=None,
            score_after=None,
        )
        assert r.improvement is None


# ── CLI ───────────────────────────────────────────────────────────────


class TestScientistCLI:
    def test_journal_command(self, tmp_path):
        from click.testing import CliRunner
        from arborist.scientist.cli import scientist_cli

        # Create a journal with some entries
        journal_dir = str(tmp_path / "journals")
        journal = Journal(session_id="cli_test", journal_dir=journal_dir, memory_url=None)
        journal.record_hypothesis("h_001", "Test hypothesis", "param_tuning", "reason")
        journal.record_result("h_001", "CONFIRMED", score_before=0.3, score_after=0.35)

        runner = CliRunner()
        result = runner.invoke(scientist_cli, [
            "journal", "--session", "cli_test",
            "--journal-dir", journal_dir,
        ])
        assert result.exit_code == 0
        assert "cli_test" in result.output
        assert "CONFIRMED" in result.output

    def test_journal_json_format(self, tmp_path):
        from click.testing import CliRunner
        from arborist.scientist.cli import scientist_cli

        journal_dir = str(tmp_path / "journals")
        journal = Journal(session_id="json_test", journal_dir=journal_dir, memory_url=None)
        journal.record_finding("Test finding")

        runner = CliRunner()
        result = runner.invoke(scientist_cli, [
            "journal", "--session", "json_test",
            "--journal-dir", journal_dir,
            "--format", "json",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1

    def test_hypotheses_command(self, tmp_path):
        from click.testing import CliRunner
        from arborist.scientist.cli import scientist_cli

        journal_dir = str(tmp_path / "journals")
        journal = Journal(session_id="hyp_test", journal_dir=journal_dir, memory_url=None)
        journal.record_hypothesis("h_001", "Tune LR", "param_tuning", "quick win")
        journal.record_result("h_001", "CONFIRMED")
        journal.record_hypothesis("h_002", "Add features", "feature_engineering", "missing signal")
        journal.record_result("h_002", "REJECTED")

        runner = CliRunner()
        result = runner.invoke(scientist_cli, [
            "hypotheses", "--session", "hyp_test",
            "--journal-dir", journal_dir,
        ])
        assert result.exit_code == 0
        assert "Tune LR" in result.output
        assert "Add features" in result.output

    def test_hypotheses_filter(self, tmp_path):
        from click.testing import CliRunner
        from arborist.scientist.cli import scientist_cli

        journal_dir = str(tmp_path / "journals")
        journal = Journal(session_id="filter_test", journal_dir=journal_dir, memory_url=None)
        journal.record_hypothesis("h_001", "Confirmed one", "param_tuning", "r")
        journal.record_result("h_001", "CONFIRMED")
        journal.record_hypothesis("h_002", "Rejected one", "param_tuning", "r")
        journal.record_result("h_002", "REJECTED")

        runner = CliRunner()
        result = runner.invoke(scientist_cli, [
            "hypotheses", "--session", "filter_test",
            "--journal-dir", journal_dir,
            "--status", "confirmed",
        ])
        assert result.exit_code == 0
        assert "Confirmed one" in result.output
        assert "Rejected one" not in result.output

    def test_empty_journal(self, tmp_path):
        from click.testing import CliRunner
        from arborist.scientist.cli import scientist_cli

        journal_dir = str(tmp_path / "journals")
        runner = CliRunner()
        result = runner.invoke(scientist_cli, [
            "journal", "--session", "nonexistent",
            "--journal-dir", journal_dir,
        ])
        assert result.exit_code == 0
        assert "No journal entries" in result.output


# ── Integration ───────────────────────────────────────────────────────


class TestObserverToHypothesisIntegration:
    """Test the flow from observation to hypothesis generation."""

    def test_observation_feeds_hypothesis(self):
        observer = Observer()
        results = {
            "classification_report": {
                "Wake": {"precision": 0.8, "recall": 0.3, "f1-score": 0.43, "support": 100},
                "N1": {"precision": 0.2, "recall": 0.1, "f1-score": 0.13, "support": 50},
                "N2": {"precision": 0.7, "recall": 0.9, "f1-score": 0.79, "support": 200},
            }
        }
        obs = observer.observe(
            results=results,
            experiment_count=50,
            best_score=0.45,
        )

        # Verify observation captures issues
        assert "N1" in obs.worst_classes
        assert len(obs.error_patterns) > 0

        # Feed to hypothesis generator (with mocked LLM)
        gen = HypothesisGenerator()
        with patch("litellm.completion", side_effect=Exception("no LLM")):
            hypotheses = gen.generate(obs, problem="Sleep staging")

        # Fallback hypotheses should reference worst class
        assert any("N1" in h.description for h in hypotheses)
