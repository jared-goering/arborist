"""Tests for FullScriptGenerator."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from arborist.executors.full_script_generator import FullScriptGenerator
from arborist.manager import BranchContext
from arborist.scientist.problem_spec import ProblemSpec


# ── Test Fixtures ────────────────────────────────────────────────────


def _make_spec(**kwargs) -> ProblemSpec:
    defaults = dict(
        name="test_problem",
        description="A test classification problem",
        dataset_path="/data/test.csv",
        target_variable="y",
        metric="f1_macro",
        task_type="classification",
    )
    defaults.update(kwargs)
    return ProblemSpec(**defaults)


def _make_context(**kwargs) -> BranchContext:
    defaults = dict(
        goal="Test hypothesis",
        depth=1,
        parent_config={},
        parent_results={},
        parent_score=0.35,
        sibling_scores=[],
    )
    defaults.update(kwargs)
    return BranchContext(**defaults)


SIMPLE_VALID_SCRIPT = '''\
#!/usr/bin/env python3
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

data = load_iris()
X, y = data.data, data.target
model = DecisionTreeClassifier(random_state=42)
scores = cross_val_score(model, X, y, cv=3, scoring="f1_macro")
val_f1_macro = scores.mean()
print(f"val_f1_macro={val_f1_macro:.4f} val_accuracy=0.9500")
'''

SCRIPT_PRINTS_METRICS = '''\
#!/usr/bin/env python3
print("val_f1_macro=0.4200 val_accuracy=0.7500 val_kappa=0.3100")
'''


# ── FullScriptGenerator Tests ────────────────────────────────────────


class TestFullScriptGenerator:
    def test_metric_parsing(self):
        gen = FullScriptGenerator()
        metrics = gen._parse_metrics(
            "Training done.\nval_f1_macro=0.3549 val_accuracy=0.6812\n"
        )
        assert abs(metrics["val_f1_macro"] - 0.3549) < 1e-6
        assert abs(metrics["val_accuracy"] - 0.6812) < 1e-6

    def test_metric_parsing_empty(self):
        gen = FullScriptGenerator()
        metrics = gen._parse_metrics("No metrics here\n")
        assert metrics == {}

    def test_validate_good_script(self):
        gen = FullScriptGenerator()
        spec = _make_spec()
        errors = gen._validate(SIMPLE_VALID_SCRIPT, spec)
        assert errors == []

    def test_validate_catches_syntax_error(self):
        gen = FullScriptGenerator()
        spec = _make_spec()
        bad_script = "def broken(\n  # missing closing paren\n"
        errors = gen._validate(bad_script, spec)
        assert any("SyntaxError" in e for e in errors)

    def test_validate_catches_missing_metric(self):
        gen = FullScriptGenerator()
        spec = _make_spec(metric="f1_macro")
        script = "x = 1\nprint('done')\n"
        errors = gen._validate(script, spec)
        assert any("val_f1_macro" in e for e in errors)

    def test_validate_catches_forbidden_pattern(self):
        gen = FullScriptGenerator()
        spec = _make_spec(forbidden_patterns=["eval("])
        script = "result = eval('1+1')\nprint(f'val_f1_macro={result}')\n"
        errors = gen._validate(script, spec)
        assert any("forbidden" in e.lower() for e in errors)

    def test_write_script_versioning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = FullScriptGenerator(output_dir=tmpdir)
            path1 = gen._write_script("# script 1", "abc123")
            path2 = gen._write_script("# script 2", "def456")
            assert os.path.exists(path1)
            assert os.path.exists(path2)
            assert "node_abc123.py" in path1
            assert "node_def456.py" in path2
            assert Path(path1).read_text() == "# script 1"
            assert Path(path2).read_text() == "# script 2"

    def test_parse_llm_response_code_fence(self):
        gen = FullScriptGenerator()
        response = f"Here's the script:\n\n```python\n{SIMPLE_VALID_SCRIPT}\n```\n\nDone."
        result = gen._parse_llm_response(response, "classification")
        assert "DecisionTreeClassifier" in result
        assert "val_f1_macro" in result

    def test_parse_llm_response_generic_fence(self):
        gen = FullScriptGenerator()
        response = f"```\n{SIMPLE_VALID_SCRIPT}\n```"
        result = gen._parse_llm_response(response, "classification")
        assert "DecisionTreeClassifier" in result

    def test_parse_llm_response_raw_code(self):
        gen = FullScriptGenerator()
        result = gen._parse_llm_response(SIMPLE_VALID_SCRIPT, "classification")
        assert "DecisionTreeClassifier" in result

    def test_parse_llm_response_no_code_raises(self):
        gen = FullScriptGenerator()
        with pytest.raises(ValueError, match="Could not extract"):
            gen._parse_llm_response("Just some text without code.", "classification")

    def test_system_prompt_includes_metric(self):
        gen = FullScriptGenerator()
        spec = _make_spec(metric="rmse")
        prompt = gen._system_prompt(spec)
        assert "val_rmse" in prompt

    def test_system_prompt_includes_forbidden(self):
        gen = FullScriptGenerator()
        spec = _make_spec(forbidden_patterns=["use eval()", "access /etc/passwd"])
        prompt = gen._system_prompt(spec)
        assert "eval()" in prompt
        assert "FORBIDDEN" in prompt

    def test_build_prompt_includes_problem(self):
        gen = FullScriptGenerator()
        spec = _make_spec(description="Classify iris species")
        context = _make_context(goal="Improve F1", parent_score=0.5)
        template = "# template"
        prompt = gen._build_prompt(spec, "Use random forest", template, context)
        assert "Classify iris species" in prompt
        assert "random forest" in prompt
        assert "Improve F1" in prompt
        assert "0.5000" in prompt

    def test_timeout_handling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "hang.py")
            Path(script_path).write_text("import time; time.sleep(9999)\n")

            gen = FullScriptGenerator(timeout=1)
            spec = _make_spec(timeout=1)
            with pytest.raises(TimeoutError):
                gen._execute(script_path, spec)

    def test_full_pipeline(self):
        """Test full pipeline: problem_spec -> LLM generation -> execution -> results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(
                    message=MagicMock(
                        content=f"```python\n{SCRIPT_PRINTS_METRICS}\n```"
                    )
                )
            ]

            gen = FullScriptGenerator(
                output_dir=os.path.join(tmpdir, "generated"),
            )
            spec = _make_spec(metric="f1_macro")
            context = _make_context()
            config = {
                "problem_spec": spec,
                "hypothesis": "Try a decision tree",
                "node_id": "test_001",
            }

            with patch("litellm.completion", return_value=mock_response):
                result = gen.run(config, context)

            assert result.get("error") is None
            assert result["val_f1_macro"] == 0.42
            assert result["val_accuracy"] == 0.75
            assert os.path.exists(result["generated_script"])
            assert "node_test_001.py" in result["generated_script"]

    def test_validation_failure_returns_error(self):
        """Test that validation errors are returned without execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(
                    message=MagicMock(content="```python\ndef broken(\n```")
                )
            ]

            gen = FullScriptGenerator(
                output_dir=os.path.join(tmpdir, "gen"),
            )
            spec = _make_spec()
            context = _make_context()
            config = {
                "problem_spec": spec,
                "hypothesis": "Break the code",
                "node_id": "bad_001",
            }

            with patch("litellm.completion", return_value=mock_response):
                result = gen.run(config, context)

            assert result["error"] == "validation_failed"
            assert len(result["validation_errors"]) > 0

    def test_generation_failure_returns_error(self):
        """Test that LLM failure returns error dict, doesn't crash."""
        gen = FullScriptGenerator()
        spec = _make_spec()
        context = _make_context()
        config = {
            "problem_spec": spec,
            "hypothesis": "Test hypothesis",
            "node_id": "err_001",
        }

        with patch("litellm.completion", side_effect=RuntimeError("API down")):
            result = gen.run(config, context)

        assert result["error"] == "generation_failed"
        assert "API down" in result["error_detail"]

    def test_timeout_returns_error(self):
        """Test that timeout returns error dict, doesn't crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_response = MagicMock()
            # Script that will hang
            hang_script = "import time\ntime.sleep(9999)\nprint('val_f1_macro=0.0')\n"
            mock_response.choices = [
                MagicMock(
                    message=MagicMock(
                        content=f"```python\n{hang_script}\n```"
                    )
                )
            ]

            gen = FullScriptGenerator(
                output_dir=os.path.join(tmpdir, "gen"),
                timeout=1,
            )
            spec = _make_spec(timeout=1)
            context = _make_context()
            config = {
                "problem_spec": spec,
                "hypothesis": "Test timeout",
                "node_id": "timeout_001",
            }

            with patch("litellm.completion", return_value=mock_response):
                result = gen.run(config, context)

            assert result["error"] == "timeout"

    def test_import_from_executors(self):
        """FullScriptGenerator is exported from arborist.executors."""
        from arborist.executors import FullScriptGenerator as FSG
        assert FSG is FullScriptGenerator


# ── Integration: Scientist with problem_spec ─────────────────────────


class TestScientistProblemSpecIntegration:
    def test_scientist_accepts_problem_spec(self):
        """Scientist accepts problem_spec and codegen_model params."""
        from arborist.scientist import Scientist

        spec = _make_spec()
        scientist = Scientist(
            problem="Test problem",
            problem_spec=spec,
            codegen_model="test-model",
            executor=lambda config: {},
            score=lambda results: 0.0,
            max_rounds=1,
            total_budget=5,
            verbose=False,
        )
        assert scientist.problem_spec is spec
        assert scientist.codegen_model == "test-model"

    def test_scientist_codegen_model_defaults_to_none(self):
        """codegen_model defaults to None."""
        from arborist.scientist import Scientist

        scientist = Scientist(
            problem="Test",
            executor=lambda config: {},
            score=lambda results: 0.0,
            verbose=False,
        )
        assert scientist.codegen_model is None
        assert scientist.problem_spec is None

    def test_full_gen_mutator_with_problem_spec(self):
        """CodeGenMutator in full-gen mode produces problem_spec children."""
        from arborist.mutators import CodeGenMutator

        spec = _make_spec()
        mutator = CodeGenMutator(
            base_script=None,
            problem_spec=spec,
        )

        config = {
            "problem_spec": spec,
            "hypothesis": "Try random forest",
        }
        results = {}
        context = _make_context()

        # Use fallback (no LLM call)
        children = mutator._fallback(config)
        assert len(children) >= 1
        assert "problem_spec" in children[0]
        assert "hypothesis" in children[0]
        assert "Additionally" in children[0]["hypothesis"]
        # Should NOT have base_script
        assert "base_script" not in children[0]

    def test_full_gen_mutator_llm_produces_correct_children(self):
        """CodeGenMutator LLM path produces problem_spec children for full-gen."""
        from arborist.mutators import CodeGenMutator

        spec = _make_spec()
        mutator = CodeGenMutator(
            base_script=None,
            problem_spec=spec,
            max_children=2,
        )

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='["Use gradient boosting with tuned lr", "Try SVM with RBF kernel"]'
                )
            )
        ]

        config = {
            "problem_spec": spec,
            "hypothesis": "Try random forest",
        }
        results = {"metrics": {"val_f1_macro": 0.35}}
        context = _make_context()

        with patch("litellm.completion", return_value=mock_response):
            children = mutator(config, results, context)

        assert len(children) == 2
        for child in children:
            assert "problem_spec" in child
            assert "hypothesis" in child
            assert child["problem_spec"] is spec
