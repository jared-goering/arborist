"""Tests for CodeGeneratorExecutor, ScriptScaffold, and CodeGenMutator."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from arborist.executors.scaffold import ScriptScaffold, ScriptSection, SectionType
from arborist.executors.code_generator import CodeGeneratorExecutor
from arborist.manager import BranchContext
from arborist.mutators.code_gen_mutator import CodeGenMutator


# ── Test Fixtures ────────────────────────────────────────────────────


SAMPLE_SCRIPT = """\
import numpy as np
import pandas as pd

# --- SECTION: data_loading [FROZEN] ---
def load_data(path):
    return pd.read_csv(path)
# --- END SECTION ---

# --- SECTION: feature_engineering [MODIFIABLE] ---
def compute_features(df):
    df["mean_accel"] = df[["x", "y", "z"]].mean(axis=1)
    df["std_accel"] = df[["x", "y", "z"]].std(axis=1)
    return df
# --- END SECTION ---

# --- SECTION: model_training [FROZEN] ---
def train_model(X, y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier()
    model.fit(X, y)
    return model
# --- END SECTION ---

# --- SECTION: evaluation [FROZEN] ---
def evaluate(model, X_test, y_test):
    from sklearn.metrics import f1_score
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds, average="macro")
    print(f"val_f1={f1:.4f} val_accuracy=0.75 val_kappa=0.50")
# --- END SECTION ---
"""

MODIFIED_SCRIPT_GOOD = """\
import numpy as np
import pandas as pd

# --- SECTION: data_loading [FROZEN] ---
def load_data(path):
    return pd.read_csv(path)
# --- END SECTION ---

# --- SECTION: feature_engineering [MODIFIABLE] ---
def compute_features(df):
    df["mean_accel"] = df[["x", "y", "z"]].mean(axis=1)
    df["std_accel"] = df[["x", "y", "z"]].std(axis=1)
    df["range_accel"] = df[["x", "y", "z"]].max(axis=1) - df[["x", "y", "z"]].min(axis=1)
    df["magnitude"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
    return df
# --- END SECTION ---

# --- SECTION: model_training [FROZEN] ---
def train_model(X, y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier()
    model.fit(X, y)
    return model
# --- END SECTION ---

# --- SECTION: evaluation [FROZEN] ---
def evaluate(model, X_test, y_test):
    from sklearn.metrics import f1_score
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds, average="macro")
    print(f"val_f1={f1:.4f} val_accuracy=0.75 val_kappa=0.50")
# --- END SECTION ---
"""

BAD_SYNTAX_SCRIPT = """\
import numpy as np

def broken(
    # Missing closing paren
"""

SIMPLE_RUNNER = """\
#!/usr/bin/env python3
print("val_f1=0.4200 val_accuracy=0.7500 val_kappa=0.3100")
"""


def _make_context(**kwargs) -> BranchContext:
    defaults = dict(
        goal="Improve sleep staging F1",
        depth=1,
        parent_config={},
        parent_results={},
        parent_score=0.35,
        sibling_scores=[],
    )
    defaults.update(kwargs)
    return BranchContext(**defaults)


# ── ScriptScaffold Tests ─────────────────────────────────────────────


class TestScriptScaffold:
    def test_parse_sections(self):
        scaffold = ScriptScaffold.from_script(SAMPLE_SCRIPT)
        assert len(scaffold.sections) == 4
        assert scaffold.sections[0].name == "data_loading"
        assert scaffold.sections[0].section_type == SectionType.FROZEN
        assert scaffold.sections[1].name == "feature_engineering"
        assert scaffold.sections[1].section_type == SectionType.MODIFIABLE

    def test_preamble(self):
        scaffold = ScriptScaffold.from_script(SAMPLE_SCRIPT)
        assert "import numpy" in scaffold.preamble
        assert "import pandas" in scaffold.preamble

    def test_modifiable_sections(self):
        scaffold = ScriptScaffold.from_script(SAMPLE_SCRIPT)
        modifiable = scaffold.get_modifiable_sections()
        assert len(modifiable) == 1
        assert modifiable[0].name == "feature_engineering"
        assert "compute_features" in modifiable[0].content

    def test_frozen_sections(self):
        scaffold = ScriptScaffold.from_script(SAMPLE_SCRIPT)
        frozen = scaffold.get_frozen_sections()
        assert len(frozen) == 3
        frozen_names = {s.name for s in frozen}
        assert frozen_names == {"data_loading", "model_training", "evaluation"}

    def test_build_prompt_context(self):
        scaffold = ScriptScaffold.from_script(SAMPLE_SCRIPT)
        prompt = scaffold.build_prompt_context()
        assert "FROZEN — do not modify" in prompt
        assert "MODIFIABLE — you may modify this" in prompt
        assert "compute_features" in prompt

    def test_reassemble_preserves_frozen(self):
        scaffold = ScriptScaffold.from_script(SAMPLE_SCRIPT)
        new_feature_code = (
            "def compute_features(df):\n"
            "    df['mag'] = (df['x']**2 + df['y']**2 + df['z']**2)**0.5\n"
            "    return df"
        )
        result = scaffold.reassemble({"feature_engineering": new_feature_code})
        assert "load_data" in result  # frozen preserved
        assert "train_model" in result  # frozen preserved
        assert "mag" in result  # new feature present

    def test_verify_frozen_preserved_good(self):
        scaffold = ScriptScaffold.from_script(SAMPLE_SCRIPT)
        violations = scaffold.verify_frozen_preserved(MODIFIED_SCRIPT_GOOD)
        assert violations == []

    def test_verify_frozen_preserved_bad(self):
        scaffold = ScriptScaffold.from_script(SAMPLE_SCRIPT)
        # Remove the evaluation section entirely
        bad_script = MODIFIED_SCRIPT_GOOD.replace(
            "# --- SECTION: evaluation [FROZEN] ---",
            "# --- SECTION: evaluation [FROZEN] ---\n# TAMPERED"
        )
        # The content will differ
        new_scaffold = ScriptScaffold.from_script(bad_script)
        violations = scaffold.verify_frozen_preserved(bad_script)
        assert len(violations) > 0
        assert any("evaluation" in v for v in violations)

    def test_empty_script(self):
        scaffold = ScriptScaffold.from_script("")
        assert len(scaffold.sections) == 0

    def test_no_markers(self):
        plain = "x = 1\ny = 2\nprint(x + y)\n"
        scaffold = ScriptScaffold.from_script(plain)
        assert len(scaffold.sections) == 0
        assert "x = 1" in scaffold.preamble


# ── CodeGeneratorExecutor Tests ──────────────────────────────────────


class TestCodeGeneratorExecutor:
    def test_metric_parsing(self):
        executor = CodeGeneratorExecutor()
        metrics = executor._parse_metrics(
            "Training done.\nval_f1=0.3549 val_accuracy=0.6812 val_kappa=0.1923\n"
        )
        assert abs(metrics["val_f1"] - 0.3549) < 1e-6
        assert abs(metrics["val_accuracy"] - 0.6812) < 1e-6
        assert abs(metrics["val_kappa"] - 0.1923) < 1e-6

    def test_metric_parsing_empty(self):
        executor = CodeGeneratorExecutor()
        metrics = executor._parse_metrics("No metrics here\n")
        assert metrics == {}

    def test_metric_parsing_mixed_output(self):
        executor = CodeGeneratorExecutor()
        metrics = executor._parse_metrics(
            "Epoch 1/10\nloss=0.5432\nEpoch 10/10\nloss=0.1234\n"
            "val_f1=0.42 val_accuracy=0.80\n"
        )
        assert "val_f1" in metrics
        assert "val_accuracy" in metrics
        assert "loss" in metrics

    def test_validation_catches_syntax_error(self):
        executor = CodeGeneratorExecutor()
        scaffold = ScriptScaffold.from_script(SAMPLE_SCRIPT)
        errors = executor._validate(BAD_SYNTAX_SCRIPT, scaffold)
        assert any("SyntaxError" in e for e in errors)

    def test_validation_catches_frozen_violation(self):
        executor = CodeGeneratorExecutor()
        scaffold = ScriptScaffold.from_script(SAMPLE_SCRIPT)
        # Script with tampered frozen section
        tampered = SAMPLE_SCRIPT.replace(
            "def load_data(path):", "def load_data(path, extra=True):"
        )
        errors = executor._validate(tampered, scaffold)
        assert any("data_loading" in e for e in errors)

    def test_validation_passes_good_script(self):
        executor = CodeGeneratorExecutor()
        scaffold = ScriptScaffold.from_script(SAMPLE_SCRIPT)
        errors = executor._validate(MODIFIED_SCRIPT_GOOD, scaffold)
        assert errors == []

    def test_write_script_versioning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = CodeGeneratorExecutor(output_dir=tmpdir)
            path1 = executor._write_script("# script 1", "abc123")
            path2 = executor._write_script("# script 2", "def456")
            assert os.path.exists(path1)
            assert os.path.exists(path2)
            assert "node_abc123.py" in path1
            assert "node_def456.py" in path2
            assert Path(path1).read_text() == "# script 1"
            assert Path(path2).read_text() == "# script 2"

    def test_compute_diff(self):
        executor = CodeGeneratorExecutor()
        diff = executor._compute_diff(
            "line1\nline2\n", "line1\nline2_modified\n",
            "old.py", "new.py",
        )
        assert "---" in diff
        assert "+++" in diff
        assert "line2_modified" in diff

    def test_parse_llm_response_code_fence(self):
        executor = CodeGeneratorExecutor()
        scaffold = ScriptScaffold.from_script(SAMPLE_SCRIPT)
        response = f"Here's the modified script:\n\n```python\n{MODIFIED_SCRIPT_GOOD}\n```\n\nI added range and magnitude."
        result = executor._parse_llm_response(response, scaffold)
        assert "range_accel" in result
        assert "magnitude" in result

    def test_parse_llm_response_raw_code(self):
        executor = CodeGeneratorExecutor()
        scaffold = ScriptScaffold.from_script(SAMPLE_SCRIPT)
        result = executor._parse_llm_response(MODIFIED_SCRIPT_GOOD, scaffold)
        assert "compute_features" in result

    def test_timeout_handling(self):
        """Test that runaway scripts are killed after timeout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a script that sleeps forever
            script_path = os.path.join(tmpdir, "hang.py")
            Path(script_path).write_text("import time; time.sleep(9999)\n")

            executor = CodeGeneratorExecutor(timeout=1)
            with pytest.raises(TimeoutError):
                executor._execute(script_path)

    def test_full_pipeline(self):
        """Test the full pipeline: base script -> LLM modification -> execution -> results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a base script with section markers
            base_path = os.path.join(tmpdir, "base.py")
            base_content = (
                "# --- SECTION: imports [FROZEN] ---\n"
                "import sys\n"
                "# --- END SECTION ---\n"
                "\n"
                "# --- SECTION: compute [MODIFIABLE] ---\n"
                "result = 1 + 1\n"
                "# --- END SECTION ---\n"
                "\n"
                "# --- SECTION: output [FROZEN] ---\n"
                'print(f"val_f1={0.42} val_accuracy={0.75} val_kappa={0.31}")\n'
                "# --- END SECTION ---\n"
            )
            Path(base_path).write_text(base_content)

            # Mock the LLM to return a valid modified script
            modified_content = (
                "# --- SECTION: imports [FROZEN] ---\n"
                "import sys\n"
                "# --- END SECTION ---\n"
                "\n"
                "# --- SECTION: compute [MODIFIABLE] ---\n"
                "result = 2 + 2\n"
                "# --- END SECTION ---\n"
                "\n"
                "# --- SECTION: output [FROZEN] ---\n"
                'print(f"val_f1={0.42} val_accuracy={0.75} val_kappa={0.31}")\n'
                "# --- END SECTION ---\n"
            )

            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message=MagicMock(content=f"```python\n{modified_content}\n```"))
            ]

            executor = CodeGeneratorExecutor(
                output_dir=os.path.join(tmpdir, "generated"),
            )
            context = _make_context()
            config = {
                "base_script": base_path,
                "modifications": "Change computation to 2+2",
                "node_id": "test_001",
            }

            with patch("litellm.completion", return_value=mock_response):
                result = executor.run(config, context)

            assert result.get("error") is None
            assert result["val_f1"] == 0.42
            assert result["val_accuracy"] == 0.75
            assert os.path.exists(result["generated_script"])
            assert "node_test_001.py" in result["generated_script"]

    def test_validation_failure_returns_error(self):
        """Test that validation errors are returned without execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = os.path.join(tmpdir, "base.py")
            Path(base_path).write_text(SAMPLE_SCRIPT)

            # Mock LLM to return syntactically invalid code
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message=MagicMock(content="```python\ndef broken(\n```"))
            ]

            executor = CodeGeneratorExecutor(output_dir=os.path.join(tmpdir, "gen"))
            context = _make_context()
            config = {
                "base_script": base_path,
                "modifications": "Break the code",
                "node_id": "bad_001",
            }

            with patch("litellm.completion", return_value=mock_response):
                result = executor.run(config, context)

            assert result["error"] == "validation_failed"
            assert len(result["validation_errors"]) > 0


# ── CodeGenMutator Tests ─────────────────────────────────────────────


class TestCodeGenMutator:
    def test_fallback_produces_children(self):
        mutator = CodeGenMutator(base_script="/path/to/base.py")
        config = {
            "base_script": "/path/to/base.py",
            "modifications": "Add rolling mean features",
        }
        children = mutator._fallback(config)
        assert len(children) >= 1
        assert "base_script" in children[0]
        assert "modifications" in children[0]
        assert "Additionally" in children[0]["modifications"]

    def test_parse_proposals_json(self):
        mutator = CodeGenMutator()
        text = '["Add rolling std over 5min", "Add FFT features"]'
        proposals = mutator._parse_proposals(text)
        assert len(proposals) == 2
        assert "rolling std" in proposals[0]

    def test_parse_proposals_markdown(self):
        mutator = CodeGenMutator()
        text = '```json\n["proposal one", "proposal two"]\n```'
        proposals = mutator._parse_proposals(text)
        assert len(proposals) == 2

    def test_parse_proposals_numbered(self):
        mutator = CodeGenMutator()
        text = "1. Add rolling window statistics\n2. Add frequency domain features via FFT"
        proposals = mutator._parse_proposals(text)
        assert len(proposals) == 2

    def test_llm_propose_builds_children(self):
        """Test that LLM-based proposal returns valid child configs."""
        mutator = CodeGenMutator(
            base_script="/path/to/base.py",
            max_children=2,
        )

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='["Add rolling mean over 3,5,10 min", "Add jerk features"]'
                )
            )
        ]

        config = {
            "base_script": "/path/to/base.py",
            "modifications": "Add basic stats",
            "generated_script": "/path/to/gen.py",
        }
        results = {"metrics": {"val_f1": 0.35}, "diff": ""}
        context = _make_context()

        with patch("litellm.completion", return_value=mock_response):
            children = mutator(config, results, context)

        assert len(children) == 2
        for child in children:
            assert "base_script" in child
            assert "modifications" in child
            # Should use generated_script as new base
            assert child["base_script"] == "/path/to/gen.py"

    def test_llm_failure_uses_fallback(self):
        """Test that LLM failure gracefully falls back."""
        mutator = CodeGenMutator(base_script="/path/to/base.py")

        config = {
            "base_script": "/path/to/base.py",
            "modifications": "Add stats",
        }
        results = {}
        context = _make_context()

        with patch("litellm.completion", side_effect=RuntimeError("API down")):
            children = mutator(config, results, context)

        assert len(children) >= 1
        assert "base_script" in children[0]


# ── Integration: Moves executor_type ─────────────────────────────────


class TestMovesExecutorType:
    def test_feature_engineering_uses_code_generator(self):
        from arborist.scientist.moves import get_move, MoveCategory
        move = get_move(MoveCategory.FEATURE_ENGINEERING)
        assert move.executor_type == "code_generator"

    def test_architecture_uses_code_generator(self):
        from arborist.scientist.moves import get_move, MoveCategory
        move = get_move(MoveCategory.ARCHITECTURE)
        assert move.executor_type == "code_generator"

    def test_param_tuning_uses_default(self):
        from arborist.scientist.moves import get_move, MoveCategory
        move = get_move(MoveCategory.PARAM_TUNING)
        assert move.executor_type == "default"

    def test_ensemble_uses_default(self):
        from arborist.scientist.moves import get_move, MoveCategory
        move = get_move(MoveCategory.ENSEMBLE)
        assert move.executor_type == "default"
