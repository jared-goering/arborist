"""Tests for ProblemSpec."""

from __future__ import annotations

from arborist.scientist.problem_spec import ProblemSpec


class TestProblemSpec:
    def test_basic_creation(self):
        spec = ProblemSpec(
            name="iris",
            description="Classify iris species",
            dataset_path="/data/iris.csv",
            target_variable="species",
            metric="f1_macro",
            task_type="classification",
        )
        assert spec.name == "iris"
        assert spec.task_type == "classification"
        assert spec.metric == "f1_macro"

    def test_defaults(self):
        spec = ProblemSpec(
            name="test",
            description="Test problem",
            dataset_path="/data/test.csv",
            target_variable="y",
            metric="accuracy",
            task_type="classification",
        )
        assert spec.python_cmd == "python3"
        assert spec.timeout == 300
        assert spec.constraints == []
        assert spec.forbidden_patterns == []
        assert spec.extra_context == ""
        assert spec.data_description == ""

    def test_to_prompt_basic(self):
        spec = ProblemSpec(
            name="iris",
            description="Classify iris species",
            dataset_path="/data/iris.csv",
            target_variable="species",
            metric="f1_macro",
            task_type="classification",
        )
        prompt = spec.to_prompt()
        assert "iris" in prompt
        assert "Classify iris species" in prompt
        assert "/data/iris.csv" in prompt
        assert "species" in prompt
        assert "f1_macro" in prompt
        assert "classification" in prompt

    def test_to_prompt_with_constraints(self):
        spec = ProblemSpec(
            name="sleep",
            description="Sleep staging",
            dataset_path="/data/sleep.csv",
            target_variable="stage",
            metric="f1_macro",
            task_type="classification",
            constraints=["subject-grouped CV", "no label leakage"],
        )
        prompt = spec.to_prompt()
        assert "subject-grouped CV" in prompt
        assert "no label leakage" in prompt

    def test_to_prompt_with_forbidden_patterns(self):
        spec = ProblemSpec(
            name="test",
            description="Test",
            dataset_path="/data/test.csv",
            target_variable="y",
            metric="accuracy",
            task_type="classification",
            forbidden_patterns=["eval()", "exec()"],
        )
        prompt = spec.to_prompt()
        assert "eval()" in prompt
        assert "Forbidden" in prompt

    def test_to_prompt_with_data_description(self):
        spec = ProblemSpec(
            name="test",
            description="Test",
            dataset_path="/data/test.csv",
            target_variable="y",
            metric="accuracy",
            task_type="classification",
            data_description="4 features, 3 classes, 150 samples",
        )
        prompt = spec.to_prompt()
        assert "4 features" in prompt

    def test_to_prompt_with_extra_context(self):
        spec = ProblemSpec(
            name="test",
            description="Test",
            dataset_path="/data/test.csv",
            target_variable="y",
            metric="accuracy",
            task_type="classification",
            extra_context="Use XGBoost preferably",
        )
        prompt = spec.to_prompt()
        assert "XGBoost" in prompt

    def test_output_format_default(self):
        spec = ProblemSpec(
            name="test",
            description="Test",
            dataset_path="/data/test.csv",
            target_variable="y",
            metric="accuracy",
            task_type="classification",
        )
        assert "pattern" in spec.output_format
        assert "example" in spec.output_format

    def test_all_task_types(self):
        for task_type in ["classification", "regression", "time_series"]:
            spec = ProblemSpec(
                name="test",
                description="Test",
                dataset_path="/data/test.csv",
                target_variable="y",
                metric="rmse",
                task_type=task_type,
            )
            assert spec.task_type == task_type

    def test_import_from_scientist(self):
        """ProblemSpec is exported from arborist.scientist."""
        from arborist.scientist import ProblemSpec as PS
        assert PS is ProblemSpec
