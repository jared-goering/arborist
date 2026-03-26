"""Tests for ScriptTemplate."""

from __future__ import annotations

import pytest

from arborist.executors.script_template import SECTION_NAMES, ScriptTemplate


class TestScriptTemplate:
    def test_available_task_types(self):
        types = ScriptTemplate.available_task_types()
        assert "classification" in types
        assert "regression" in types
        assert "time_series" in types

    def test_get_template_classification(self):
        template = ScriptTemplate.get_template_for_task("classification")
        assert "{imports}" in template
        assert "{data_loading}" in template
        assert "{feature_engineering}" in template
        assert "{model_definition}" in template
        assert "{training_loop}" in template
        assert "{evaluation}" in template
        assert "classification" in template

    def test_get_template_regression(self):
        template = ScriptTemplate.get_template_for_task("regression")
        assert "{imports}" in template
        assert "regression" in template

    def test_get_template_time_series(self):
        template = ScriptTemplate.get_template_for_task("time_series")
        assert "{imports}" in template
        assert "time series" in template

    def test_get_template_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown task type"):
            ScriptTemplate.get_template_for_task("unknown_type")

    def test_render_basic(self):
        sections = {
            "imports": "import pandas as pd",
            "data_loading": "df = pd.read_csv('data.csv')",
            "feature_engineering": "X = df.drop('y', axis=1)",
            "model_definition": "from sklearn.tree import DecisionTreeClassifier\nmodel = DecisionTreeClassifier()",
            "training_loop": "model.fit(X, y)",
            "evaluation": "print(f'val_f1={0.42}')",
        }
        script = ScriptTemplate.render(sections, task_type="classification")
        assert "import pandas as pd" in script
        assert "pd.read_csv" in script
        assert "DecisionTreeClassifier" in script
        assert "val_f1" in script
        # Template boilerplate should be present
        assert "SEED = 42" in script
        assert "import numpy as np" in script

    def test_render_missing_sections_get_pass(self):
        script = ScriptTemplate.render(
            {"imports": "import os"},
            task_type="classification",
        )
        assert "import os" in script
        # Missing sections should get 'pass'
        assert "pass" in script

    def test_render_all_task_types(self):
        sections = {name: f"# {name} code" for name in SECTION_NAMES}
        for task_type in ScriptTemplate.available_task_types():
            script = ScriptTemplate.render(sections, task_type=task_type)
            for name in SECTION_NAMES:
                assert f"# {name} code" in script

    def test_render_produces_valid_python(self):
        sections = {
            "imports": "import os",
            "data_loading": "data = None",
            "feature_engineering": "features = None",
            "model_definition": "model = None",
            "training_loop": "pass",
            "evaluation": "print('val_f1=0.0')",
        }
        script = ScriptTemplate.render(sections)
        # Should be valid Python
        compile(script, "<test>", "exec")

    def test_section_names_constant(self):
        assert len(SECTION_NAMES) == 6
        assert "imports" in SECTION_NAMES
        assert "evaluation" in SECTION_NAMES

    def test_template_has_seed_setting(self):
        for task_type in ScriptTemplate.available_task_types():
            template = ScriptTemplate.get_template_for_task(task_type)
            assert "SEED = 42" in template
            assert "random.seed" in template
            assert "np.random.seed" in template

    def test_import_from_executors(self):
        """ScriptTemplate is exported from arborist.executors."""
        from arborist.executors import ScriptTemplate as ST
        assert ST is ScriptTemplate
