"""ScriptTemplate — boilerplate templates for full script generation."""

from __future__ import annotations

from typing import Any


# ── Base template with placeholders ──────────────────────────────────

_CLASSIFICATION_TEMPLATE = '''\
#!/usr/bin/env python3
"""Auto-generated training script for classification."""

import argparse
import os
import random
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# ── Reproducibility ──────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Imports ──────────────────────────────────────────────────────────
{imports}

# ── Data Loading ─────────────────────────────────────────────────────
{data_loading}

# ── Feature Engineering ──────────────────────────────────────────────
{feature_engineering}

# ── Model Definition ─────────────────────────────────────────────────
{model_definition}

# ── Training Loop ────────────────────────────────────────────────────
{training_loop}

# ── Evaluation ───────────────────────────────────────────────────────
{evaluation}
'''

_REGRESSION_TEMPLATE = '''\
#!/usr/bin/env python3
"""Auto-generated training script for regression."""

import argparse
import os
import random
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# ── Reproducibility ──────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Imports ──────────────────────────────────────────────────────────
{imports}

# ── Data Loading ─────────────────────────────────────────────────────
{data_loading}

# ── Feature Engineering ──────────────────────────────────────────────
{feature_engineering}

# ── Model Definition ─────────────────────────────────────────────────
{model_definition}

# ── Training Loop ────────────────────────────────────────────────────
{training_loop}

# ── Evaluation ───────────────────────────────────────────────────────
{evaluation}
'''

_TIME_SERIES_TEMPLATE = '''\
#!/usr/bin/env python3
"""Auto-generated training script for time series."""

import argparse
import os
import random
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# ── Reproducibility ──────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Imports ──────────────────────────────────────────────────────────
{imports}

# ── Data Loading ─────────────────────────────────────────────────────
{data_loading}

# ── Feature Engineering ──────────────────────────────────────────────
{feature_engineering}

# ── Model Definition ─────────────────────────────────────────────────
{model_definition}

# ── Training Loop ────────────────────────────────────────────────────
{training_loop}

# ── Evaluation ───────────────────────────────────────────────────────
{evaluation}
'''

_TEMPLATES: dict[str, str] = {
    "classification": _CLASSIFICATION_TEMPLATE,
    "regression": _REGRESSION_TEMPLATE,
    "time_series": _TIME_SERIES_TEMPLATE,
}

SECTION_NAMES = [
    "imports",
    "data_loading",
    "feature_engineering",
    "model_definition",
    "training_loop",
    "evaluation",
]


class ScriptTemplate:
    """Provides boilerplate templates so the LLM only fills in ML-specific logic.

    Each template has placeholders for: {imports}, {data_loading},
    {feature_engineering}, {model_definition}, {training_loop}, {evaluation}.
    """

    @staticmethod
    def get_template_for_task(task_type: str) -> str:
        """Return the template string for a given task type.

        Args:
            task_type: One of 'classification', 'regression', 'time_series'.

        Returns:
            Template string with {section} placeholders.

        Raises:
            ValueError: If task_type is not recognized.
        """
        template = _TEMPLATES.get(task_type)
        if template is None:
            raise ValueError(
                f"Unknown task type: {task_type!r}. "
                f"Available: {list(_TEMPLATES.keys())}"
            )
        return template

    @staticmethod
    def render(sections: dict[str, str], task_type: str = "classification") -> str:
        """Render a complete script from section contents.

        Args:
            sections: Mapping of section name -> code content.
            task_type: Template to use ('classification', 'regression', 'time_series').

        Returns:
            Complete Python script as a string.

        Raises:
            ValueError: If required sections are missing.
        """
        template = ScriptTemplate.get_template_for_task(task_type)

        # Fill in provided sections, default to 'pass' for missing ones
        filled: dict[str, str] = {}
        for name in SECTION_NAMES:
            filled[name] = sections.get(name, "pass")

        return template.format(**filled)

    @staticmethod
    def available_task_types() -> list[str]:
        """Return list of available task types."""
        return list(_TEMPLATES.keys())
