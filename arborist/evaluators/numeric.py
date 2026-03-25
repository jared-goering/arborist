"""Numeric evaluator — extracts a score from results by field name or callable."""

from __future__ import annotations

from typing import Any, Callable

from arborist.evaluators.base import Evaluator


class NumericEvaluator(Evaluator):
    """Evaluates experiments by extracting a numeric value from results.

    Can use either a field name (dot-notation supported) or a callable.

    Examples:
        NumericEvaluator(field="f1")            # results["f1"]
        NumericEvaluator(field="metrics.f1")    # results["metrics"]["f1"]
        NumericEvaluator(fn=lambda r: r["a"] + r["b"])
    """

    def __init__(
        self,
        field: str | None = None,
        fn: Callable[[dict[str, Any]], float] | None = None,
    ) -> None:
        if not field and not fn:
            raise ValueError("Must provide either 'field' or 'fn'")
        self.field = field
        self.fn = fn

    def evaluate(self, results: dict[str, Any], config: dict[str, Any]) -> float:
        if self.fn:
            return float(self.fn(results))

        # Dot-notation traversal
        value: Any = results
        for key in self.field.split("."):  # type: ignore[union-attr]
            if isinstance(value, dict):
                value = value[key]
            else:
                raise KeyError(
                    f"Cannot traverse into non-dict value at key '{key}' "
                    f"(got {type(value).__name__})"
                )
        return float(value)
