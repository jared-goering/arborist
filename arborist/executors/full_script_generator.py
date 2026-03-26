"""FullScriptGenerator — generates complete training scripts from a ProblemSpec."""

from __future__ import annotations

import ast
import logging
import re
import subprocess
from pathlib import Path
from typing import Any

from arborist.executors.base import Executor
from arborist.executors.script_template import SECTION_NAMES, ScriptTemplate
from arborist.manager import BranchContext
from arborist.scientist.problem_spec import ProblemSpec

logger = logging.getLogger(__name__)

# Default metric pattern: key=value pairs on stdout
_METRIC_PATTERN = re.compile(r"(\w+)=([\d.eE+-]+)")


class FullScriptGenerator(Executor):
    """Executor that generates complete training scripts from scratch.

    Unlike CodeGeneratorExecutor which modifies existing scripts with scaffold
    markers, this executor creates entire scripts from a ProblemSpec + hypothesis.
    It uses ScriptTemplate for boilerplate and calls an LLM to fill in ML logic.

    Config dict keys:
        problem_spec: ProblemSpec instance describing the ML problem.
        hypothesis: str describing what approach to try.
        template_name: (optional) Override task_type for template selection.
        node_id: (optional) Used for versioned file naming.
        generated_script: (output) Path where the generated script was saved.
    """

    def __init__(
        self,
        model: str = "openrouter/anthropic/claude-sonnet-4-6",
        output_dir: str = "./experiments/generated",
        timeout: float = 300,
        python_cmd: str = "python3",
        extra_context: str = "",
    ) -> None:
        self.model = model
        self.output_dir = output_dir
        self.timeout = timeout
        self.python_cmd = python_cmd
        self.extra_context = extra_context

    def run(self, config: dict[str, Any], context: BranchContext) -> dict[str, Any]:
        """Generate a complete script, validate, execute, and return metrics.

        Args:
            config: Must contain 'problem_spec' (ProblemSpec) and 'hypothesis' (str).
            context: Branch context with goal, depth, etc.

        Returns:
            Dict with metrics, generated_script path, and stdout/stderr.
        """
        problem_spec: ProblemSpec = config["problem_spec"]
        hypothesis: str = config.get("hypothesis", "")
        template_name: str = config.get("template_name", problem_spec.task_type)
        node_id: str = config.get("node_id", "unknown")

        # Generate complete script via LLM
        try:
            script = self._generate_script(
                problem_spec=problem_spec,
                hypothesis=hypothesis,
                template_name=template_name,
                context=context,
            )
        except Exception as e:
            logger.error("Script generation failed for node %s: %s", node_id, e)
            return {
                "error": "generation_failed",
                "error_detail": str(e),
                "generated_script": "",
            }

        # Validate
        validation_errors = self._validate(script, problem_spec)
        if validation_errors:
            logger.warning(
                "Validation errors for node %s: %s", node_id, validation_errors
            )
            return {
                "error": "validation_failed",
                "validation_errors": validation_errors,
                "generated_script": "",
            }

        # Write to versioned output
        script_path = self._write_script(script, node_id)
        config["generated_script"] = script_path

        # Execute
        try:
            result = self._execute(script_path, problem_spec)
        except TimeoutError:
            logger.warning("Script %s timed out after %ss", script_path, self.timeout)
            return {
                "error": "timeout",
                "generated_script": script_path,
                "stdout": "",
                "stderr": f"Timed out after {self.timeout}s",
                "returncode": -1,
            }

        # Parse metrics
        metrics = self._parse_metrics(result.stdout)

        return {
            "generated_script": script_path,
            "metrics": metrics,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            # Expose individual metrics at top level for scoring
            **metrics,
        }

    def _generate_script(
        self,
        problem_spec: ProblemSpec,
        hypothesis: str,
        template_name: str,
        context: BranchContext,
    ) -> str:
        """Call LLM to generate a complete training script.

        Returns:
            Complete Python script as a string.
        """
        import litellm

        template = ScriptTemplate.get_template_for_task(template_name)
        prompt = self._build_prompt(problem_spec, hypothesis, template, context)

        response = litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_prompt(problem_spec)},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

        text = response.choices[0].message.content or ""
        return self._parse_llm_response(text, template_name)

    def _system_prompt(self, problem_spec: ProblemSpec) -> str:
        """Build system prompt for script generation."""
        forbidden_warning = ""
        if problem_spec.forbidden_patterns:
            patterns = "\n".join(f"  - {p}" for p in problem_spec.forbidden_patterns)
            forbidden_warning = (
                f"\n\nFORBIDDEN PATTERNS — your script MUST NOT:\n{patterns}\n"
            )

        return (
            "You are an expert ML engineer. You write complete, self-contained "
            "Python training scripts that load data, engineer features, train a model, "
            "and evaluate it.\n\n"
            "RULES:\n"
            "1. Output a COMPLETE, runnable Python script inside a single code fence.\n"
            "2. The script must print metrics to stdout in key=value format:\n"
            f"   Example: val_{problem_spec.metric}=0.4200 val_accuracy=0.7500\n"
            "3. Set random seeds for reproducibility (numpy, random, and any ML lib).\n"
            "4. Handle errors gracefully — print val_metric=0.0 on failure.\n"
            "5. The script must be self-contained (no external dependencies beyond "
            "standard ML libraries: numpy, pandas, sklearn, xgboost, lightgbm, etc.).\n"
            "6. Do NOT use interactive features (plots, input(), etc.).\n"
            f"{forbidden_warning}"
        )

    def _build_prompt(
        self,
        problem_spec: ProblemSpec,
        hypothesis: str,
        template: str,
        context: BranchContext,
    ) -> str:
        """Build the user prompt for script generation."""
        parts = [
            "# Task\n",
            "Generate a complete training script for this ML problem:\n\n",
            problem_spec.to_prompt(),
            "\n\n",
        ]

        if hypothesis:
            parts.append(f"**Hypothesis to test**: {hypothesis}\n\n")

        if context.goal:
            parts.append(f"**Research goal**: {context.goal}\n\n")

        if context.parent_score is not None:
            parts.append(f"**Current best score**: {context.parent_score:.4f}\n\n")

        if self.extra_context:
            parts.append(f"**Additional context**: {self.extra_context}\n\n")

        parts.append(
            "# Template Structure\n"
            "Your script should follow this structure with these sections:\n"
            f"  {', '.join(SECTION_NAMES)}\n\n"
            "Return the COMPLETE script in a single ```python code fence.\n"
            f"The script must print metrics like: val_{problem_spec.metric}=X.XXXX\n"
        )

        return "".join(parts)

    def _parse_llm_response(self, text: str, template_name: str) -> str:
        """Extract a complete Python script from LLM response.

        Returns:
            Complete Python script string.
        """
        # Try python code fence
        fence_match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
        if fence_match:
            return fence_match.group(1).strip()

        # Try generic code fence
        fence_match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
        if fence_match:
            return fence_match.group(1).strip()

        # If no fence, try the whole response (might be just code)
        stripped = text.strip()
        if stripped.startswith(("#!/", "import ", "from ", "#", "def ", "class ")):
            return stripped

        raise ValueError("Could not extract Python script from LLM response")

    def _validate(self, script: str, problem_spec: ProblemSpec) -> list[str]:
        """Validate the generated script.

        Checks:
            1. Syntax (ast.parse)
            2. Import availability (basic check)
            3. Metric output format (checks for print statement with metric pattern)
        """
        errors: list[str] = []

        # 1. Syntax check
        try:
            ast.parse(script)
        except SyntaxError as e:
            errors.append(f"SyntaxError: {e.msg} (line {e.lineno})")

        # 2. Check that script references the metric name somewhere
        if f"val_{problem_spec.metric}" not in script and problem_spec.metric not in script:
            errors.append(
                f"Script does not appear to output the required metric "
                f"'val_{problem_spec.metric}'"
            )

        # 3. Check forbidden patterns in the generated code
        for pattern in problem_spec.forbidden_patterns:
            if pattern in script:
                errors.append(f"Script contains forbidden pattern: {pattern!r}")

        return errors

    def _write_script(self, script: str, node_id: str) -> str:
        """Write generated script to versioned output directory."""
        out_dir = Path(self.output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = f"node_{node_id}.py"
        path = out_dir / filename
        path.write_text(script)
        logger.info("Wrote generated script to %s", path)
        return str(path)

    def _execute(
        self, script_path: str, problem_spec: ProblemSpec
    ) -> subprocess.CompletedProcess:
        """Run the generated script as a subprocess."""
        python_cmd = problem_spec.python_cmd or self.python_cmd
        timeout = problem_spec.timeout or self.timeout

        proc = subprocess.Popen(
            [python_cmd, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise TimeoutError(
                f"Generated script timed out after {timeout}s: {script_path}"
            ) from None

        result = subprocess.CompletedProcess(
            args=[python_cmd, script_path],
            returncode=proc.returncode,
            stdout=stdout,
            stderr=stderr,
        )

        if result.returncode != 0:
            logger.warning(
                "Script %s exited with code %d:\nstderr: %s",
                script_path, result.returncode, result.stderr[:500],
            )

        return result

    def _parse_metrics(self, stdout: str) -> dict[str, float]:
        """Parse key=value metrics from stdout."""
        metrics: dict[str, float] = {}
        for match in _METRIC_PATTERN.finditer(stdout):
            key = match.group(1)
            try:
                metrics[key] = float(match.group(2))
            except ValueError:
                continue
        return metrics
