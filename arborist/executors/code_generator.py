"""CodeGeneratorExecutor — LLM-powered code modification and execution."""

from __future__ import annotations

import ast
import difflib
import json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arborist.executors.base import Executor
from arborist.executors.scaffold import ScriptScaffold, SectionType
from arborist.manager import BranchContext

logger = logging.getLogger(__name__)

# Default metric pattern: key=value pairs on stdout
_METRIC_PATTERN = re.compile(r"(\w+)=([\d.eE+-]+)")


@dataclass
class CodeGenResult:
    """Result of a code generation + execution cycle."""

    generated_script: str
    script_path: str
    diff: str
    metrics: dict[str, float]
    stdout: str
    stderr: str
    returncode: int
    validation_errors: list[str] = field(default_factory=list)


class CodeGeneratorExecutor(Executor):
    """Executor that uses an LLM to modify a base script and run the result.

    Instead of mutating numeric hyperparameters, this executor takes a base
    Python script with scaffold section markers and asks an LLM to rewrite
    the MODIFIABLE sections according to natural-language instructions.

    The modified script is validated (syntax, frozen-section integrity),
    written to a versioned output directory, and executed as a subprocess.
    Metrics are parsed from stdout.

    Config dict keys:
        base_script: Path to the base Python script (with section markers).
        modifications: Natural-language instructions for what to change.
        generated_script: (output) Path where the generated script was saved.
        node_id: (optional) Used for versioned file naming.
    """

    def __init__(
        self,
        model: str = "openrouter/anthropic/claude-sonnet-4-5",
        output_dir: str = "./experiments/generated",
        timeout: float = 300,
        python_cmd: str = "python3",
        metric_names: list[str] | None = None,
        extra_context: str = "",
        cwd: str | None = None,
    ) -> None:
        self.model = model
        self.output_dir = output_dir
        self.timeout = timeout
        self.python_cmd = python_cmd
        self.metric_names = metric_names or ["val_f1", "val_accuracy", "val_kappa"]
        self.extra_context = extra_context
        self.cwd = cwd  # Working directory for script execution; defaults to base_script's dir

    def run(self, config: dict[str, Any], context: BranchContext) -> dict[str, Any]:
        """Generate modified script, validate, execute, and return metrics.

        Args:
            config: Must contain 'base_script' (path) and 'modifications' (str).
            context: Branch context with goal, depth, etc.

        Returns:
            Dict with metrics, generated_script path, diff, and stdout/stderr.
        """
        base_script_path = config["base_script"]
        modifications = config.get("modifications", "")
        node_id = config.get("node_id", "unknown")

        # Read base script
        base_script = Path(base_script_path).read_text()
        scaffold = ScriptScaffold.from_script(base_script)

        # Set CWD to base script's directory if not explicitly configured
        if self.cwd is None:
            self._run_cwd = str(Path(base_script_path).parent.resolve())
        else:
            self._run_cwd = self.cwd

        # Generate modified script via LLM
        modified_script, modified_sections = self._generate_code(
            scaffold=scaffold,
            modifications=modifications,
            context=context,
        )

        # Validate (includes label leakage check via modified_sections)
        validation_errors = self._validate(modified_script, scaffold, modified_sections)
        if validation_errors:
            logger.warning(
                "Validation errors for node %s: %s", node_id, validation_errors
            )
            return {
                "error": "validation_failed",
                "validation_errors": validation_errors,
                "generated_script": "",
                "diff": "",
            }

        # Inject PROJECT_ROOT override so generated scripts find data
        # regardless of where they're saved. Replace the first occurrence of
        # PROJECT_ROOT = Path(__file__).parent with the actual base script dir.
        base_dir = str(Path(base_script_path).parent.resolve())
        modified_script = re.sub(
            r'PROJECT_ROOT\s*=\s*Path\(__file__\)\.parent',
            f'PROJECT_ROOT = Path({base_dir!r})',
            modified_script,
            count=1,  # Only replace the first occurrence
        )

        # Write to versioned output
        script_path = self._write_script(modified_script, node_id)
        config["generated_script"] = script_path

        # Compute diff
        diff = self._compute_diff(base_script, modified_script, base_script_path, script_path)

        # Execute
        result = self._execute(script_path)

        # Parse metrics
        metrics = self._parse_metrics(result.stdout)

        return {
            "generated_script": script_path,
            "diff": diff,
            "metrics": metrics,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            # Expose individual metrics at top level for scoring
            **metrics,
        }

    def _generate_code(
        self,
        scaffold: ScriptScaffold,
        modifications: str,
        context: BranchContext,
    ) -> tuple[str, dict[str, str]]:
        """Call LLM to generate modified script sections.

        Returns:
            Tuple of (assembled_script, modified_sections_dict).
            The dict maps section names to their new content for validation.
        """
        import litellm

        modifiable = scaffold.get_modifiable_sections()
        if not modifiable:
            raise ValueError("No MODIFIABLE sections found in base script")

        prompt = self._build_prompt(scaffold, modifications, context)

        response = litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_prompt(scaffold)},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

        text = response.choices[0].message.content or ""
        return self._parse_llm_response(text, scaffold)

    def _system_prompt(self, scaffold: ScriptScaffold | None = None) -> str:
        # Collect all forbidden vars across sections for the global warning
        all_forbidden: set[str] = set()
        if scaffold:
            for sec in scaffold.get_modifiable_sections():
                all_forbidden.update(sec.forbidden_vars)

        forbidden_warning = ""
        if all_forbidden:
            vars_list = ", ".join(sorted(all_forbidden))
            forbidden_warning = (
                f"\n\n⚠️ CRITICAL — LABEL LEAKAGE PREVENTION ⚠️\n"
                f"The following variables contain ground truth labels and MUST NOT "
                f"be referenced in feature engineering code: {vars_list}\n"
                f"Any code that reads, indexes, iterates, or derives features from "
                f"these variables will be REJECTED. Features must be computed from "
                f"X (input features), subjects, and feature_names ONLY.\n"
                f"Specifically forbidden patterns:\n"
                f"- Using y/y_train/y_val to compute class counts, transitions, or durations\n"
                f"- Creating 'time since last class X' features (requires labels)\n"
                f"- Using predicted probabilities from a prior model as features (two-stage leakage)\n"
                f"- Any feature that encodes which sleep stage an epoch belongs to\n"
            )

        return (
            "You are an expert ML engineer. You modify Python training scripts "
            "to implement feature engineering and architecture changes.\n\n"
            "RULES:\n"
            "1. You will be shown a script with FROZEN and MODIFIABLE sections.\n"
            "2. Return ONLY the MODIFIABLE section content you want to change.\n"
            "3. For each section you modify, wrap it in a labeled code fence:\n"
            "   ```python section:Feature Computation\n"
            "   ...your modified code...\n"
            "   ```\n"
            "4. The script must remain syntactically valid Python.\n"
            "5. All imports needed by your changes must be added at the top of "
            "the modifiable section.\n"
            "6. Do NOT change the evaluation metrics output format.\n"
            "7. Preserve subject-grouped cross-validation — never leak across subjects.\n"
            "8. Do NOT return FROZEN sections — they are automatically preserved.\n"
            f"{forbidden_warning}"
        )

    def _build_prompt(
        self,
        scaffold: ScriptScaffold,
        modifications: str,
        context: BranchContext,
    ) -> str:
        parts = [
            "# Task\n",
            f"Modify the following training script according to these instructions:\n\n"
            f"**Instructions**: {modifications}\n",
        ]

        if context.goal:
            parts.append(f"\n**Research goal**: {context.goal}\n")

        if context.parent_score is not None:
            parts.append(f"\n**Current best score**: {context.parent_score:.4f}\n")

        if self.extra_context:
            parts.append(f"\n**Additional context**: {self.extra_context}\n")

        parts.append("\n# Current Script (with section markers)\n\n```python\n")
        parts.append(scaffold.build_prompt_context())
        parts.append("\n```\n")

        modifiable_names = [s.name for s in scaffold.get_modifiable_sections()]
        parts.append(
            "\n# Instructions\n"
            f"Modifiable sections: {', '.join(modifiable_names)}\n\n"
            "Return ONLY the modified section contents, each in a labeled code fence:\n"
            "```python section:SectionName\n"
            "...code...\n"
            "```\n\n"
            "If a modifiable section doesn't need changes, omit it (the original is kept).\n"
            "Do NOT include FROZEN sections or section markers in your output.\n"
        )

        return "".join(parts)

    def _parse_llm_response(
        self, text: str, scaffold: ScriptScaffold
    ) -> tuple[str, dict[str, str]]:
        """Extract modified sections from LLM response and reassemble full script.

        Returns:
            Tuple of (assembled_script, modified_sections_dict).
        """
        modified_sections: dict[str, str] = {}

        # Try labeled fences: ```python section:SectionName
        labeled_pattern = re.compile(
            r"```python\s+section:([^\n]+)\s*\n(.*?)```", re.DOTALL
        )
        for match in labeled_pattern.finditer(text):
            section_name = match.group(1).strip()
            section_code = match.group(2).rstrip()
            modified_sections[section_name] = section_code

        # If we got labeled sections, reassemble
        if modified_sections:
            return scaffold.reassemble(modified_sections), modified_sections

        # Fallback: try a single code fence (old behavior for backward compat)
        fence_match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
        if fence_match:
            code = fence_match.group(1).strip()
            # Check if it's a full script (has section markers)
            if "# --- SECTION:" in code:
                return code, {}
            # Otherwise assume it's a single modifiable section replacement
            modifiable = scaffold.get_modifiable_sections()
            if len(modifiable) == 1:
                modified_sections = {modifiable[0].name: code}
                return scaffold.reassemble(modified_sections), modified_sections

        # Try generic code fence
        fence_match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
        if fence_match:
            return fence_match.group(1).strip(), {}

        # If no fence, try the whole response (it might be just code)
        stripped = text.strip()
        if stripped.startswith(("import ", "from ", "#", "def ", "class ")):
            return stripped, {}

        raise ValueError("Could not extract Python script from LLM response")

    def _validate(
        self,
        script: str,
        original_scaffold: ScriptScaffold,
        modified_sections: dict[str, str] | None = None,
    ) -> list[str]:
        """Validate generated script: syntax + frozen sections + label leakage."""
        errors = []

        # Syntax check
        try:
            ast.parse(script)
        except SyntaxError as e:
            errors.append(f"SyntaxError: {e.msg} (line {e.lineno})")

        # Frozen section check
        violations = original_scaffold.verify_frozen_preserved(script)
        errors.extend(violations)

        # Label leakage check (forbidden variable access)
        if modified_sections:
            leakage = original_scaffold.check_forbidden_vars(modified_sections)
            errors.extend(leakage)

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

    def _compute_diff(
        self, old: str, new: str, old_path: str, new_path: str
    ) -> str:
        """Compute unified diff between base and modified scripts."""
        old_lines = old.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)
        diff = difflib.unified_diff(
            old_lines, new_lines,
            fromfile=old_path, tofile=new_path,
        )
        return "".join(diff)

    def _execute(self, script_path: str) -> subprocess.CompletedProcess:
        """Run the generated script as a subprocess."""
        run_cwd = getattr(self, "_run_cwd", None)
        try:
            result = subprocess.run(
                [self.python_cmd, script_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=run_cwd,
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError(
                f"Generated script timed out after {self.timeout}s: {script_path}"
            ) from None

        if result.returncode != 0:
            logger.warning(
                "Script %s exited with code %d:\nstderr: %s",
                script_path, result.returncode, result.stderr[:500],
            )

        return result

    def _parse_metrics(self, stdout: str) -> dict[str, float]:
        """Parse key=value metrics from stdout.

        Looks for patterns like: val_f1=0.3549 val_accuracy=0.6812 val_kappa=0.1923
        """
        metrics: dict[str, float] = {}
        for match in _METRIC_PATTERN.finditer(stdout):
            key = match.group(1)
            try:
                metrics[key] = float(match.group(2))
            except ValueError:
                continue

        # Warn if expected metrics are missing
        for name in self.metric_names:
            if name not in metrics:
                logger.warning("Expected metric '%s' not found in stdout", name)

        return metrics
