"""Shell executor — runs commands as subprocesses, parses JSON from stdout."""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from typing import Any

from arborist.executors.base import Executor
from arborist.manager import BranchContext

logger = logging.getLogger(__name__)


class ShellExecutor(Executor):
    """Executes shell commands as experiments.

    The command template can use {key} placeholders that are filled from the config.
    A special {config_path} placeholder writes the full config to a temp JSON file.
    Results are parsed as JSON from stdout.
    """

    def __init__(
        self,
        command: str,
        timeout: float | None = 300,
        shell: bool = True,
    ) -> None:
        self.command = command
        self.timeout = timeout
        self.shell = shell

    def run(self, config: dict[str, Any], context: BranchContext) -> dict[str, Any]:
        # Write config to temp file for {config_path} placeholder
        config_path = None
        if "{config_path}" in self.command:
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, prefix="arborist_"
            )
            json.dump(config, tmp)
            tmp.close()
            config_path = tmp.name

        # Substitute placeholders using simple string replacement
        # (avoids str.format() which chokes on JSON braces in commands)
        fmt_kwargs = dict(config)
        if config_path:
            fmt_kwargs["config_path"] = config_path

        cmd = self.command
        for key, value in fmt_kwargs.items():
            cmd = cmd.replace("{" + key + "}", str(value))

        logger.debug("Running: %s", cmd)

        try:
            result = subprocess.run(
                cmd,
                shell=self.shell,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError(
                f"Shell command timed out after {self.timeout}s: {cmd}"
            ) from None

        if result.returncode != 0:
            raise RuntimeError(
                f"Command exited with code {result.returncode}.\n"
                f"stderr: {result.stderr.strip()}"
            )

        # Parse JSON from stdout
        stdout = result.stdout.strip()
        if not stdout:
            return {"stdout": "", "stderr": result.stderr.strip()}

        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            # If not valid JSON, return as raw output
            return {"stdout": stdout, "stderr": result.stderr.strip()}
