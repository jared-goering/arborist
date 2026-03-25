"""Python executor — runs callables in a thread pool."""

from __future__ import annotations

import logging
from typing import Any, Callable

from arborist.executors.base import Executor
from arborist.manager import BranchContext

logger = logging.getLogger(__name__)


class PythonExecutor(Executor):
    """Executes a Python callable as an experiment.

    The callable receives a config dict and returns a results dict.
    Optionally wraps with a timeout.
    """

    def __init__(
        self,
        fn: Callable[[dict[str, Any]], dict[str, Any]],
        timeout: float | None = None,
    ) -> None:
        self.fn = fn
        self.timeout = timeout

    def run(self, config: dict[str, Any], context: BranchContext) -> dict[str, Any]:
        """Run the callable with the given config.

        If a timeout is set, the callable is run in a thread and cancelled
        if it exceeds the timeout. Note: thread-based timeout cannot
        forcibly kill the callable — it relies on cooperative cancellation.
        """
        import concurrent.futures

        if self.timeout:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self.fn, config)
                try:
                    result = future.result(timeout=self.timeout)
                except concurrent.futures.TimeoutError:
                    raise TimeoutError(
                        f"Experiment timed out after {self.timeout}s"
                    ) from None
        else:
            result = self.fn(config)

        if not isinstance(result, dict):
            result = {"value": result}

        return result
