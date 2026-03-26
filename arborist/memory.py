"""Optional Ultramemory integration — HTTP client for localhost:8642."""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from typing import Any

logger = logging.getLogger(__name__)


class MemoryClient:
    """Client for optional Ultramemory integration.

    Sends experiment results and insights to a local Ultramemory instance
    for cross-search knowledge building.
    """

    def __init__(self, base_url: str = "http://localhost:8642") -> None:
        self.base_url = base_url.rstrip("/")
        self._available: bool | None = None

    @property
    def available(self) -> bool:
        """Check if Ultramemory is reachable."""
        if self._available is not None:
            return self._available
        try:
            req = urllib.request.Request(f"{self.base_url}/api/health", method="GET")
            urllib.request.urlopen(req, timeout=2)
            self._available = True
        except (urllib.error.URLError, OSError):
            self._available = False
            logger.debug("Ultramemory not available at %s", self.base_url)
        return self._available

    def ingest(self, content: str, metadata: dict[str, Any] | None = None) -> bool:
        """Post a fact to Ultramemory."""
        if not self.available:
            return False
        try:
            payload = json.dumps({"text": content, "source": "arborist", "metadata": metadata or {}}).encode()
            req = urllib.request.Request(
                f"{self.base_url}/api/ingest",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
            return True
        except (urllib.error.URLError, OSError) as e:
            logger.warning("Failed to ingest to Ultramemory: %s", e)
            return False

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search Ultramemory for related experiments."""
        if not self.available:
            return []
        try:
            payload = json.dumps({"query": query, "limit": limit}).encode()
            req = urllib.request.Request(
                f"{self.base_url}/api/search",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=5)
            data = json.loads(resp.read().decode())
            return data.get("results", [])
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to search Ultramemory: %s", e)
            return []

    def record_node_completion(
        self,
        tree_id: str,
        node_id: str,
        config: dict[str, Any],
        results: dict[str, Any],
        score: float | None,
        goal: str,
    ) -> None:
        """Record a completed experiment to memory."""
        content = (
            f"Experiment for goal: {goal}\n"
            f"Config: {json.dumps(config)}\n"
            f"Score: {score}\n"
            f"Results: {json.dumps(results)}"
        )
        self.ingest(content, metadata={
            "type": "experiment",
            "tree_id": tree_id,
            "node_id": node_id,
            "score": score,
        })

    def record_insight(
        self,
        tree_id: str,
        insight_type: str,
        content: str,
        goal: str,
    ) -> None:
        """Record an insight to memory."""
        self.ingest(
            f"Insight ({insight_type}) for goal: {goal}\n{content}",
            metadata={"type": "insight", "tree_id": tree_id, "insight_type": insight_type},
        )
