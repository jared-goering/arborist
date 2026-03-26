"""Research journal — persistent experiment history with Ultramemory + file fallback."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from arborist.memory import MemoryClient

logger = logging.getLogger(__name__)


@dataclass
class JournalEntry:
    """A single entry in the research journal."""

    timestamp: str
    entry_type: str  # hypothesis, result, finding, dead_end, pattern
    hypothesis_id: str | None = None
    content: str = ""
    outcome: str | None = None  # CONFIRMED, REJECTED, INCONCLUSIVE
    score_before: float | None = None
    score_after: float | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "entry_type": self.entry_type,
            "hypothesis_id": self.hypothesis_id,
            "content": self.content,
            "outcome": self.outcome,
            "score_before": self.score_before,
            "score_after": self.score_after,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JournalEntry:
        return cls(
            timestamp=data.get("timestamp", ""),
            entry_type=data.get("entry_type", ""),
            hypothesis_id=data.get("hypothesis_id"),
            content=data.get("content", ""),
            outcome=data.get("outcome"),
            score_before=data.get("score_before"),
            score_after=data.get("score_after"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


class Journal:
    """Research journal with Ultramemory integration and file-based fallback.

    Stores experiment history, findings, and dead ends to inform
    future hypothesis generation.
    """

    def __init__(
        self,
        session_id: str,
        journal_dir: str = "./scientist_journals",
        memory_url: str | None = "http://localhost:8642",
    ) -> None:
        self.session_id = session_id
        self.journal_dir = journal_dir
        self._entries: list[JournalEntry] = []

        # Try Ultramemory first
        self._memory: MemoryClient | None = None
        if memory_url:
            client = MemoryClient(memory_url)
            if client.available:
                self._memory = client
                logger.info("Journal connected to Ultramemory at %s", memory_url)
            else:
                logger.debug("Ultramemory not available, using file fallback")

        # File-based fallback (always active for local persistence)
        self._journal_path = os.path.join(journal_dir, f"{session_id}.jsonl")
        self._ensure_dir()
        self._load_from_file()

    def _ensure_dir(self) -> None:
        os.makedirs(self.journal_dir, exist_ok=True)

    def _load_from_file(self) -> None:
        """Load existing entries from JSONL file."""
        if not os.path.exists(self._journal_path):
            return
        try:
            with open(self._journal_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        self._entries.append(JournalEntry.from_dict(data))
            logger.debug("Loaded %d journal entries from %s", len(self._entries), self._journal_path)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Error loading journal file: %s", e)

    def _append_to_file(self, entry: JournalEntry) -> None:
        """Append a single entry to the JSONL file."""
        try:
            with open(self._journal_path, "a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except OSError as e:
            logger.warning("Failed to write journal entry: %s", e)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ── Recording ─────────────────────────────────────────────────────

    def record_hypothesis(
        self,
        hypothesis_id: str,
        description: str,
        move_category: str,
        rationale: str,
    ) -> None:
        """Record a hypothesis being tested."""
        entry = JournalEntry(
            timestamp=self._now(),
            entry_type="hypothesis",
            hypothesis_id=hypothesis_id,
            content=description,
            tags=[move_category],
            metadata={"rationale": rationale},
        )
        self._entries.append(entry)
        self._append_to_file(entry)

        if self._memory:
            self._memory.ingest(
                f"[HYPOTHESIS] {hypothesis_id}: {description}\n"
                f"Category: {move_category}\n"
                f"Rationale: {rationale}",
                metadata={
                    "type": "hypothesis",
                    "session_id": self.session_id,
                    "hypothesis_id": hypothesis_id,
                },
            )

    def record_result(
        self,
        hypothesis_id: str,
        outcome: str,
        score_before: float | None = None,
        score_after: float | None = None,
        findings: list[str] | None = None,
        best_config: dict[str, Any] | None = None,
    ) -> None:
        """Record the result of testing a hypothesis."""
        improvement = ""
        if score_before is not None and score_after is not None:
            delta = score_after - score_before
            improvement = f" (delta: {delta:+.4f})"

        content = f"Outcome: {outcome}{improvement}"
        if findings:
            content += "\nFindings:\n" + "\n".join(f"  - {f}" for f in findings)

        entry = JournalEntry(
            timestamp=self._now(),
            entry_type="result",
            hypothesis_id=hypothesis_id,
            content=content,
            outcome=outcome,
            score_before=score_before,
            score_after=score_after,
            metadata={
                "findings": findings or [],
                "best_config": best_config,
            },
        )
        self._entries.append(entry)
        self._append_to_file(entry)

        if self._memory:
            self._memory.ingest(
                f"[RESULT] Hypothesis {hypothesis_id}: {outcome}{improvement}\n"
                f"{content}",
                metadata={
                    "type": "result",
                    "session_id": self.session_id,
                    "hypothesis_id": hypothesis_id,
                    "outcome": outcome,
                },
            )

    def record_finding(self, content: str, tags: list[str] | None = None) -> None:
        """Record a general finding or insight."""
        entry = JournalEntry(
            timestamp=self._now(),
            entry_type="finding",
            content=content,
            tags=tags or [],
        )
        self._entries.append(entry)
        self._append_to_file(entry)

        if self._memory:
            self._memory.ingest(
                f"[FINDING] {content}",
                metadata={
                    "type": "finding",
                    "session_id": self.session_id,
                },
            )

    def record_dead_end(self, hypothesis_id: str, reason: str) -> None:
        """Record a dead end to avoid in future sessions."""
        entry = JournalEntry(
            timestamp=self._now(),
            entry_type="dead_end",
            hypothesis_id=hypothesis_id,
            content=reason,
        )
        self._entries.append(entry)
        self._append_to_file(entry)

        if self._memory:
            self._memory.ingest(
                f"[DEAD END] Hypothesis {hypothesis_id}: {reason}",
                metadata={
                    "type": "dead_end",
                    "session_id": self.session_id,
                    "hypothesis_id": hypothesis_id,
                },
            )

    # ── Querying ──────────────────────────────────────────────────────

    @property
    def entries(self) -> list[JournalEntry]:
        return list(self._entries)

    def get_hypothesis_outcomes(self) -> dict[str, str]:
        """Map hypothesis_id -> outcome for all tested hypotheses."""
        outcomes: dict[str, str] = {}
        for entry in self._entries:
            if entry.entry_type == "result" and entry.hypothesis_id and entry.outcome:
                outcomes[entry.hypothesis_id] = entry.outcome
        return outcomes

    def get_dead_ends(self) -> list[str]:
        """Get all dead-end descriptions."""
        return [
            entry.content
            for entry in self._entries
            if entry.entry_type == "dead_end"
        ]

    def get_confirmed_findings(self) -> list[str]:
        """Get findings from confirmed hypotheses."""
        confirmed_ids = {
            hid for hid, outcome in self.get_hypothesis_outcomes().items()
            if outcome == "CONFIRMED"
        }
        findings = []
        for entry in self._entries:
            if entry.entry_type == "result" and entry.hypothesis_id in confirmed_ids:
                for f in entry.metadata.get("findings", []):
                    findings.append(f)
        return findings

    def get_context_for_llm(self, max_entries: int = 20) -> str:
        """Build a text summary suitable for LLM context."""
        if not self._entries:
            return "No previous experiments in this session."

        lines = [f"## Research Journal (session: {self.session_id})\n"]

        # Summarize outcomes
        outcomes = self.get_hypothesis_outcomes()
        if outcomes:
            confirmed = sum(1 for o in outcomes.values() if o == "CONFIRMED")
            rejected = sum(1 for o in outcomes.values() if o == "REJECTED")
            lines.append(
                f"Hypotheses tested: {len(outcomes)} "
                f"({confirmed} confirmed, {rejected} rejected)\n"
            )

        # Recent entries
        recent = self._entries[-max_entries:]
        for entry in recent:
            prefix = entry.entry_type.upper()
            hid = f" [{entry.hypothesis_id}]" if entry.hypothesis_id else ""
            outcome_str = f" -> {entry.outcome}" if entry.outcome else ""
            lines.append(f"- [{prefix}]{hid}{outcome_str}: {entry.content}")

        # Dead ends
        dead_ends = self.get_dead_ends()
        if dead_ends:
            lines.append("\n### Dead Ends (avoid these)")
            for de in dead_ends:
                lines.append(f"  - {de}")

        return "\n".join(lines)

    def search_memory(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search Ultramemory for related experiments across sessions."""
        if self._memory:
            return self._memory.search(query, limit=limit)
        return []
