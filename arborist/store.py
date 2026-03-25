"""SQLite persistence layer for trees, nodes, and insights."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generate_id() -> str:
    return uuid.uuid4().hex[:12]


class Store:
    """Thread-safe SQLite store for arborist data."""

    def __init__(self, db_path: str = "./arborist.db") -> None:
        self.db_path = db_path
        self._local = threading.local()
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def _cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_schema(self) -> None:
        with self._cursor() as cur:
            cur.executescript("""
                CREATE TABLE IF NOT EXISTS trees (
                    id TEXT PRIMARY KEY,
                    goal TEXT NOT NULL,
                    strategy TEXT NOT NULL DEFAULT 'ucb',
                    status TEXT NOT NULL DEFAULT 'running',
                    config TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    tree_id TEXT NOT NULL REFERENCES trees(id),
                    parent_id TEXT REFERENCES nodes(id),
                    depth INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'pending',
                    hypothesis TEXT,
                    config TEXT NOT NULL,
                    results TEXT,
                    score REAL,
                    cost_usd REAL,
                    duration_ms INTEGER,
                    pruned INTEGER NOT NULL DEFAULT 0,
                    prune_reason TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    FOREIGN KEY (tree_id) REFERENCES trees(id)
                );

                CREATE TABLE IF NOT EXISTS insights (
                    id TEXT PRIMARY KEY,
                    tree_id TEXT NOT NULL REFERENCES trees(id),
                    source_node_ids TEXT NOT NULL,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    confidence REAL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (tree_id) REFERENCES trees(id)
                );

                CREATE INDEX IF NOT EXISTS idx_nodes_tree ON nodes(tree_id);
                CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_id);
                CREATE INDEX IF NOT EXISTS idx_nodes_status ON nodes(status);
                CREATE INDEX IF NOT EXISTS idx_nodes_score ON nodes(score);
                CREATE INDEX IF NOT EXISTS idx_insights_tree ON insights(tree_id);
            """)

    # ── Trees ──────────────────────────────────────────────────────────

    def create_tree(
        self,
        goal: str,
        strategy: str = "ucb",
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        tree_id = _generate_id()
        now = _now()
        # Serialize config if it's a dict; pass through if already a string
        if isinstance(config, dict):
            config_str = json.dumps(config)
        elif isinstance(config, str):
            config_str = config
        else:
            config_str = None
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO trees (id, goal, strategy, status, config, created_at, updated_at) "
                "VALUES (?, ?, ?, 'running', ?, ?, ?)",
                (tree_id, goal, strategy, config_str, now, now),
            )
        return self.get_tree(tree_id)  # type: ignore[return-value]

    def get_tree(self, tree_id: str) -> dict[str, Any] | None:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM trees WHERE id = ?", (tree_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def update_tree(self, tree_id: str, **kwargs: Any) -> None:
        kwargs["updated_at"] = _now()
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [tree_id]
        with self._cursor() as cur:
            cur.execute(f"UPDATE trees SET {sets} WHERE id = ?", vals)

    def list_trees(self) -> list[dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM trees ORDER BY created_at DESC")
            return [dict(r) for r in cur.fetchall()]

    # ── Nodes ──────────────────────────────────────────────────────────

    def create_node(
        self,
        tree_id: str,
        config: dict[str, Any],
        parent_id: str | None = None,
        depth: int = 0,
        hypothesis: str | None = None,
    ) -> dict[str, Any]:
        node_id = _generate_id()
        now = _now()
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO nodes (id, tree_id, parent_id, depth, status, hypothesis, config, created_at) "
                "VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)",
                (node_id, tree_id, parent_id, depth, hypothesis, json.dumps(config), now),
            )
        return self.get_node(node_id)  # type: ignore[return-value]

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM nodes WHERE id = ?", (node_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def update_node(self, node_id: str, **kwargs: Any) -> None:
        serialized: dict[str, Any] = {}
        for k, v in kwargs.items():
            if k in ("results", "config") and isinstance(v, dict):
                serialized[k] = json.dumps(v)
            else:
                serialized[k] = v
        sets = ", ".join(f"{k} = ?" for k in serialized)
        vals = list(serialized.values()) + [node_id]
        with self._cursor() as cur:
            cur.execute(f"UPDATE nodes SET {sets} WHERE id = ?", vals)

    def get_tree_nodes(
        self,
        tree_id: str,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._cursor() as cur:
            if status:
                cur.execute(
                    "SELECT * FROM nodes WHERE tree_id = ? AND status = ? ORDER BY depth, created_at",
                    (tree_id, status),
                )
            else:
                cur.execute(
                    "SELECT * FROM nodes WHERE tree_id = ? ORDER BY depth, created_at",
                    (tree_id,),
                )
            return [dict(r) for r in cur.fetchall()]

    def get_children(self, node_id: str) -> list[dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM nodes WHERE parent_id = ? ORDER BY created_at",
                (node_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    def get_siblings(self, node_id: str) -> list[dict[str, Any]]:
        node = self.get_node(node_id)
        if not node:
            return []
        with self._cursor() as cur:
            if node["parent_id"]:
                cur.execute(
                    "SELECT * FROM nodes WHERE parent_id = ? AND id != ? ORDER BY created_at",
                    (node["parent_id"], node_id),
                )
            else:
                cur.execute(
                    "SELECT * FROM nodes WHERE tree_id = ? AND parent_id IS NULL AND id != ? ORDER BY created_at",
                    (node["tree_id"], node_id),
                )
            return [dict(r) for r in cur.fetchall()]

    def count_nodes(self, tree_id: str, status: str | None = None) -> int:
        with self._cursor() as cur:
            if status:
                cur.execute(
                    "SELECT COUNT(*) FROM nodes WHERE tree_id = ? AND status = ?",
                    (tree_id, status),
                )
            else:
                cur.execute("SELECT COUNT(*) FROM nodes WHERE tree_id = ?", (tree_id,))
            return cur.fetchone()[0]

    def get_best_node(self, tree_id: str) -> dict[str, Any] | None:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM nodes WHERE tree_id = ? AND score IS NOT NULL AND pruned = 0 "
                "ORDER BY score DESC LIMIT 1",
                (tree_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def get_top_nodes(self, tree_id: str, k: int = 5) -> list[dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM nodes WHERE tree_id = ? AND score IS NOT NULL AND pruned = 0 "
                "ORDER BY score DESC LIMIT ?",
                (tree_id, k),
            )
            return [dict(r) for r in cur.fetchall()]

    # ── Insights ───────────────────────────────────────────────────────

    def create_insight(
        self,
        tree_id: str,
        source_node_ids: list[str],
        insight_type: str,
        content: str,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        insight_id = _generate_id()
        now = _now()
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO insights (id, tree_id, source_node_ids, type, content, confidence, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (insight_id, tree_id, json.dumps(source_node_ids), insight_type, content, confidence, now),
            )
        return {"id": insight_id, "tree_id": tree_id, "type": insight_type, "content": content}

    def get_insights(self, tree_id: str) -> list[dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM insights WHERE tree_id = ? ORDER BY created_at",
                (tree_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
