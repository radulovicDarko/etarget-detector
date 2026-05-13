"""SQLite-backed session storage.

SQLite is built into Python's stdlib (``sqlite3``) so no extra deps are
needed. A single file at ``smokeless-range-software/sessions.db`` holds
all sessions and their hits. WAL journal + busy timeout make the database
robust to the concurrent reads (HTTP threads) and writes (cv2 main loop)
the control server does.

Schema:

  sessions(id TEXT PRIMARY KEY, shooter_id, discipline, started_at,
           ended_at, shots_per_target, targets_per_session)
  hits(session_id REF sessions(id), ts, x_norm, y_norm, score, ring,
       x_mm, y_mm, dist_mm, is_inner_ten)

The ``total_score`` and ``shot_count`` summary fields are computed on
demand via SUM/COUNT — keeps the schema simple and the numbers always
match the underlying hits.

Rough capacity (back-of-the-envelope):
  ~120 bytes per hit row on disk
  60 shots/session × 10,000 sessions ≈ 60 MB total. The Pi's SD card
  has many GB free, no external storage is required.
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional


_DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "sessions.db",
)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    shooter_id TEXT NOT NULL,
    discipline TEXT NOT NULL,
    started_at REAL NOT NULL,
    ended_at REAL,
    shots_per_target INTEGER,
    targets_per_session INTEGER
);

CREATE INDEX IF NOT EXISTS idx_sessions_shooter
    ON sessions(shooter_id, started_at DESC);

CREATE TABLE IF NOT EXISTS hits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    ts REAL NOT NULL,
    x_norm REAL NOT NULL,
    y_norm REAL NOT NULL,
    score INTEGER NOT NULL,
    ring INTEGER NOT NULL,
    x_mm REAL NOT NULL,
    y_mm REAL NOT NULL,
    dist_mm REAL NOT NULL,
    is_inner_ten INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_hits_session_ts
    ON hits(session_id, ts);
"""


class SessionStore:
    """Thread-safe SQLite-backed session storage."""

    def __init__(self, path: str = _DEFAULT_PATH) -> None:
        self.path = path
        # WAL allows concurrent reads while a writer is active.
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
        self._lock = threading.RLock()
        self._active_session_id: Optional[str] = None

    @contextmanager
    def _connect(self):
        # Per-call connection: each HTTP thread + the cv2 thread can call
        # methods concurrently without sharing a single connection. SQLite
        # serialises writes internally; busy_timeout makes us wait briefly
        # rather than getting "database is locked" errors.
        conn = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
        finally:
            conn.close()

    # ---- active session bookkeeping ------------------------------------
    def get_active_id(self) -> Optional[str]:
        with self._lock:
            return self._active_session_id

    def _set_active_id(self, session_id: Optional[str]) -> None:
        with self._lock:
            self._active_session_id = session_id

    # ---- CRUD ----------------------------------------------------------
    def start_session(
        self,
        shooter_id: str,
        discipline: str,
        shots_per_target: Optional[int],
        targets_per_session: Optional[int],
    ) -> Dict[str, Any]:
        session_id = uuid.uuid4().hex
        started_at = time.time()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions(id, shooter_id, discipline, started_at, "
                "ended_at, shots_per_target, targets_per_session) "
                "VALUES (?, ?, ?, ?, NULL, ?, ?)",
                (session_id, shooter_id, discipline, started_at,
                 shots_per_target, targets_per_session),
            )
        self._set_active_id(session_id)
        return {"session_id": session_id, "started_at": started_at}

    def end_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        ended_at = time.time()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, ended_at FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            if row["ended_at"] is None:
                conn.execute(
                    "UPDATE sessions SET ended_at = ? WHERE id = ?",
                    (ended_at, session_id),
                )
            else:
                ended_at = float(row["ended_at"])
            totals = conn.execute(
                "SELECT COALESCE(SUM(score), 0) AS total_score, "
                "COUNT(*) AS shot_count FROM hits WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if self.get_active_id() == session_id:
            self._set_active_id(None)
        return {
            "session_id": session_id,
            "ended_at": ended_at,
            "total_score": int(totals["total_score"] or 0),
            "shot_count": int(totals["shot_count"] or 0),
        }

    def reset_session(self, session_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return False
            conn.execute("DELETE FROM hits WHERE session_id = ?", (session_id,))
        return True

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            srow = conn.execute(
                "SELECT * FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if srow is None:
                return None
            hits = [
                {
                    "ts": float(h["ts"]),
                    "x_norm": float(h["x_norm"]),
                    "y_norm": float(h["y_norm"]),
                    "score": int(h["score"]),
                    "ring": int(h["ring"]),
                    "x_mm": float(h["x_mm"]),
                    "y_mm": float(h["y_mm"]),
                    "dist_mm": float(h["dist_mm"]),
                    "is_inner_ten": bool(h["is_inner_ten"]),
                }
                for h in conn.execute(
                    "SELECT * FROM hits WHERE session_id = ? ORDER BY ts ASC",
                    (session_id,),
                )
            ]
            totals = conn.execute(
                "SELECT COALESCE(SUM(score), 0) AS total_score, "
                "COUNT(*) AS shot_count FROM hits WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return {
            "id": srow["id"],
            "shooter_id": srow["shooter_id"],
            "discipline": srow["discipline"],
            "started_at": float(srow["started_at"]),
            "ended_at": float(srow["ended_at"]) if srow["ended_at"] is not None else None,
            "total_score": int(totals["total_score"] or 0),
            "shot_count": int(totals["shot_count"] or 0),
            "hits": hits,
            "shots_per_target": srow["shots_per_target"],
            "targets_per_session": srow["targets_per_session"],
        }

    def list_sessions(
        self,
        shooter_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        params: List[Any] = []
        where = ""
        if shooter_id:
            where = " WHERE s.shooter_id = ?"
            params.append(shooter_id)

        with self._connect() as conn:
            total_row = conn.execute(
                f"SELECT COUNT(*) AS n FROM sessions s{where}",
                params,
            ).fetchone()
            total = int(total_row["n"] or 0)

            rows = conn.execute(
                f"""
                SELECT s.id, s.shooter_id, s.discipline, s.started_at, s.ended_at,
                       COALESCE(SUM(h.score), 0) AS total_score,
                       COUNT(h.id)               AS shot_count
                FROM sessions s
                LEFT JOIN hits h ON h.session_id = s.id
                {where}
                GROUP BY s.id
                ORDER BY s.started_at DESC
                LIMIT ? OFFSET ?
                """,
                (*params, limit, offset),
            ).fetchall()

        items = [
            {
                "id": r["id"],
                "shooter_id": r["shooter_id"],
                "discipline": r["discipline"],
                "started_at": float(r["started_at"]),
                "ended_at": float(r["ended_at"]) if r["ended_at"] is not None else None,
                "total_score": int(r["total_score"] or 0),
                "shot_count": int(r["shot_count"] or 0),
            }
            for r in rows
        ]
        return {"items": items, "total": total}

    def append_hit(self, session_id: str, ts: float, hit: Dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO hits(session_id, ts, x_norm, y_norm, score, ring, "
                "x_mm, y_mm, dist_mm, is_inner_ten) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    float(ts),
                    float(hit.get("x_norm", 0.0)),
                    float(hit.get("y_norm", 0.0)),
                    int(hit.get("score", 0)),
                    int(hit.get("ring", 0)),
                    float(hit.get("x_mm", 0.0)),
                    float(hit.get("y_mm", 0.0)),
                    float(hit.get("dist_mm", 0.0)),
                    1 if hit.get("is_inner_ten") else 0,
                ),
            )
