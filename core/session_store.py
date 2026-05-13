"""SQLite-backed local session storage for the Pi control server.

The React Native app does not currently rely on these endpoints (it uses a
remote backend for history), but the control server streams hits and can
optionally persist them locally for debugging and future offline history.

Design goals:
- stdlib only (sqlite3 + json)
- thread-safe (HTTP server is multi-threaded)
- minimal schema that matches the REST shapes emitted by control_server.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, List, Optional


def _default_db_path() -> str:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    return os.environ.get("SHOOTERRANGE_SESSION_DB", os.path.join(root, "sessions.sqlite3"))


class SessionStore:
    def __init__(self, db_path: Optional[str] = None) -> None:
        self._lock = threading.RLock()
        self._db_path = db_path or _default_db_path()
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._active_id = self._load_active_id()

    # ---- schema ----
    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    shooter_id TEXT NOT NULL,
                    discipline TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    ended_at REAL,
                    total_score REAL NOT NULL DEFAULT 0,
                    shot_count INTEGER NOT NULL DEFAULT 0,
                    shots_per_target INTEGER,
                    targets_per_session INTEGER,
                    notes TEXT
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS hits (
                    session_id TEXT NOT NULL,
                    ts REAL NOT NULL,
                    payload TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_hits_session_ts ON hits(session_id, ts)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC)"
            )

    def _load_active_id(self) -> Optional[str]:
        row = self._conn.execute(
            "SELECT id FROM sessions WHERE ended_at IS NULL ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        return str(row["id"]) if row else None

    # ---- public API (called by ControlState) ----
    def get_active_id(self) -> Optional[str]:
        with self._lock:
            return self._active_id

    def start_session(
        self,
        shooter_id: str,
        discipline: str,
        shots_per_target: Optional[int],
        targets_per_session: Optional[int],
    ) -> Dict[str, Any]:
        with self._lock, self._conn:
            sid = str(uuid.uuid4())
            started_at = time.time()
            self._conn.execute(
                """
                INSERT INTO sessions(
                    id, shooter_id, discipline, started_at, ended_at,
                    total_score, shot_count, shots_per_target, targets_per_session
                ) VALUES(?, ?, ?, ?, NULL, 0, 0, ?, ?)
                """,
                (
                    sid,
                    shooter_id,
                    discipline,
                    float(started_at),
                    int(shots_per_target) if shots_per_target is not None else None,
                    int(targets_per_session) if targets_per_session is not None else None,
                ),
            )
            self._active_id = sid
            return {"session_id": sid, "started_at": started_at}

    def end_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT id, total_score, shot_count FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if not row:
                return None
            ended_at = time.time()
            self._conn.execute(
                "UPDATE sessions SET ended_at = ? WHERE id = ?",
                (float(ended_at), session_id),
            )
            if self._active_id == session_id:
                self._active_id = None
            return {
                "session_id": str(row["id"]),
                "ended_at": ended_at,
                "total_score": float(row["total_score"]),
                "shot_count": int(row["shot_count"]),
            }

    def reset_session(self, session_id: str) -> bool:
        with self._lock, self._conn:
            row = self._conn.execute("SELECT id FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if not row:
                return False
            self._conn.execute("DELETE FROM hits WHERE session_id = ?", (session_id,))
            self._conn.execute(
                "UPDATE sessions SET total_score = 0, shot_count = 0 WHERE id = ?",
                (session_id,),
            )
            # Do not implicitly reactivate ended sessions.
            return True

    def append_hit(self, session_id: str, ts: float, hit: Dict[str, Any]) -> None:
        with self._lock, self._conn:
            # Verify session exists.
            row = self._conn.execute(
                "SELECT total_score, shot_count, ended_at FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if not row:
                return
            if row["ended_at"] is not None:
                return

            payload = json.dumps({**hit, "ts": float(ts)}, separators=(",", ":"))
            self._conn.execute(
                "INSERT INTO hits(session_id, ts, payload) VALUES(?, ?, ?)",
                (session_id, float(ts), payload),
            )
            score = hit.get("score", 0)
            try:
                score_i = int(score)
            except Exception:
                score_i = 0
            total = float(row["total_score"]) + score_i
            count = int(row["shot_count"]) + 1
            self._conn.execute(
                "UPDATE sessions SET total_score = ?, shot_count = ? WHERE id = ?",
                (float(total), int(count), session_id),
            )

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            srow = self._conn.execute(
                """
                SELECT id, shooter_id, discipline, started_at, ended_at,
                       total_score, shot_count, shots_per_target, targets_per_session, notes
                FROM sessions WHERE id = ?
                """,
                (session_id,),
            ).fetchone()
            if not srow:
                return None
            hrows = self._conn.execute(
                "SELECT payload FROM hits WHERE session_id = ? ORDER BY ts ASC",
                (session_id,),
            ).fetchall()
            hits: List[Dict[str, Any]] = []
            for hr in hrows:
                try:
                    hits.append(json.loads(hr["payload"]))
                except Exception:
                    continue
            return {
                "id": str(srow["id"]),
                "shooter_id": str(srow["shooter_id"]),
                "discipline": str(srow["discipline"]),
                "started_at": float(srow["started_at"]),
                "ended_at": float(srow["ended_at"]) if srow["ended_at"] is not None else None,
                "total_score": float(srow["total_score"]),
                "shot_count": int(srow["shot_count"]),
                "hits": hits,
                **({"notes": str(srow["notes"])} if srow["notes"] else {}),
                "shots_per_target": int(srow["shots_per_target"]) if srow["shots_per_target"] is not None else None,
                "targets_per_session": int(srow["targets_per_session"]) if srow["targets_per_session"] is not None else None,
            }

    def list_sessions(self, shooter_id: Optional[str] = None, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        lim = max(1, min(500, int(limit)))
        off = max(0, int(offset))
        with self._lock:
            where = ""
            params: List[Any] = []
            if shooter_id:
                where = "WHERE shooter_id = ?"
                params.append(shooter_id)
            total_row = self._conn.execute(
                f"SELECT COUNT(1) AS n FROM sessions {where}",
                tuple(params),
            ).fetchone()
            total = int(total_row["n"]) if total_row else 0
            rows = self._conn.execute(
                f"""
                SELECT id, shooter_id, discipline, started_at, ended_at, total_score, shot_count
                FROM sessions {where}
                ORDER BY started_at DESC
                LIMIT ? OFFSET ?
                """,
                tuple(params + [lim, off]),
            ).fetchall()
            items = [
                {
                    "id": str(r["id"]),
                    "shooter_id": str(r["shooter_id"]),
                    "discipline": str(r["discipline"]),
                    "started_at": float(r["started_at"]),
                    "ended_at": float(r["ended_at"]) if r["ended_at"] is not None else None,
                    "total_score": float(r["total_score"]),
                    "shot_count": int(r["shot_count"]),
                }
                for r in rows
            ]
            return {"items": items, "total": total}
