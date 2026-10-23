"""SQLite storage of records, sessions, profiles, beats, events and alerts (ARQUITETURA §8)."""

from __future__ import annotations

import datetime
import json
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    record_id TEXT PRIMARY KEY,
    split TEXT NOT NULL,
    duration_s REAL,
    fs INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    record_id TEXT NOT NULL,
    use_profile INTEGER NOT NULL,
    feedback_enabled INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    model_run TEXT
);
CREATE TABLE IF NOT EXISTS profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    version INTEGER NOT NULL,
    bins_json TEXT NOT NULL,
    q_json TEXT NOT NULL,
    mu REAL NOT NULL,
    sigma REAL NOT NULL,
    k REAL NOT NULL,
    L REAL NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS beats (
    session_id TEXT NOT NULL REFERENCES sessions(id),
    sample INTEGER NOT NULL,
    cls TEXT NOT NULL,
    prob REAL NOT NULL,
    attention_json TEXT NOT NULL,
    true_label TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    kind TEXT NOT NULL,
    sample INTEGER NOT NULL,
    level TEXT NOT NULL,
    cls TEXT,
    kl REAL,
    L REAL,
    recv_ts REAL NOT NULL,
    emit_ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    event_id INTEGER REFERENCES events(id),
    episode_id INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    decided_at TEXT,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS beats_session ON beats(session_id, sample);
CREATE INDEX IF NOT EXISTS events_session ON events(session_id, sample);
CREATE INDEX IF NOT EXISTS alerts_session ON alerts(session_id);
"""

PENDING = "pending"
DECISION_STATUS = {"confirm": "confirmed", "discard": "discarded"}


def now() -> str:
    return datetime.datetime.now().isoformat(timespec="milliseconds")


class Database:
    """One connection shared by the server threads; a lock serializes the writes."""

    def __init__(self, path: str | Path) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def _write(self, sql: str, params: Sequence[Any] = ()) -> int:
        with self._lock, self._conn:
            return self._conn.execute(sql, params).lastrowid

    def _query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def upsert_records(self, rows: Iterable[tuple[str, str, float | None, int]]) -> None:
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT INTO records VALUES (?, ?, ?, ?) ON CONFLICT(record_id) DO UPDATE SET "
                "split = excluded.split, duration_s = excluded.duration_s, fs = excluded.fs",
                list(rows),
            )

    def records(self) -> list[dict[str, Any]]:
        return self._query("SELECT * FROM records ORDER BY split DESC, record_id")

    def create_session(
        self,
        session_id: str,
        record_id: str,
        use_profile: bool,
        feedback_enabled: bool,
        model_run: str | None,
    ) -> None:
        self._write(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, NULL, ?)",
            (session_id, record_id, int(use_profile), int(feedback_enabled), now(), model_run),
        )

    def end_session(self, session_id: str) -> None:
        self._write("UPDATE sessions SET ended_at = ? WHERE id = ? AND ended_at IS NULL", (now(), session_id))

    def session(self, session_id: str) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM sessions WHERE id = ?", (session_id,))
        return rows[0] if rows else None

    def sessions(self) -> list[dict[str, Any]]:
        return self._query("SELECT * FROM sessions ORDER BY started_at DESC")

    def save_profile(self, session_id: str, profile: dict[str, Any], k: float, threshold: float) -> None:
        self._write(
            "INSERT INTO profiles (session_id, version, bins_json, q_json, mu, sigma, k, L, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                profile["version"],
                json.dumps(profile["edges"]),
                json.dumps(profile["q"]),
                profile["mu"],
                profile["sigma"],
                k,
                threshold,
                now(),
            ),
        )

    def profiles(self, session_id: str) -> list[dict[str, Any]]:
        return self._query(
            "SELECT id, version, mu, sigma, k, L, created_at FROM profiles "
            "WHERE session_id = ? ORDER BY version",
            (session_id,),
        )

    def save_beats(self, session_id: str, rows: Iterable[tuple[int, str, float, list[float]]]) -> None:
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT INTO beats (session_id, sample, cls, prob, attention_json) VALUES (?, ?, ?, ?, ?)",
                [(session_id, s, c, p, json.dumps(a)) for s, c, p, a in rows],
            )

    def beat_count(self, session_id: str) -> int:
        return self._query("SELECT COUNT(*) AS n FROM beats WHERE session_id = ?", (session_id,))[0]["n"]

    def save_event(
        self,
        session_id: str,
        kind: str,
        sample: int,
        level: str,
        cls: str | None,
        kl: float | None,
        threshold: float | None,
        recv_ts: float,
        emit_ts: float,
    ) -> int:
        return self._write(
            "INSERT INTO events (session_id, kind, sample, level, cls, kl, L, recv_ts, emit_ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, kind, sample, level, cls, kl, threshold, recv_ts, emit_ts),
        )

    def events(self, session_id: str) -> list[dict[str, Any]]:
        return self._query("SELECT * FROM events WHERE session_id = ? ORDER BY sample, id", (session_id,))

    def create_alert(
        self, alert_id: str, session_id: str, event_id: int | None, episode_id: int, payload: dict[str, Any]
    ) -> None:
        self._write(
            "INSERT INTO alerts (id, session_id, event_id, episode_id, payload_json) VALUES (?, ?, ?, ?, ?)",
            (alert_id, session_id, event_id, episode_id, json.dumps(payload)),
        )

    def alert(self, alert_id: str) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM alerts WHERE id = ?", (alert_id,))
        return _alert_row(rows[0]) if rows else None

    def alerts(self, session_id: str) -> list[dict[str, Any]]:
        rows = self._query("SELECT * FROM alerts WHERE session_id = ? ORDER BY rowid", (session_id,))
        return [_alert_row(r) for r in rows]

    def decide_alert(self, alert_id: str, decision: str) -> bool:
        """Set the status of a pending alert. Return False if the alert is not pending."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE alerts SET status = ?, decided_at = ? WHERE id = ? AND status = ?",
                (DECISION_STATUS[decision], now(), alert_id, PENDING),
            )
            return cur.rowcount == 1


def _alert_row(row: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(row.pop("payload_json"))
    return {**row, "payload": payload}
