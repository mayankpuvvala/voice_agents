"""SQLite persistence for calls, transcripts, reservations and notes."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator
from zoneinfo import ZoneInfo

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at        TEXT NOT NULL,
    ended_at          TEXT,
    caller_name       TEXT,
    caller_phone      TEXT,
    summary           TEXT,
    follow_ups        TEXT,
    recording_path    TEXT,
    recording_seconds REAL
);

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id    INTEGER NOT NULL REFERENCES calls(id) ON DELETE CASCADE,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- No capacity/conflict checking on purpose: the owner manages the physical
-- table, the bot just logs who asked for what and when.
CREATE TABLE IF NOT EXISTS reservations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id       INTEGER REFERENCES calls(id) ON DELETE SET NULL,
    name          TEXT NOT NULL,
    phone         TEXT,
    guests_count  TEXT,
    starts_at     TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'booked',
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS notes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id    INTEGER REFERENCES calls(id) ON DELETE SET NULL,
    category   TEXT NOT NULL DEFAULT 'general',
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_call ON messages(call_id);
CREATE INDEX IF NOT EXISTS idx_reservations_start ON reservations(starts_at);
"""


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(settings.db_path, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after a database already existed.

    CREATE TABLE IF NOT EXISTS only helps on a fresh database — an existing
    calls table from before recording support needs these added by hand.
    """
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(calls)")}
    if "recording_path" not in existing:
        conn.execute("ALTER TABLE calls ADD COLUMN recording_path TEXT")
    if "recording_seconds" not in existing:
        conn.execute("ALTER TABLE calls ADD COLUMN recording_seconds REAL")


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)


def _now() -> str:
    """Wall-clock time in the restaurant's own timezone, not the server's."""
    return datetime.now(ZoneInfo(settings.business_timezone)).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- calls


def start_call() -> int:
    with connect() as conn:
        cur = conn.execute("INSERT INTO calls (started_at) VALUES (?)", (_now(),))
        return int(cur.lastrowid)


def end_call(call_id: int, summary: str, caller_name: str | None,
             caller_phone: str | None, follow_ups: list[str],
             recording_path: str | None = None,
             recording_seconds: float | None = None) -> None:
    with connect() as conn:
        conn.execute(
            """UPDATE calls
                  SET ended_at = ?, summary = ?, caller_name = ?,
                      caller_phone = ?, follow_ups = ?,
                      recording_path = ?, recording_seconds = ?
                WHERE id = ?""",
            (_now(), summary, caller_name, caller_phone, json.dumps(follow_ups),
             recording_path, recording_seconds, call_id),
        )


def add_message(call_id: int, role: str, content: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO messages (call_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (call_id, role, content, _now()),
        )


def list_calls(limit: int = 100) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            """SELECT c.*,
                      (SELECT COUNT(*) FROM messages m WHERE m.call_id = c.id)      AS message_count,
                      (SELECT COUNT(*) FROM reservations r WHERE r.call_id = c.id)  AS reservation_count,
                      (SELECT COUNT(*) FROM notes n WHERE n.call_id = c.id)         AS note_count
                 FROM calls c
                ORDER BY c.id DESC
                LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_call(call_id: int) -> dict[str, Any] | None:
    with connect() as conn:
        call = conn.execute("SELECT * FROM calls WHERE id = ?", (call_id,)).fetchone()
        if call is None:
            return None
        messages = conn.execute(
            "SELECT role, content, created_at FROM messages WHERE call_id = ? ORDER BY id",
            (call_id,),
        ).fetchall()
        reservations = conn.execute(
            "SELECT * FROM reservations WHERE call_id = ? ORDER BY starts_at", (call_id,)
        ).fetchall()
        notes = conn.execute(
            "SELECT * FROM notes WHERE call_id = ? ORDER BY id", (call_id,)
        ).fetchall()
    out = dict(call)
    out["follow_ups"] = json.loads(out["follow_ups"]) if out.get("follow_ups") else []
    out["messages"] = [dict(m) for m in messages]
    out["reservations"] = [dict(r) for r in reservations]
    out["notes"] = [dict(n) for n in notes]
    return out


# -------------------------------------------------------------------- reservations


def create_reservation(call_id: int | None, name: str, phone: str | None,
                        guests_count: str | None, starts_at: datetime) -> int:
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO reservations
                   (call_id, name, phone, guests_count, starts_at, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (call_id, name, phone, guests_count,
             starts_at.isoformat(timespec="minutes"), _now()),
        )
        return int(cur.lastrowid)


def list_reservations(limit: int = 200) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM reservations ORDER BY starts_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def cancel_reservation(reservation_id: int) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE reservations SET status = 'cancelled' WHERE id = ?", (reservation_id,)
        )


# --------------------------------------------------------------------------- notes


def create_note(call_id: int | None, category: str, content: str) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO notes (call_id, category, content, created_at) VALUES (?, ?, ?, ?)",
            (call_id, category, content, _now()),
        )
        return int(cur.lastrowid)


def list_notes(limit: int = 200) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM notes ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
