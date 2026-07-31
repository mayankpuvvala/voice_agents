"""SQLite persistence for calls, transcripts, appointments and notes."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Iterator

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL,
    ended_at      TEXT,
    caller_name   TEXT,
    caller_phone  TEXT,
    summary       TEXT,
    follow_ups    TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id    INTEGER NOT NULL REFERENCES calls(id) ON DELETE CASCADE,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS appointments (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id          INTEGER REFERENCES calls(id) ON DELETE SET NULL,
    name             TEXT NOT NULL,
    phone            TEXT,
    email            TEXT,
    starts_at        TEXT NOT NULL,
    duration_minutes INTEGER NOT NULL,
    reason           TEXT,
    status           TEXT NOT NULL DEFAULT 'booked',
    created_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS notes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id    INTEGER REFERENCES calls(id) ON DELETE SET NULL,
    category   TEXT NOT NULL DEFAULT 'general',
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_call ON messages(call_id);
CREATE INDEX IF NOT EXISTS idx_appointments_start ON appointments(starts_at);
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


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# --------------------------------------------------------------------------- calls


def start_call() -> int:
    with connect() as conn:
        cur = conn.execute("INSERT INTO calls (started_at) VALUES (?)", (_now(),))
        return int(cur.lastrowid)


def end_call(call_id: int, summary: str, caller_name: str | None,
             caller_phone: str | None, follow_ups: list[str]) -> None:
    with connect() as conn:
        conn.execute(
            """UPDATE calls
                  SET ended_at = ?, summary = ?, caller_name = ?,
                      caller_phone = ?, follow_ups = ?
                WHERE id = ?""",
            (_now(), summary, caller_name, caller_phone, json.dumps(follow_ups), call_id),
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
                      (SELECT COUNT(*) FROM messages m WHERE m.call_id = c.id)     AS message_count,
                      (SELECT COUNT(*) FROM appointments a WHERE a.call_id = c.id) AS appointment_count,
                      (SELECT COUNT(*) FROM notes n WHERE n.call_id = c.id)        AS note_count
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
        appts = conn.execute(
            "SELECT * FROM appointments WHERE call_id = ? ORDER BY starts_at", (call_id,)
        ).fetchall()
        notes = conn.execute(
            "SELECT * FROM notes WHERE call_id = ? ORDER BY id", (call_id,)
        ).fetchall()
    out = dict(call)
    out["follow_ups"] = json.loads(out["follow_ups"]) if out.get("follow_ups") else []
    out["messages"] = [dict(m) for m in messages]
    out["appointments"] = [dict(a) for a in appts]
    out["notes"] = [dict(n) for n in notes]
    return out


# -------------------------------------------------------------------- appointments


def find_conflict(starts_at: datetime, duration_minutes: int) -> dict[str, Any] | None:
    """Return an existing booked appointment that overlaps the given window."""
    ends_at = starts_at + timedelta(minutes=duration_minutes)
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM appointments WHERE status = 'booked'"
        ).fetchall()
    for row in rows:
        try:
            other_start = datetime.fromisoformat(row["starts_at"])
        except ValueError:
            continue
        other_end = other_start + timedelta(minutes=row["duration_minutes"])
        if starts_at < other_end and other_start < ends_at:
            return dict(row)
    return None


def create_appointment(call_id: int | None, name: str, phone: str | None, email: str | None,
                       starts_at: datetime, duration_minutes: int, reason: str | None) -> int:
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO appointments
                   (call_id, name, phone, email, starts_at, duration_minutes, reason, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (call_id, name, phone, email, starts_at.isoformat(timespec="minutes"),
             duration_minutes, reason, _now()),
        )
        return int(cur.lastrowid)


def booked_on(day: datetime) -> list[dict[str, Any]]:
    prefix = day.strftime("%Y-%m-%d")
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM appointments WHERE status = 'booked' AND starts_at LIKE ? ORDER BY starts_at",
            (f"{prefix}%",),
        ).fetchall()
    return [dict(r) for r in rows]


def list_appointments(limit: int = 200) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM appointments ORDER BY starts_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def cancel_appointment(appointment_id: int) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE appointments SET status = 'cancelled' WHERE id = ?", (appointment_id,)
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
