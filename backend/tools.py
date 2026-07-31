"""Tools the receptionist can call: knowledge lookup, availability, booking, notes."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from . import db
from .config import settings
from .rag import retriever

def _tool(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """Wrap a schema in the OpenAI chat-completions function-tool envelope."""
    return {"type": "function",
            "function": {"name": name, "description": description, "parameters": parameters}}


TOOLS: list[dict[str, Any]] = [
    _tool(
        "search_knowledge_base",
        "Search the business's own documents (services, pricing, policies, hours, "
        "directions, staff) for an answer. Call this whenever the caller asks "
        "something factual about the business that you have not already been given "
        "in the conversation. Prefer this over answering from memory.",
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look up, in the caller's own words.",
                }
            },
            "required": ["query"],
        },
    ),
    _tool(
        "check_availability",
        "List open appointment slots for a specific date. Call this before offering "
        "the caller any time, so you never propose a slot that is already taken.",
        {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "The date to check, as YYYY-MM-DD.",
                },
                "duration_minutes": {
                    "type": "integer",
                    "description": f"Appointment length. Defaults to {settings.slot_minutes}.",
                },
            },
            "required": ["date"],
        },
    ),
    _tool(
        "book_appointment",
        "Book an appointment. Only call this once you have confirmed the caller's "
        "name, a contact number, and a specific start time you verified with "
        "check_availability.",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Caller's full name."},
                "phone": {"type": "string", "description": "Callback phone number."},
                "email": {"type": "string", "description": "Email address, if given."},
                "starts_at": {
                    "type": "string",
                    "description": "Start time as YYYY-MM-DDTHH:MM (24-hour clock).",
                },
                "duration_minutes": {
                    "type": "integer",
                    "description": f"Length in minutes. Defaults to {settings.slot_minutes}.",
                },
                "reason": {"type": "string", "description": "Short reason for the visit."},
            },
            "required": ["name", "phone", "starts_at"],
        },
    ),
    _tool(
        "save_note",
        "Record something the team needs to see after the call: a message to pass "
        "on, a complaint, a callback request, or a detail worth keeping. Use it as "
        "soon as the caller says something worth recording — do not wait until the "
        "end of the call.",
        {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The note, written for a colleague who was not on the call.",
                },
                "category": {
                    "type": "string",
                    "enum": ["message", "callback", "complaint", "general"],
                    "description": "Which bucket this note belongs in.",
                },
            },
            "required": ["content"],
        },
    ),
]


# --------------------------------------------------------------------------- helpers


def _parse_dt(value: str) -> datetime | None:
    value = (value or "").strip().replace("Z", "")
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _within_hours(start: datetime, duration: int) -> str | None:
    """Return an error message when the slot falls outside business hours."""
    if start.weekday() not in settings.open_days:
        return f"{start:%A} is not a working day. Open {settings.open_days_label}."
    end = start + timedelta(minutes=duration)
    open_at = start.replace(hour=settings.open_hour, minute=0, second=0, microsecond=0)
    close_at = start.replace(hour=settings.close_hour, minute=0, second=0, microsecond=0)
    if start < open_at or end > close_at:
        return f"That is outside opening hours ({settings.hours_label})."
    return None


def free_slots(day: datetime, duration: int) -> list[str]:
    if day.weekday() not in settings.open_days:
        return []
    booked = db.booked_on(day)
    taken: list[tuple[datetime, datetime]] = []
    for row in booked:
        start = _parse_dt(row["starts_at"])
        if start:
            taken.append((start, start + timedelta(minutes=row["duration_minutes"])))

    slots: list[str] = []
    cursor = day.replace(hour=settings.open_hour, minute=0, second=0, microsecond=0)
    close_at = day.replace(hour=settings.close_hour, minute=0, second=0, microsecond=0)
    now = datetime.now()
    while cursor + timedelta(minutes=duration) <= close_at:
        end = cursor + timedelta(minutes=duration)
        overlaps = any(cursor < t_end and t_start < end for t_start, t_end in taken)
        if not overlaps and cursor > now:
            slots.append(cursor.strftime("%H:%M"))
        cursor += timedelta(minutes=settings.slot_minutes)
    return slots


# --------------------------------------------------------------------------- dispatch


def execute(name: str, tool_input: dict[str, Any], call_id: int | None) -> tuple[str, dict[str, Any]]:
    """Run a tool. Returns (result_text_for_the_model, event_payload_for_the_ui)."""
    try:
        if name == "search_knowledge_base":
            query = str(tool_input.get("query", ""))
            hits = retriever.search(query, k=4)
            if not hits:
                outline = retriever.outline()
                message = (
                    "No passage matched those words. This is a keyword search, so try "
                    "the wording the documents would use"
                    + (f". Sections available:\n{outline}" if outline else ".")
                    + "\nIf nothing here covers it, tell the caller you will have "
                    "someone follow up, and save a note."
                )
                return message, {"query": query, "hits": 0}
            body = "\n\n---\n\n".join(chunk.render() for chunk, _ in hits)
            return body, {"query": query, "hits": len(hits),
                          "sources": sorted({c.source for c, _ in hits})}

        if name == "check_availability":
            duration = int(tool_input.get("duration_minutes") or settings.slot_minutes)
            day = _parse_dt(f"{tool_input.get('date', '')}T00:00")
            if day is None:
                return "Invalid date. Use YYYY-MM-DD.", {"error": "bad_date"}
            slots = free_slots(day, duration)
            if not slots:
                return (
                    f"No {duration}-minute slots free on {day:%A %d %B %Y}. "
                    f"Open {settings.open_days_label}, {settings.hours_label}.",
                    {"date": day.strftime("%Y-%m-%d"), "slots": []},
                )
            return (
                f"Free {duration}-minute slots on {day:%A %d %B %Y}: " + ", ".join(slots),
                {"date": day.strftime("%Y-%m-%d"), "slots": slots},
            )

        if name == "book_appointment":
            start = _parse_dt(str(tool_input.get("starts_at", "")))
            if start is None:
                return "Invalid start time. Use YYYY-MM-DDTHH:MM.", {"error": "bad_time"}
            duration = int(tool_input.get("duration_minutes") or settings.slot_minutes)

            if start < datetime.now():
                return "That time is in the past — offer a future slot.", {"error": "past"}

            hours_error = _within_hours(start, duration)
            if hours_error:
                return f"Not booked. {hours_error}", {"error": "hours", "detail": hours_error}

            conflict = db.find_conflict(start, duration)
            if conflict:
                alternatives = free_slots(start, duration)[:4]
                return (
                    "Not booked — that slot is already taken. "
                    + (f"Still free that day: {', '.join(alternatives)}."
                       if alternatives else "Nothing else free that day."),
                    {"error": "conflict", "alternatives": alternatives},
                )

            name_value = str(tool_input.get("name", "")).strip()
            phone = str(tool_input.get("phone", "")).strip()
            if not name_value or not phone:
                return "Not booked — a name and phone number are required.", {"error": "missing"}

            appt_id = db.create_appointment(
                call_id=call_id,
                name=name_value,
                phone=phone,
                email=(tool_input.get("email") or None),
                starts_at=start,
                duration_minutes=duration,
                reason=(tool_input.get("reason") or None),
            )
            return (
                f"Booked (#{appt_id}) for {name_value} on {start:%A %d %B} at {start:%H:%M}, "
                f"{duration} minutes. Confirm this back to the caller.",
                {
                    "id": appt_id,
                    "name": name_value,
                    "starts_at": start.isoformat(timespec="minutes"),
                    "duration_minutes": duration,
                },
            )

        if name == "save_note":
            content = str(tool_input.get("content", "")).strip()
            if not content:
                return "Empty note — nothing saved.", {"error": "empty"}
            category = str(tool_input.get("category") or "general")
            note_id = db.create_note(call_id, category, content)
            return f"Note #{note_id} saved under '{category}'.", {
                "id": note_id, "category": category, "content": content
            }

        return f"Unknown tool '{name}'.", {"error": "unknown_tool"}

    except Exception as exc:  # surface the failure to the model rather than dropping the turn
        return f"Tool '{name}' failed: {exc}", {"error": "exception", "detail": str(exc)}


def summary_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "Two or three sentences on what the caller wanted and what happened.",
            },
            "caller_name": {"type": "string", "description": "Caller's name, or empty if unknown."},
            "caller_phone": {"type": "string", "description": "Phone number, or empty if unknown."},
            "follow_ups": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Concrete actions a staff member must take. Empty if none.",
            },
        },
        "required": ["summary", "caller_name", "caller_phone", "follow_ups"],
        "additionalProperties": False,
    }


def parse_summary(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"summary": text.strip(), "caller_name": "", "caller_phone": "", "follow_ups": []}
    return {
        "summary": str(data.get("summary", "")).strip(),
        "caller_name": str(data.get("caller_name", "")).strip(),
        "caller_phone": str(data.get("caller_phone", "")).strip(),
        "follow_ups": [str(f) for f in data.get("follow_ups", []) if str(f).strip()],
    }
