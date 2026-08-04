"""Tools the receptionist can call: knowledge lookup, reservations, notes."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from . import db, n8n
from .config import settings
from .rag import retriever


def _tool(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """Wrap a schema in the OpenAI chat-completions function-tool envelope."""
    return {"type": "function",
            "function": {"name": name, "description": description, "parameters": parameters}}


TOOLS: list[dict[str, Any]] = [
    _tool(
        "search_knowledge_base",
        "Search the business's own documents (menu, hours, delivery, pricing, "
        "policies, directions) for an answer. Call this whenever the caller asks "
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
        "create_reservation",
        "Log a table reservation. There is no capacity limit to check and no "
        "conflict to reject — the owner manages the physical table, not you. "
        "Only call this once you have a name, a guest count, and a requested "
        "date/time. A phone number is not required for a reservation. If the "
        "caller wants a calendar invite emailed to them, also ask for their "
        "email — this is optional, never block the reservation on it.",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name to hold the table under."},
                "phone": {
                    "type": "string",
                    "description": "Callback number, only if the caller offers one. Empty string otherwise.",
                },
                "email": {
                    "type": "string",
                    "description": "Email for a calendar invite, only if the caller offers one. Empty string otherwise.",
                },
                "guests_count": {"type": "string", "description": "Number of guests, e.g. '4'."},
                "requested_time": {
                    "type": "string",
                    "description": "Requested date/time as YYYY-MM-DDTHH:MM (24-hour clock).",
                },
            },
            "required": ["name", "guests_count", "requested_time"],
        },
    ),
    _tool(
        "update_reservation",
        "Change the guest count and/or time of a reservation that already exists "
        "— found either earlier this call or via find_caller_history. Only the "
        "fields the caller wants changed need to be given.",
        {
            "type": "object",
            "properties": {
                "reservation_id": {"type": "integer", "description": "The reservation's id."},
                "guests_count": {
                    "type": "string",
                    "description": "New guest count, e.g. '4'. Empty string to leave unchanged.",
                },
                "requested_time": {
                    "type": "string",
                    "description": "New date/time as YYYY-MM-DDTHH:MM. Empty string to leave unchanged.",
                },
            },
            "required": ["reservation_id"],
        },
    ),
    _tool(
        "cancel_reservation",
        "Cancel a reservation that already exists — found either earlier this "
        "call or via find_caller_history. Confirm which reservation with the "
        "caller before cancelling if they have more than one upcoming.",
        {
            "type": "object",
            "properties": {
                "reservation_id": {"type": "integer", "description": "The reservation's id."},
            },
            "required": ["reservation_id"],
        },
    ),
    _tool(
        "find_caller_history",
        "Look up this caller's past reservations and past call summaries by "
        "name and/or phone number. Use this as soon as you have a name or "
        "number, near the start of the call, so you can greet a returning "
        "caller warmly and reference what they've contacted about before. "
        "Also use it to find the reservation_id needed to update or cancel a "
        "booking made in an earlier call. Returns nothing found for a new "
        "caller — that's normal, just continue the call as usual.",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Caller's name, if given. Empty string otherwise."},
                "phone": {"type": "string", "description": "Caller's phone number, if given. Empty string otherwise."},
            },
            "required": [],
        },
    ),
    _tool(
        "save_note",
        "Record something the team needs to see after the call: a message to pass "
        "on, a complaint, a callback request, or a detail worth keeping — including "
        "when a caller pushes back on something you've already declined. Use it as "
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


def _now_local() -> datetime:
    """Naive local time in the restaurant's own timezone, to compare against
    the naive YYYY-MM-DDTHH:MM times the model sends."""
    return datetime.now(ZoneInfo(settings.business_timezone)).replace(tzinfo=None)


# --------------------------------------------------------------------------- dispatch


async def execute(name: str, tool_input: dict[str, Any], call_id: int | None) -> tuple[str, dict[str, Any]]:
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

        if name == "create_reservation":
            start = _parse_dt(str(tool_input.get("requested_time", "")))
            if start is None:
                return "Invalid time. Use YYYY-MM-DDTHH:MM.", {"error": "bad_time"}

            if start < _now_local():
                return "That time is in the past — ask for a future date/time.", {"error": "past"}

            name_value = str(tool_input.get("name", "")).strip()
            guests_count = str(tool_input.get("guests_count", "")).strip()
            if not name_value or not guests_count:
                return "Not logged — a name and guest count are required.", {"error": "missing"}

            phone = str(tool_input.get("phone", "")).strip() or None
            email = str(tool_input.get("email", "")).strip() or None

            reservation_id = db.create_reservation(
                call_id=call_id,
                name=name_value,
                phone=phone,
                email=email,
                guests_count=guests_count,
                starts_at=start,
            )

            invite_note = ""
            if email:
                reservation = db.get_reservation(reservation_id)
                result = await n8n.notify("created", reservation)
                event_id = (result or {}).get("calendar_event_id")
                if event_id:
                    db.set_calendar_event_id(reservation_id, str(event_id))
                    invite_note = " A calendar invite has been emailed to them."

            return (
                f"Reservation #{reservation_id} logged for {name_value}, "
                f"{guests_count} guests, on {start:%A %d %B} at {start:%H:%M}."
                f"{invite_note} Confirm this back to the caller and let them know "
                "the owner will have the table ready.",
                {
                    "id": reservation_id,
                    "name": name_value,
                    "guests_count": guests_count,
                    "starts_at": start.isoformat(timespec="minutes"),
                },
            )

        if name == "update_reservation":
            reservation_id = tool_input.get("reservation_id")
            reservation = db.get_reservation(int(reservation_id)) if reservation_id else None
            if reservation is None:
                return "No reservation with that id.", {"error": "not_found"}
            if reservation["status"] != "booked":
                return "That reservation is already cancelled.", {"error": "cancelled"}

            guests_count = str(tool_input.get("guests_count", "")).strip() or None
            time_raw = str(tool_input.get("requested_time", "")).strip()
            start = _parse_dt(time_raw) if time_raw else None
            if time_raw and start is None:
                return "Invalid time. Use YYYY-MM-DDTHH:MM.", {"error": "bad_time"}
            if start and start < _now_local():
                return "That time is in the past — ask for a future date/time.", {"error": "past"}

            db.update_reservation(reservation["id"], guests_count=guests_count, starts_at=start)
            updated = db.get_reservation(reservation["id"])
            if updated.get("calendar_event_id"):
                await n8n.notify("updated", updated)

            return (
                f"Reservation #{updated['id']} updated — now {updated['guests_count']} "
                f"guests at {updated['starts_at'].replace('T', ' ')}. Confirm this back "
                "to the caller.",
                {"id": updated["id"], "guests_count": updated["guests_count"],
                 "starts_at": updated["starts_at"]},
            )

        if name == "cancel_reservation":
            reservation_id = tool_input.get("reservation_id")
            reservation = db.get_reservation(int(reservation_id)) if reservation_id else None
            if reservation is None:
                return "No reservation with that id.", {"error": "not_found"}

            db.cancel_reservation(reservation["id"])
            if reservation.get("calendar_event_id"):
                await n8n.notify("cancelled", reservation)

            return (
                f"Reservation #{reservation['id']} for {reservation['name']} cancelled. "
                "Let the caller know it's taken care of.",
                {"id": reservation["id"], "cancelled": True},
            )

        if name == "find_caller_history":
            caller_name = str(tool_input.get("name", "")).strip() or None
            caller_phone = str(tool_input.get("phone", "")).strip() or None
            if not caller_name and not caller_phone:
                return "Give a name or phone number to look up.", {"error": "missing"}

            reservations = db.find_reservations(caller_name, caller_phone)
            calls = db.find_recent_calls(caller_name, caller_phone)
            if not reservations and not calls:
                return "No past reservations or calls found for this caller — treat them as new.", {
                    "reservations": 0, "calls": 0
                }

            lines = []
            if reservations:
                lines.append("Past/upcoming reservations:")
                for r in reservations:
                    lines.append(
                        f"- #{r['id']}: {r['guests_count']} guests, "
                        f"{r['starts_at'].replace('T', ' ')}, status={r['status']}"
                    )
            if calls:
                lines.append("Past call summaries:")
                for c in calls:
                    lines.append(f"- {c['started_at']}: {c['summary']}")

            return "\n".join(lines), {"reservations": len(reservations), "calls": len(calls)}

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
