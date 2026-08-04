"""Notifies an n8n workflow of reservation events (create/update/cancel) so
it can sync a Google Calendar event and let Google email the caller an
invite. Entirely optional — every call site treats a failure here as
non-fatal, since the reservation itself must never be lost just because the
calendar sync is unreachable or unconfigured.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import httpx

from .config import settings

log = logging.getLogger("receptionist.n8n")


async def notify(event: str, reservation: dict[str, Any]) -> dict[str, Any] | None:
    """POST a reservation event to n8n. Returns its JSON response, or None on
    any failure (not configured, network error, non-2xx, bad JSON) — callers
    should treat None as "no calendar sync happened this time" and continue.
    """
    url = settings.n8n_reservation_webhook_url
    if not url:
        return None

    starts_at = reservation.get("starts_at")
    if isinstance(starts_at, datetime):
        starts_at = starts_at.isoformat(timespec="minutes")

    payload = {
        "event": event,  # "created" | "updated" | "cancelled"
        "reservation_id": reservation.get("id"),
        "calendar_event_id": reservation.get("calendar_event_id"),
        "name": reservation.get("name"),
        "email": reservation.get("email"),
        "phone": reservation.get("phone"),
        "guests_count": reservation.get("guests_count"),
        "starts_at": starts_at,
        "business_name": settings.business_name,
        "business_timezone": settings.business_timezone,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
    except Exception:
        log.exception("n8n reservation webhook (%s) failed for reservation %s",
                      event, reservation.get("id"))
        return None
