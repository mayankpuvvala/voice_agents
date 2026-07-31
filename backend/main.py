"""FastAPI app: voice call WebSocket + admin API + static frontend."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import agent, db, stt
from .config import settings
from .rag import SUPPORTED_SUFFIXES, retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
log = logging.getLogger("receptionist")

app = FastAPI(title="AI Receptionist")


@app.on_event("startup")
async def _startup() -> None:
    db.init_db()
    stats = retriever.stats
    log.info(
        "Knowledge base: %s file(s), %s chars — %s",
        stats["files"],
        stats["characters"],
        "inlined in the prompt" if stats["inlined"] else f"retrieved from {stats['chunks']} chunks",
    )
    if not os.environ.get("OPENAI_API_KEY"):
        log.warning(
            "No OPENAI_API_KEY in the environment. Set it in .env — otherwise every "
            "reply will fail with an auth error."
        )
    log.info("Loading speech model '%s' (first run downloads it)...", settings.whisper_model)
    asyncio.create_task(asyncio.to_thread(stt.warm_up))


# ------------------------------------------------------------------ the call itself


class CallSession:
    """One caller: model history, plain transcript, and the DB row."""

    def __init__(self) -> None:
        self.call_id = db.start_call()
        self.history: list[dict[str, Any]] = []
        self.transcript: list[dict[str, str]] = []
        self.closed = False

    def record(self, role: str, text: str) -> None:
        self.transcript.append({"role": role, "content": text})
        db.add_message(self.call_id, role, text)

    async def finish(self) -> dict[str, Any]:
        if self.closed:
            return {}
        self.closed = True
        result = await agent.summarise(self.transcript)
        db.end_call(
            self.call_id,
            summary=result.get("summary", ""),
            caller_name=result.get("caller_name") or None,
            caller_phone=result.get("caller_phone") or None,
            follow_ups=result.get("follow_ups", []),
        )
        return result


@app.websocket("/ws/call")
async def call_socket(ws: WebSocket) -> None:
    await ws.accept()
    session = CallSession()
    log.info("Call %s started", session.call_id)

    greeting = settings.greeting
    session.record("assistant", greeting)
    await ws.send_json({"type": "ready", "call_id": session.call_id, "greeting": greeting})

    try:
        while True:
            message = await ws.receive()

            if message.get("type") == "websocket.disconnect":
                break

            user_text: str | None = None

            if message.get("bytes") is not None:
                await ws.send_json({"type": "status", "stage": "transcribing"})
                try:
                    user_text = await stt.transcribe(message["bytes"])
                except Exception as exc:  # noqa: BLE001 - report, keep the call alive
                    log.exception("Transcription failed")
                    await ws.send_json(
                        {"type": "error", "message": f"Could not transcribe audio: {exc}"}
                    )
                    continue
                if not user_text:
                    await ws.send_json({"type": "status", "stage": "idle", "note": "no_speech"})
                    continue

            elif message.get("text") is not None:
                import json

                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                kind = payload.get("type")
                if kind == "end":
                    break
                if kind == "text":
                    user_text = str(payload.get("text", "")).strip()
                    if not user_text:
                        continue
                else:
                    continue

            if user_text is None:
                continue

            await ws.send_json({"type": "transcript", "role": "user", "text": user_text})
            session.record("user", user_text)
            await ws.send_json({"type": "status", "stage": "thinking"})

            reply_parts: list[str] = []
            async for event in agent.respond(session.history, user_text, session.call_id):
                if event["type"] == "turn_end":
                    session.history = event["messages"]
                    reply = event.get("reply", "").strip()
                    if reply:
                        session.record("assistant", reply)
                        await ws.send_json(
                            {"type": "transcript", "role": "assistant", "text": reply}
                        )
                    await ws.send_json({"type": "turn_end"})
                else:
                    if event["type"] == "speech":
                        reply_parts.append(event["text"])
                    await ws.send_json(event)

    except WebSocketDisconnect:
        log.info("Call %s disconnected", session.call_id)
    except Exception:  # noqa: BLE001
        log.exception("Call %s failed", session.call_id)
    finally:
        summary = await session.finish()
        log.info("Call %s ended", session.call_id)
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "summary", **summary})
            await ws.close()


# ------------------------------------------------------------------------ admin API


@app.get("/api/config")
async def api_config() -> dict[str, Any]:
    return {
        "business_name": settings.business_name,
        "receptionist_name": settings.receptionist_name,
        "greeting": settings.greeting,
        "model": settings.model,
        "hours": settings.hours_label,
        "open_days": settings.open_days_label,
        "knowledge": retriever.stats,
    }


@app.get("/api/calls")
async def api_calls() -> list[dict[str, Any]]:
    return db.list_calls()


@app.get("/api/calls/{call_id}")
async def api_call(call_id: int) -> dict[str, Any]:
    call = db.get_call(call_id)
    if call is None:
        raise HTTPException(status_code=404, detail="No such call")
    return call


@app.get("/api/appointments")
async def api_appointments() -> list[dict[str, Any]]:
    return db.list_appointments()


@app.post("/api/appointments/{appointment_id}/cancel")
async def api_cancel(appointment_id: int) -> dict[str, str]:
    db.cancel_appointment(appointment_id)
    return {"status": "cancelled"}


@app.get("/api/notes")
async def api_notes() -> list[dict[str, Any]]:
    return db.list_notes()


@app.get("/api/knowledge")
async def api_knowledge() -> dict[str, Any]:
    return {"sources": retriever.sources(), **retriever.stats}


@app.post("/api/knowledge/reindex")
async def api_reindex() -> dict[str, Any]:
    return retriever.reindex()


@app.post("/api/knowledge/upload")
async def api_upload(file: UploadFile) -> dict[str, Any]:
    name = os.path.basename(file.filename or "")
    if not name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename")
    suffix = os.path.splitext(name)[1].lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported type '{suffix}'. Allowed: {', '.join(sorted(SUPPORTED_SUFFIXES))}",
        )
    target = settings.knowledge_dir / name
    target.write_bytes(await file.read())
    stats = retriever.reindex()
    return {"saved": name, **stats}


# ------------------------------------------------------------------------- frontend


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(settings.frontend_dir / "index.html")


@app.get("/admin")
async def admin() -> FileResponse:
    return FileResponse(settings.frontend_dir / "admin.html")


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"ok": True, "knowledge": retriever.stats})


app.mount("/static", StaticFiles(directory=settings.frontend_dir), name="static")


def run() -> None:
    import uvicorn

    uvicorn.run("backend.main:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    run()
