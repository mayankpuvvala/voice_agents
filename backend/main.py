"""FastAPI app: voice call WebSocket + admin API + static frontend."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import os
import secrets
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles

from . import agent, db, stt, tts
from .config import settings
from .rag import SUPPORTED_SUFFIXES, retriever
from .recording import CallRecorder

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
log = logging.getLogger("receptionist")

app = FastAPI(title="AI Receptionist")

_basic_auth = HTTPBasic(auto_error=False)


def require_admin(
    credentials: Annotated[HTTPBasicCredentials | None, Depends(_basic_auth)] = None,
) -> None:
    """Gate the admin dashboard and API behind HTTP Basic auth.

    No-ops when ADMIN_USERNAME/ADMIN_PASSWORD aren't set, which is only
    acceptable for local development on 127.0.0.1 — see the startup warning.
    """
    if not settings.auth_required:
        return
    valid = bool(credentials) and secrets.compare_digest(
        credentials.username, settings.admin_username
    ) and secrets.compare_digest(credentials.password, settings.admin_password)
    if not valid:
        raise HTTPException(
            status_code=401,
            detail="Admin credentials required.",
            headers={"WWW-Authenticate": "Basic"},
        )


AdminAuth = Depends(require_admin)


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
    if not settings.auth_required:
        log.warning(
            "ADMIN_USERNAME/ADMIN_PASSWORD not set — the admin dashboard and API are "
            "wide open. Fine on 127.0.0.1, not once this is deployed publicly."
        )
    log.info("Loading speech model '%s' (first run downloads it)...", settings.whisper_model)
    asyncio.create_task(asyncio.to_thread(stt.warm_up))
    if settings.tts_enabled and settings.tts_provider == "piper":
        log.info("Loading Piper voices (first run downloads them)...")
        asyncio.create_task(asyncio.to_thread(tts.warm_up))


# ------------------------------------------------------------------ the call itself


class CallSession:
    """One caller: model history, plain transcript, the DB row, and whatever
    turn is currently being generated (so a barge-in has something to cancel).
    """

    def __init__(self) -> None:
        self.call_id = db.start_call()
        self.history: list[dict[str, Any]] = []
        self.transcript: list[dict[str, str]] = []
        self.closed = False
        self.current_task: asyncio.Task | None = None
        self.recorder = CallRecorder(self.call_id) if settings.call_recording_enabled else None

    def record(self, role: str, text: str) -> None:
        self.transcript.append({"role": role, "content": text})
        db.add_message(self.call_id, role, text)

    async def cancel_current(self) -> bool:
        """Cancel the in-flight turn, if any. Returns whether one was running."""
        task = self.current_task
        if task is None or task.done():
            return False
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        return True

    async def finish(self) -> dict[str, Any]:
        if self.closed:
            return {}
        self.closed = True
        result = await agent.summarise(self.transcript)

        recording_path: str | None = None
        recording_seconds: float | None = None
        if self.recorder is not None:
            try:
                saved = await asyncio.to_thread(self.recorder.save)
            except Exception:
                log.exception("Call %s: failed to save the recording", self.call_id)
            else:
                if saved is not None:
                    recording_path = saved.name
                    recording_seconds = self.recorder.duration_seconds

        db.end_call(
            self.call_id,
            summary=result.get("summary", ""),
            caller_name=result.get("caller_name") or None,
            caller_phone=result.get("caller_phone") or None,
            follow_ups=result.get("follow_ups", []),
            recording_path=recording_path,
            recording_seconds=recording_seconds,
        )
        return result


@app.websocket("/ws/call")
async def call_socket(ws: WebSocket) -> None:
    await ws.accept()
    session = CallSession()
    log.info("Call %s started", session.call_id)

    # A reply is generated as a background task rather than awaited inline,
    # so the loop below stays free to read the caller's next message — audio
    # bytes or an explicit "interrupt" — while a previous turn is still
    # streaming. Both this loop and the task send over the same socket, so
    # every send is serialized through this lock to keep frames from
    # interleaving.
    send_lock = asyncio.Lock()

    async def send(payload: dict[str, Any]) -> None:
        async with send_lock:
            await ws.send_json(payload)

    async def run_turn(user_text: str) -> None:
        try:
            await send({"type": "transcript", "role": "user", "text": user_text})
            session.record("user", user_text)
            await send({"type": "status", "stage": "thinking"})

            async for event in agent.respond(session.history, user_text, session.call_id):
                if event["type"] == "turn_end":
                    session.history = event["messages"]
                    reply = event.get("reply", "").strip()
                    if reply:
                        session.record("assistant", reply)
                        await send({"type": "transcript", "role": "assistant", "text": reply})
                    await send({"type": "turn_end"})
                else:
                    if event["type"] == "audio" and session.recorder is not None:
                        session.recorder.add(base64.b64decode(event["data"]))
                    await send(event)
        except asyncio.CancelledError:
            log.info("Call %s: turn interrupted by caller", session.call_id)
            raise
        except Exception:
            log.exception("Call %s: turn failed", session.call_id)
            reply = "Sorry, something went wrong on my end — could you say that again?"
            session.record("assistant", reply)
            await send({"type": "transcript", "role": "assistant", "text": reply})
            await send({"type": "turn_end"})
        finally:
            session.current_task = None

    greeting = settings.greeting
    session.record("assistant", greeting)
    await send({
        "type": "ready",
        "call_id": session.call_id,
        "greeting": greeting,
        "tts_enabled": settings.tts_enabled,
    })

    if settings.tts_enabled:
        try:
            greeting_audio = await asyncio.to_thread(tts.synthesize, greeting)
        except Exception:
            log.exception(
                "Call %s: TTS synthesis failed for the greeting — falling back "
                "to the browser voice", session.call_id
            )
        else:
            if session.recorder is not None:
                session.recorder.add(greeting_audio)
            await send({
                "type": "audio",
                "data": base64.b64encode(greeting_audio).decode("ascii"),
            })

    try:
        while True:
            message = await ws.receive()

            if message.get("type") == "websocket.disconnect":
                break

            user_text: str | None = None

            if message.get("bytes") is not None:
                # Recorded regardless of what STT makes of it below — the
                # recording should reflect what was actually said on the
                # call, not just the utterances that transcribed cleanly.
                if session.recorder is not None:
                    session.recorder.add(message["bytes"])
                # New audio always wins over whatever reply is in flight —
                # belt-and-suspenders alongside the explicit "interrupt"
                # message below, in case that one was dropped or delayed.
                await session.cancel_current()
                await send({"type": "status", "stage": "transcribing"})
                try:
                    user_text = await stt.transcribe(message["bytes"])
                except Exception as exc:  # noqa: BLE001 - report, keep the call alive
                    log.exception("Transcription failed")
                    await send(
                        {"type": "error", "message": f"Could not transcribe audio: {exc}"}
                    )
                    continue
                if not user_text:
                    await send({"type": "status", "stage": "idle", "note": "no_speech"})
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
                if kind == "interrupt":
                    if await session.cancel_current():
                        await send({"type": "status", "stage": "idle", "note": "interrupted"})
                continue

            if user_text is None:
                continue

            session.current_task = asyncio.create_task(run_turn(user_text))

    except WebSocketDisconnect:
        log.info("Call %s disconnected", session.call_id)
    except Exception:  # noqa: BLE001
        log.exception("Call %s failed", session.call_id)
    finally:
        await session.cancel_current()
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
        "timezone": settings.timezone_label,
        "knowledge": retriever.stats,
    }


@app.get("/api/calls", dependencies=[AdminAuth])
async def api_calls() -> list[dict[str, Any]]:
    return db.list_calls()


@app.get("/api/calls/{call_id}", dependencies=[AdminAuth])
async def api_call(call_id: int) -> dict[str, Any]:
    call = db.get_call(call_id)
    if call is None:
        raise HTTPException(status_code=404, detail="No such call")
    return call


@app.get("/api/calls/{call_id}/recording", dependencies=[AdminAuth])
async def api_call_recording(call_id: int) -> FileResponse:
    call = db.get_call(call_id)
    if call is None or not call.get("recording_path"):
        raise HTTPException(status_code=404, detail="No recording for this call")
    path = settings.recordings_dir / call["recording_path"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="Recording file is missing on disk")
    return FileResponse(path, media_type="audio/wav", filename=f"call-{call_id}.wav")


@app.get("/api/reservations", dependencies=[AdminAuth])
async def api_reservations() -> list[dict[str, Any]]:
    return db.list_reservations()


@app.post("/api/reservations/{reservation_id}/cancel", dependencies=[AdminAuth])
async def api_cancel(reservation_id: int) -> dict[str, str]:
    db.cancel_reservation(reservation_id)
    return {"status": "cancelled"}


@app.get("/api/notes", dependencies=[AdminAuth])
async def api_notes() -> list[dict[str, Any]]:
    return db.list_notes()


@app.get("/api/knowledge", dependencies=[AdminAuth])
async def api_knowledge() -> dict[str, Any]:
    return {"sources": retriever.sources(), **retriever.stats}


@app.post("/api/knowledge/reindex", dependencies=[AdminAuth])
async def api_reindex() -> dict[str, Any]:
    return retriever.reindex()


@app.post("/api/knowledge/upload", dependencies=[AdminAuth])
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


@app.get("/admin", dependencies=[AdminAuth])
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
