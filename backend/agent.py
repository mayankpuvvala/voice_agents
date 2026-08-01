"""The receptionist brain: OpenAI + RAG context + tools, streamed for low-latency speech."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, AsyncIterator
from zoneinfo import ZoneInfo

import openai
from openai import AsyncOpenAI

from . import tools as toolkit
from . import tts
from .config import settings
from .rag import retriever

log = logging.getLogger("receptionist.agent")

# Short replies keep speech snappy; spoken turns are a couple of sentences.
MAX_TOKENS = 1024

AUTH_HINT = (
    "No OpenAI credentials found. Put OPENAI_API_KEY in .env and restart the server."
)


class MissingCredentials(RuntimeError):
    """Raised instead of blowing up at import time when no API key is configured."""


_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    """Construct the client lazily — AsyncOpenAI() raises if the key is absent."""
    global _client
    if _client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise MissingCredentials(AUTH_HINT)
        # base_url is picked up from OPENAI_BASE_URL when set, so Azure and
        # OpenAI-compatible gateways work without code changes.
        _client = AsyncOpenAI()
    return _client


def system_prompt() -> str:
    """Stable across every turn and every call.

    OpenAI caches long prompt prefixes automatically, so keeping this
    byte-identical between requests is what makes the discount apply. Nothing
    volatile (no clock, no caller id) may go in here. A small knowledge base is
    inlined rather than retrieved, which removes retrieval misses entirely.
    """
    corpus = retriever.full_text()
    if corpus:
        knowledge = (
            "\n\n# Knowledge base\n"
            "Everything the business has told you is below. Answer factual questions "
            "from it and nothing else. If it is not here, say you will have someone "
            "follow up and save a note.\n\n" + corpus
        )
    else:
        outline = retriever.outline()
        knowledge = (
            "\n\n# Knowledge base\n"
            "Use search_knowledge_base to read the business's documents. Sections "
            "available:\n" + outline
            if outline
            else "\n\n# Knowledge base\nNo documents have been loaded. You cannot answer "
            "factual questions about the business — take a message instead."
        )

    return f"""You are {settings.receptionist_name}, the receptionist for {settings.business_name}. \
You are speaking with a caller on the phone. Everything you say is read aloud, so write \
for the ear, not the page.

# Language
Callers speak either English or Hindi. Answer in whichever language the caller is currently \
using — read their last message and match it. If they switch language mid-call, switch with \
them on your very next reply. Do not mix both languages in a single reply unless the caller \
just did.

# How to speak
- Short turns. One or two sentences unless the caller asked for detail.
- Plain spoken language. No bullet points, headings, markdown, emoji, or URLs.
- Say numbers, times and dates the way a person says them out loud: "two thirty on Tuesday \
the fourth", not "2:30 PM 2024-06-04".
- Never say you are an AI unless the caller asks directly. Then say so plainly and carry on.
- One question at a time. Do not stack three questions into one turn.
- If you did not catch something, say so and ask them to repeat it.

# What you can do
- Answer questions about the business using search_knowledge_base.
- Log a table reservation with create_reservation.
- Record messages, callbacks and complaints with save_note.

# Rules that matter
- Answer factual questions about {settings.business_name} from the knowledge base, not from \
memory. If the knowledge base does not cover it, say you will have someone follow up, and \
save a note — do not guess at prices, policies, or availability.
- For a reservation you need a name, a guest count, and a requested date/time — a phone \
number is not required. There is no capacity limit to check: confirm the details back once, \
call create_reservation, then let the caller know the owner will have the table ready.
- Call save_note or create_reservation the moment a topic is resolved, not at the end of the \
call — callers often hang up with no goodbye, and anything you were planning to log "later" \
simply never gets logged if that happens.
- If a caller pushes back, repeats, or rephrases a request after you've already declined it \
per policy (asking again for something outside the knowledge base, or an exception you can't \
grant), treat that as real interest the owner should know about: say you can't do it yourself \
but will pass it along, and save a note with their name and callback number — even though your \
answer to them stays the same.{knowledge}"""


# Trailing "." here is a sentence boundary only by accident, so never split on it.
_ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "st", "approx", "no", "vs",
    "etc", "e.g", "i.e", "a.m", "p.m", "am", "pm",
}


def _sentences(buffer: str) -> tuple[list[str], str]:
    """Pull complete sentences off the front of the buffer so speech can start early.

    Only splits on terminal punctuation that is followed by whitespace — a
    boundary at the very end of the buffer is left alone, because more text may
    still be streaming in. The caller flushes the remainder when the turn ends.
    Includes the Hindi/Devanagari full stop "।" alongside the Latin terminators.
    """
    out: list[str] = []
    start = 0
    for idx, char in enumerate(buffer):
        if char == "\n":
            piece = buffer[start:idx].strip()
            if len(piece) > 1:
                out.append(piece)
                start = idx + 1
            continue

        if char not in ".!?…।":
            continue

        following = buffer[idx + 1 : idx + 2]
        if not following or not following.isspace():
            continue  # end of buffer, or a decimal point like "2.30"

        word = re.search(r"([A-Za-z.]+)$", buffer[start:idx])
        if word and word.group(1).lower().strip(".") in _ABBREVIATIONS:
            continue

        piece = buffer[start : idx + 1].strip()
        if len(piece) > 1:
            out.append(piece)
            start = idx + 1

    return out, buffer[start:].lstrip()


def build_user_turn(text: str) -> str:
    """User speech plus the retrieved context and clock, kept out of the cached prefix."""
    now = datetime.now(ZoneInfo(settings.business_timezone))
    # When the corpus is inlined in the system prompt there is nothing to retrieve.
    context = "" if retriever.full_text() else retriever.context_for(text, k=3)
    parts: list[str] = []
    if context:
        parts.append(
            "Possibly relevant extracts from the knowledge base (the caller cannot see "
            "these; use them only if they actually answer the question, and search "
            "again if they do not):\n\n" + context
        )
    parts.append(
        f"Current date and time: {now:%A %d %B %Y, %H:%M} ({settings.timezone_label})."
    )
    parts.append(f"Caller said: {text}")
    return "\n\n".join(parts)


def _request_kwargs(messages: list[dict[str, Any]]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": settings.model,
        "messages": [{"role": "system", "content": system_prompt()}] + messages,
        "tools": toolkit.TOOLS,
        "tool_choice": "auto",
        "max_completion_tokens": MAX_TOKENS,
    }
    # Reasoning models reject temperature, so it is opt-out via an empty setting.
    if settings.temperature is not None:
        kwargs["temperature"] = settings.temperature
    return kwargs


async def _audio_event(text: str) -> dict[str, Any] | None:
    """Synthesize `text` with Piper and wrap it as a websocket event.

    Returns None (rather than raising) on any failure or when TTS is turned
    off, so the frontend just falls back to its own speechSynthesis for that
    sentence instead of the turn breaking.
    """
    if not settings.tts_enabled:
        return None
    try:
        audio = await asyncio.to_thread(tts.synthesize, text)
    except Exception:
        log.exception("TTS synthesis failed for a sentence — falling back to browser voice")
        return None
    return {"type": "audio", "data": base64.b64encode(audio).decode("ascii")}


async def respond(
    history: list[dict[str, Any]],
    user_text: str,
    call_id: int | None,
) -> AsyncIterator[dict[str, Any]]:
    """Drive one caller turn to completion.

    Yields UI/audio events and, last, a `turn_end` carrying the updated history.
    """
    messages = history + [{"role": "user", "content": build_user_turn(user_text)}]
    spoken: list[str] = []

    def finish() -> dict[str, Any]:
        return {"type": "turn_end", "messages": messages, "reply": " ".join(spoken)}

    async def say(sentence: str) -> AsyncIterator[dict[str, Any]]:
        spoken.append(sentence)
        yield {"type": "speech", "text": sentence}
        audio_event = await _audio_event(sentence)
        if audio_event:
            yield audio_event

    while True:
        buffer = ""
        text_out = ""
        # Streamed tool calls arrive in fragments keyed by index.
        partial: dict[int, dict[str, str]] = {}
        finish_reason: str | None = None

        try:
            client = get_client()
            stream = await client.chat.completions.create(
                stream=True, **_request_kwargs(messages)
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                delta = choice.delta
                if delta is None:
                    continue

                if delta.content:
                    buffer += delta.content
                    text_out += delta.content
                    ready, buffer = _sentences(buffer)
                    for sentence in ready:
                        async for event in say(sentence):
                            yield event

                for fragment in delta.tool_calls or []:
                    entry = partial.setdefault(
                        fragment.index, {"id": "", "name": "", "arguments": ""}
                    )
                    if fragment.id:
                        entry["id"] = fragment.id
                    if fragment.function and fragment.function.name:
                        entry["name"] += fragment.function.name
                    if fragment.function and fragment.function.arguments:
                        entry["arguments"] += fragment.function.arguments

        except MissingCredentials as exc:
            yield {"type": "error", "message": str(exc)}
            yield finish()
            return
        except openai.AuthenticationError:
            yield {"type": "error", "message": AUTH_HINT}
            yield finish()
            return
        except openai.NotFoundError:
            yield {
                "type": "error",
                "message": (
                    f"Model '{settings.model}' is not available on this key. "
                    "Set OPENAI_MODEL in .env to one you have access to."
                ),
            }
            yield finish()
            return
        except openai.RateLimitError:
            yield {"type": "error", "message": "Rate limited by OpenAI — try again shortly."}
            yield finish()
            return
        except openai.APIStatusError as exc:
            log.warning("OpenAI returned %s: %s", exc.status_code, exc.message)
            yield {"type": "error", "message": f"Model error ({exc.status_code}). Please try again."}
            yield finish()
            return
        except openai.APIConnectionError:
            yield {"type": "error", "message": "Lost the connection to OpenAI."}
            yield finish()
            return
        except Exception as exc:  # noqa: BLE001 - a live call must never drop the socket
            log.exception("Turn failed")
            yield {"type": "error", "message": f"Something went wrong handling that turn: {exc}"}
            yield finish()
            return

        if buffer.strip():
            async for event in say(buffer.strip()):
                yield event

        if finish_reason == "content_filter":
            line = "I'm not able to help with that one, but I can take a message for the team."
            async for event in say(line):
                yield event
            yield finish()
            return

        calls = [partial[i] for i in sorted(partial) if partial[i]["name"]]

        assistant_message: dict[str, Any] = {"role": "assistant", "content": text_out or None}
        if calls:
            assistant_message["tool_calls"] = [
                {
                    "id": c["id"],
                    "type": "function",
                    "function": {"name": c["name"], "arguments": c["arguments"] or "{}"},
                }
                for c in calls
            ]
        messages = messages + [assistant_message]

        if not calls:
            break

        for call in calls:
            try:
                tool_input = json.loads(call["arguments"] or "{}")
                if not isinstance(tool_input, dict):
                    raise ValueError("arguments were not an object")
            except (json.JSONDecodeError, ValueError) as exc:
                result_text = f"Could not read the arguments for {call['name']}: {exc}"
                payload: dict[str, Any] = {"error": "bad_arguments"}
                tool_input = {}
            else:
                result_text, payload = toolkit.execute(call["name"], tool_input, call_id)

            yield {"type": "tool", "name": call["name"], "input": tool_input, "result": payload}
            messages = messages + [
                {"role": "tool", "tool_call_id": call["id"], "content": result_text}
            ]

    yield finish()


async def summarise(transcript: list[dict[str, str]]) -> dict[str, Any]:
    """Post-call notes: summary, caller details, and follow-up actions."""
    blank = {"summary": "", "caller_name": "", "caller_phone": "", "follow_ups": []}
    if not transcript:
        return {**blank, "summary": "No conversation took place."}

    lines = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in transcript)
    instructions = (
        f"You write post-call notes for {settings.business_name}. Be factual and concise. "
        "Write the notes in English regardless of what language the call was conducted in. "
        "Only list a follow-up if a person actually has to do something; a reservation "
        "already logged during the call is not a follow-up."
    )
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": f"Call transcript:\n\n{lines}"},
    ]
    base: dict[str, Any] = {"model": settings.model, "messages": messages,
                            "max_completion_tokens": 800}
    if settings.temperature is not None:
        base["temperature"] = settings.temperature

    strict_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "call_summary",
            "strict": True,
            "schema": toolkit.summary_schema(),
        },
    }

    try:
        client = get_client()
        try:
            response = await client.chat.completions.create(
                response_format=strict_format, **base
            )
        except openai.BadRequestError:
            # Older models reject json_schema; plain JSON mode still works.
            log.info("json_schema unsupported on %s — falling back to JSON mode",
                     settings.model)
            messages[0]["content"] = (
                instructions
                + " Reply with a JSON object with keys: summary (string), caller_name "
                "(string), caller_phone (string), follow_ups (array of strings)."
            )
            response = await client.chat.completions.create(
                response_format={"type": "json_object"}, **base
            )
    except Exception as exc:  # noqa: BLE001 - never let cleanup break the hang-up path
        log.warning("Post-call summary failed: %s", exc)
        return {**blank, "summary": "Automatic summary unavailable — see the transcript."}

    message = response.choices[0].message
    if getattr(message, "refusal", None):
        return {**blank, "summary": "Summary withheld — see the transcript."}

    return toolkit.parse_summary(message.content or "")
