# AI Receptionist

A browser-based voice receptionist. A visitor speaks into their microphone; the
app transcribes locally, answers using OpenAI grounded in your own documents,
replies out loud, takes notes, and books appointments into a local database.

```
browser mic ──> WebSocket ──> faster-whisper ──> OpenAI chat ──> sentences ──> speechSynthesis
                                (local STT)     + your documents     │
                                                + tools ────────────┴──> SQLite
                                                                         (appointments, notes,
                                                                          transcripts)
```

No phone number, no telephony account, no cloud speech service. Speech-to-text
runs locally, so the only external call is the chat completion.

---

## Quick start

```powershell
# 1. Get an API key from https://platform.openai.com/api-keys
#    and put it in .env as OPENAI_API_KEY=sk-...

# 2. Run
.\run.ps1
```

Then open <http://127.0.0.1:8000>, click **Start call**, allow the microphone,
and talk. The dashboard is at <http://127.0.0.1:8000/admin>.

First run downloads the Whisper model (~150 MB) in the background; the first
call may wait a few seconds on it.

> **Use headphones.** Without them the speaker output is picked up by the
> microphone and the receptionist starts answering itself.

### Manual setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env    # then edit it
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --port 8000
```

---

## Making it yours

Everything business-specific lives in two places.

**`.env`** — API key, model, name, greeting, opening hours, appointment length,
Whisper model size. `OPENAI_MODEL` defaults to `gpt-4o`; set it to whatever your
key can reach (you get a clear in-call error on a 404). Blank `OPENAI_TEMPERATURE`
to omit the parameter entirely, which reasoning models require. `OPENAI_BASE_URL`
points the whole app at Azure OpenAI or any OpenAI-compatible gateway.

**`knowledge/`** — drop in `.md`, `.txt` or `.pdf` files. This is the only thing
the receptionist answers factual questions from. Delete
`knowledge/business-info.md` and put your own documents there, then restart (or
hit **Re-index** on the dashboard). You can also upload files from the
dashboard's Knowledge base tab.

If the receptionist starts inventing prices or policies, the fix is almost
always a gap in `knowledge/`, not the prompt.

---

## How it works

### Turn taking

The browser does the endpointing. An `AnalyserNode` measures microphone level,
calibrates the room's noise floor for the first ~0.9 s, then waits for speech to
start and for 1.1 s of silence to end it. `MediaRecorder` captures exactly that
utterance and the complete WebM/Opus blob goes over the WebSocket in one piece —
so every blob is independently decodable, which is what makes local Whisper
practical without a streaming ASR service.

If auto-detection misfires in a noisy room, **Hold to talk** overrides it, and
there is a text box for typing instead of speaking.

### Speech to text

`faster-whisper` on CPU, `base.en` by default, with its own VAD filter as a
second pass. Set `WHISPER_MODEL=small.en` in `.env` for better accuracy at
roughly 2–3× the latency. It bundles PyAV, so no ffmpeg install is needed.

This stays local deliberately — it keeps audio off the network and costs
nothing. If you would rather use OpenAI's hosted transcription now that you have
a key, `backend/stt.py` is the only file that changes: swap `_transcribe_sync`
for a call to `client.audio.transcriptions.create`.

### Grounding (RAG)

Two modes, picked automatically by corpus size:

- **Under 20 000 characters** — the whole knowledge base goes into the system
  prompt. Perfect recall, no retrieval misses, and prompt caching makes the
  repeated tokens cost about a tenth of normal input after the first turn.
- **Larger** — documents are chunked and retrieved with TF-IDF over word bigrams
  *and* character n-grams (so "park" matches "parking"). Every miss returns the
  outline of available sections so the model can re-query with better words.

Lexical retrieval cannot bridge true synonyms — "braces" will not find a section
that only says "orthodontics". If you outgrow the inline mode and that starts to
bite, replace `Retriever` in `backend/rag.py` with dense embeddings; nothing
else in the app touches retrieval.

### The model call

OpenAI chat completions with streaming and function tools. Complete sentences
are pushed to the browser as they finish, so speech starts before the model has
finished writing.

The system prompt is byte-identical on every request. OpenAI caches long prompt
prefixes automatically, and the knowledge base sits inside that prefix — so the
clock and any retrieved context deliberately go in the *user* turn instead.
Putting them in the system prompt would change the prefix every request and
silently forfeit the discount.

Streamed tool calls arrive as fragments keyed by index, so they are accumulated
(`id`, name, and a JSON argument string built up across chunks) and only
executed once the stream closes. Malformed arguments are reported back to the
model as a tool result rather than raising, which lets it correct itself.
`finish_reason: "content_filter"` is handled as a spoken fallback line.

The client is constructed lazily — `AsyncOpenAI()` raises when no key is set, so
building it at import time would stop the server from starting at all.

### Tools

| Tool | What it does |
|---|---|
| `search_knowledge_base` | Look something up in your documents |
| `check_availability` | Free slots on a date, honouring opening hours and existing bookings |
| `book_appointment` | Writes to SQLite; rejects past times, out-of-hours times, and double-bookings |
| `save_note` | Messages, callbacks, complaints |

Booking rules live in `backend/tools.py`: past times, out-of-hours times and
overlapping bookings are rejected there, not left to the model, and a rejection
comes back with the alternative slots so it can offer them.

When the caller hangs up, a second call produces a structured post-call summary
(summary, caller name and number, follow-up actions) using a strict
`response_format` JSON schema, falling back to plain JSON mode on models that
do not support schemas.

### Storage

SQLite at `data/receptionist.db` — `calls`, `messages`, `appointments`, `notes`.
Browse it all from the dashboard.

---

## Layout

```
backend/
  main.py     FastAPI app, call WebSocket, admin API
  agent.py    OpenAI conversation, streaming, tool loop, post-call summary
  tools.py    Tool schemas + executors (booking rules live here)
  rag.py      Document loading, chunking, retrieval
  stt.py      faster-whisper wrapper
  db.py       SQLite schema and queries
  config.py   .env loading
frontend/
  index.html / app.js     Call console: VAD, recording, TTS
  admin.html / admin.js   Dashboard
  styles.css
knowledge/    Your documents — the only source of factual answers
data/         SQLite database (created on first run)
```

---

## What has been tested

A full four-turn conversation was run live against `gpt-4o`, covering:

- parallel streamed tool calls in a single turn (two knowledge lookups at once)
- grounded answers, including correctly declining a service the business does
  not offer and naming the referral partner instead
- `check_availability` before any time was offered
- a booking and a callback note written in the same turn
- the structured post-call summary, with caller name, number and follow-up
  extracted
- persistence: nine transcript messages, one appointment and one note in SQLite

Also verified: server start-up, graceful behaviour with no API key (the socket
survives and reports it), retrieval, the sentence splitter, booking rejection
for conflicts / out-of-hours / past times, the dashboard API, and the Whisper
load-and-decode path.

**Not exercised:** real microphone audio through a browser. The audio path was
tested with a synthetic file, so the speech pipeline decodes and transcribes,
but the in-browser voice-activity detection thresholds have only been reasoned
about, not tuned against a real room. Expect to adjust the `VAD` constants at
the top of `frontend/app.js` if it clips you or waits too long.

## Limits worth knowing

- **One caller at a time is the tested path.** Multiple concurrent WebSockets
  will work but share one Whisper model instance, so transcription serialises.
- **No barge-in.** The microphone is gated while the receptionist is speaking;
  interrupting means pressing Hold to talk.
- **`speechSynthesis` voice quality varies by browser and OS.** Edge on Windows
  has the best built-in voices. Now that you have an OpenAI key, swapping in
  their TTS is the single biggest quality win available: replace `speak()` in
  `frontend/app.js` and stream the audio back over the existing socket.
- **No authentication on the dashboard.** It is bound to `127.0.0.1` — do not
  expose it as-is.
- **Times are naive local times**, with no timezone handling or DST awareness.
