# AI Receptionist — Spice Route Kitchen

A browser-based voice receptionist for a restaurant. A caller speaks into
their microphone; the app transcribes it, answers using OpenAI grounded in
the restaurant's own facts, replies out loud in English or Hindi, logs
reservations and messages, and lets the owner browse it all from a
dashboard.

```
browser mic ──> WebSocket ──> STT ──> OpenAI chat ──> sentences ──> Piper TTS ──> browser
                              (Groq Whisper,          + your facts     │           audio
                               local fallback)         + tools ────────┴──> SQLite
                                                                            (calls, reservations,
                                                                             notes, transcripts)
```

No phone number, no telephony account yet — that's a deliberately separate,
later step. Everything above already runs for real over a browser mic, which
is enough to demo the full flow end to end.

---

## Quick start

```powershell
# 1. Get an OpenAI key (https://platform.openai.com/api-keys) and a free Groq
#    key (https://console.groq.com/keys), put them in .env

# 2. Run
.\run.ps1
```

Then open <http://127.0.0.1:8000>, click **Start call**, allow the
microphone, and talk. The dashboard is at <http://127.0.0.1:8000/admin>.

First run downloads the local Whisper fallback model and both Piper voices
in the background; the first call or two may wait a few seconds on it.

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

**`.env`** — API keys, model, name, greeting, timezone, admin credentials. See
`.env.example` for every option and what each one does, including how to
point the chat model at Groq instead of OpenAI for $0 spend end to end.

**`knowledge/`** — drop in `.md`, `.txt` or `.pdf` files. This is the only
thing the receptionist answers factual questions from. Delete
`knowledge/business-info.md` and put your own restaurant's facts there, then
restart (or hit **Re-index** on the dashboard). You can also upload files
from the dashboard's Knowledge base tab.

If the receptionist starts inventing prices or policies, the fix is almost
always a gap in `knowledge/`, not the prompt.

---

## How it works

### Turn taking

The browser does the endpointing. An `AnalyserNode` measures microphone
level, calibrates the room's noise floor for the first ~0.9 s, then waits for
speech to start and for 1.1 s of silence to end it. `MediaRecorder` captures
exactly that utterance and the complete WebM/Opus blob goes over the
WebSocket in one piece.

If auto-detection misfires in a noisy room, **Hold to talk** overrides it,
and there is a text box for typing instead of speaking.

### Speech to text

Two providers, selected by `STT_PROVIDER`:

- **Groq-hosted Whisper** (`whisper-large-v3-turbo`, the default) — free
  tier, multilingual (English and Hindi both covered), and it offloads
  transcription off this process entirely.
- **Local `faster-whisper`** — the offline fallback, used automatically if
  Groq errors or rate-limits, or directly if `STT_PROVIDER=local`. Set
  `WHISPER_MODEL` to a multilingual model (`small`, `medium` — not the
  `.en`-suffixed ones, which are English-only) since this app needs Hindi
  too.

### Text to speech

Self-hosted **Piper**, free at any call volume — the frontend's own
`speechSynthesis` is only used as a fallback when `TTS_ENABLED=false` or a
voice fails to load. The voice is chosen per reply by checking whether the
generated text is in Devanagari script (Hindi) or not (English), rather than
trusting whichever language the caller spoke — that way the voice can never
mismatch the text it's reading, even if the model doesn't follow the
"reply in the caller's language" instruction. Sentences are synthesized as
soon as each one is complete, reusing the existing sentence-streaming seam,
so speech starts before the model has finished writing the whole reply.

### Grounding (RAG)

Two modes, picked automatically by corpus size:

- **Under 20 000 characters** — the whole knowledge base goes into the system
  prompt. Perfect recall, no retrieval misses, and prompt caching makes the
  repeated tokens cost about a tenth of normal input after the first turn.
- **Larger** — documents are chunked and retrieved with TF-IDF over word
  bigrams *and* character n-grams (so "park" matches "parking"). Every miss
  returns the outline of available sections so the model can re-query with
  better words.

### The model call

OpenAI chat completions with streaming and function tools. `OPENAI_BASE_URL`
can point this at Azure, Groq, or any OpenAI-compatible gateway without a
code change — see `.env.example` for running the chat model at $0 on Groq
too.

### Tools

| Tool | What it does |
|---|---|
| `search_knowledge_base` | Look something up in the restaurant's documents |
| `create_reservation` | Logs a table reservation — name, guest count, requested time. No capacity or conflict check: the owner manages the physical table, not the bot |
| `save_note` | Messages, callbacks, complaints — including when a caller pushes back on something already declined |

The system prompt tells the model to call `save_note`/`create_reservation`
the moment a topic is resolved, not at the end of the call, since a real
caller can hang up with no goodbye.

**Not implemented, on purpose:** emergency-language handling. It was in the
original template's prompt and has been deliberately left out of this build
until asked for explicitly.

When the caller hangs up, a second call produces a structured post-call
summary (summary, caller name and number, follow-up actions) using a strict
`response_format` JSON schema, falling back to plain JSON mode on models
that do not support schemas.

### Storage

SQLite at `data/receptionist.db` — `calls`, `messages`, `reservations`,
`notes`. Browse it all from the dashboard.

### Admin dashboard

Gated behind HTTP Basic Auth once `ADMIN_USERNAME`/`ADMIN_PASSWORD` are set
in `.env` — the browser prompts natively, no extra UI needed. Leaving both
blank disables auth entirely, which is only fine for local development; the
server logs a warning at startup if it's left that way.

---

## Layout

```
backend/
  main.py     FastAPI app, call WebSocket, admin API, auth
  agent.py    OpenAI conversation, streaming, tool loop, post-call summary
  tools.py    Tool schemas + executors (search, reservations, notes)
  rag.py      Document loading, chunking, retrieval
  stt.py      Groq Whisper + local faster-whisper fallback
  tts.py      Piper multilingual text-to-speech
  db.py       SQLite schema and queries
  config.py   .env loading
frontend/
  index.html / app.js     Call console: VAD, recording, audio playback
  admin.html / admin.js   Dashboard
  styles.css
knowledge/    Your documents — the only source of factual answers
data/         SQLite database + downloaded Piper voices (created on first run)
```

---

## Limits worth knowing

- **One caller at a time is the tested path.** Multiple concurrent
  WebSockets will work but share one local Whisper model instance if the
  Groq fallback path is in use, so transcription serialises in that case.
- **No barge-in.** The microphone is gated while the receptionist is
  speaking; interrupting means pressing Hold to talk.
- **English and Hindi only.** Telugu was considered and dropped for this
  pass — revisit once Piper's Telugu voice quality has been checked
  hands-on.
- **No emergency-language handling** — see "Tools" above; intentionally out
  of scope until asked for.
- **No real phone number yet.** This is a browser-mic build; bridging real
  inbound calls (Twilio Media Streams or similar) is a deliberately separate
  next step, not started here.
- **Times use the restaurant's configured `BUSINESS_TIMEZONE`**, not the
  server's local time — set it correctly in `.env` for the venue's actual
  location.
