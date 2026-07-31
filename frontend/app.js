/* AI Receptionist — call console.
 *
 * Turn taking is done in the browser: an AnalyserNode watches the mic level,
 * MediaRecorder captures one utterance at a time, and the complete WebM blob is
 * pushed over the WebSocket when the caller stops speaking. Replies stream back
 * sentence by sentence and are spoken with the Web Speech API.
 */

const $ = (id) => document.getElementById(id);

const el = {
  orb: $("orb"),
  status: $("status"),
  level: $("level-bar"),
  start: $("start"),
  hangup: $("hangup"),
  ptt: $("ptt"),
  muteTts: $("mute-tts"),
  transcript: $("transcript"),
  activity: $("activity"),
  badge: $("call-badge"),
  connDot: $("conn-dot"),
  business: $("business-name"),
  typeForm: $("type-form"),
  typeInput: $("type-input"),
  typeSend: $("type-send"),
};

const VAD = {
  frameMs: 60,
  startFrames: 3,      // consecutive loud frames before we open the mic
  silenceMs: 1100,     // quiet run that ends an utterance
  minUtteranceMs: 350,
  maxUtteranceMs: 25000,
  floorFrames: 14,     // ~0.85s of calibration
};

let ws = null;
let stream = null;
let audioCtx = null;
let analyser = null;
let recorder = null;
let chunks = [];

let state = "idle";              // idle | listening | recording | thinking | speaking
let vadTimer = null;
let frameBuffer = null;
let noiseSamples = [];
let threshold = 0.02;
let loudRun = 0;
let quietSince = 0;
let utteranceStart = 0;
let pushToTalk = false;
let ttsMuted = false;
let pendingSpeech = 0;

/* ------------------------------------------------------------------ helpers */

function setState(next, label) {
  state = next;
  const orbState = next === "recording" ? "listening" : next;
  el.orb.dataset.state = ["listening", "thinking", "speaking"].includes(orbState)
    ? orbState
    : "idle";
  if (label) el.status.textContent = label;
}

function addTurn(role, text) {
  const div = document.createElement("div");
  div.className = `turn ${role}`;
  const who = document.createElement("span");
  who.className = "who";
  who.textContent = role === "user" ? "Caller" : "Receptionist";
  div.append(who, document.createTextNode(text));
  el.transcript.append(div);
  el.transcript.scrollTop = el.transcript.scrollHeight;
}

function addEvent(name, detail, kind = "") {
  const first = el.activity.querySelector(".muted");
  if (first) first.remove();
  const div = document.createElement("div");
  div.className = `event ${kind}`;
  const label = document.createElement("span");
  label.className = "name";
  label.textContent = name;
  const body = document.createElement("span");
  body.className = "detail";
  body.textContent = detail;
  div.append(label, body);
  el.activity.append(div);
  el.activity.scrollTop = el.activity.scrollHeight;
}

function describeTool(evt) {
  const r = evt.result || {};
  switch (evt.name) {
    case "search_knowledge_base":
      return r.hits
        ? [`Looked up “${r.query}”`, `${r.hits} passage(s) from ${(r.sources || []).join(", ")}`, "ok"]
        : [`Looked up “${r.query}”`, "Nothing in the knowledge base", "warn"];
    case "check_availability":
      return r.slots && r.slots.length
        ? [`Checked ${r.date}`, `Free: ${r.slots.join(", ")}`, "ok"]
        : [`Checked ${r.date || "date"}`, "No free slots", "warn"];
    case "book_appointment":
      return r.id
        ? ["Appointment booked", `#${r.id} — ${r.name}, ${r.starts_at.replace("T", " ")}`, "ok"]
        : ["Booking rejected", r.detail || r.error || "See transcript", "err"];
    case "save_note":
      return r.id
        ? [`Note saved (${r.category})`, r.content, "ok"]
        : ["Note not saved", r.error || "", "err"];
    default:
      return [evt.name, JSON.stringify(r), ""];
  }
}

/* ---------------------------------------------------------------------- TTS */

function pickVoice() {
  const voices = speechSynthesis.getVoices();
  if (!voices.length) return null;
  const preferred = [/natural/i, /google us english/i, /zira/i, /samantha/i];
  for (const rx of preferred) {
    const hit = voices.find((v) => rx.test(v.name) && v.lang.startsWith("en"));
    if (hit) return hit;
  }
  return voices.find((v) => v.lang.startsWith("en")) || voices[0];
}

function speak(text) {
  if (ttsMuted || !("speechSynthesis" in window)) return;
  const u = new SpeechSynthesisUtterance(text);
  const voice = pickVoice();
  if (voice) u.voice = voice;
  u.rate = 1.03;
  u.pitch = 1.0;
  pendingSpeech += 1;
  setState("speaking", "Speaking…");
  const done = () => {
    pendingSpeech = Math.max(0, pendingSpeech - 1);
    if (pendingSpeech === 0 && ws) resumeListening();
  };
  u.onend = done;
  u.onerror = done;
  speechSynthesis.speak(u);
}

function stopSpeaking() {
  if ("speechSynthesis" in window) speechSynthesis.cancel();
  pendingSpeech = 0;
}

function resumeListening() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  setState("listening", "Listening…");
  loudRun = 0;
  quietSince = 0;
}

/* --------------------------------------------------------------- recording */

function bestMimeType() {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/mp4",
  ];
  for (const type of candidates) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(type)) return type;
  }
  return "";
}

function startRecording() {
  if (recorder && recorder.state === "recording") return;
  chunks = [];
  const mimeType = bestMimeType();
  try {
    recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
  } catch (err) {
    addEvent("Recorder error", String(err), "err");
    return;
  }
  recorder.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
  recorder.onstop = () => {
    const elapsed = performance.now() - utteranceStart;
    const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
    chunks = [];
    if (elapsed < VAD.minUtteranceMs || blob.size < 1500) {
      resumeListening();
      return;
    }
    if (ws && ws.readyState === WebSocket.OPEN) {
      setState("thinking", "Transcribing…");
      blob.arrayBuffer().then((buf) => ws.send(buf));
    }
  };
  utteranceStart = performance.now();
  recorder.start();
  setState("recording", "Listening — go ahead");
}

function stopRecording() {
  if (recorder && recorder.state === "recording") recorder.stop();
}

/* ---------------------------------------------------------------- mic + VAD */

function vadFrame() {
  if (!analyser) return;
  analyser.getFloatTimeDomainData(frameBuffer);
  let sum = 0;
  for (let i = 0; i < frameBuffer.length; i += 1) sum += frameBuffer[i] * frameBuffer[i];
  const rms = Math.sqrt(sum / frameBuffer.length);

  el.level.style.width = `${Math.min(100, (rms / 0.25) * 100).toFixed(1)}%`;

  // Calibrate the room's noise floor before we start reacting to it.
  if (noiseSamples.length < VAD.floorFrames) {
    noiseSamples.push(rms);
    if (noiseSamples.length === VAD.floorFrames) {
      const floor = noiseSamples.reduce((a, b) => a + b, 0) / noiseSamples.length;
      threshold = Math.max(0.014, floor * 2.6 + 0.006);
      resumeListening();
    }
    return;
  }

  if (pushToTalk) return;
  if (state === "thinking" || state === "speaking") return;

  const loud = rms > threshold;

  if (state === "listening") {
    loudRun = loud ? loudRun + 1 : 0;
    if (loudRun >= VAD.startFrames) startRecording();
    return;
  }

  if (state === "recording") {
    const now = performance.now();
    if (loud) {
      quietSince = 0;
    } else if (!quietSince) {
      quietSince = now;
    } else if (now - quietSince > VAD.silenceMs) {
      stopRecording();
    }
    if (now - utteranceStart > VAD.maxUtteranceMs) stopRecording();
  }
}

async function openMic() {
  stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
      channelCount: 1,
    },
  });
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  await audioCtx.resume();
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 1024;
  analyser.smoothingTimeConstant = 0.4;
  frameBuffer = new Float32Array(analyser.fftSize);
  audioCtx.createMediaStreamSource(stream).connect(analyser);
  noiseSamples = [];
  vadTimer = setInterval(vadFrame, VAD.frameMs);
}

function closeMic() {
  if (vadTimer) clearInterval(vadTimer);
  vadTimer = null;
  if (recorder && recorder.state === "recording") recorder.stop();
  recorder = null;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  if (audioCtx) audioCtx.close().catch(() => {});
  audioCtx = null;
  analyser = null;
  el.level.style.width = "0%";
}

/* --------------------------------------------------------------- WebSocket */

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/call`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    el.connDot.classList.add("live");
    el.connDot.classList.remove("error");
  };

  ws.onmessage = (msg) => {
    let data;
    try { data = JSON.parse(msg.data); } catch { return; }
    handle(data);
  };

  ws.onclose = () => {
    el.connDot.classList.remove("live");
    endCall(false);
  };

  ws.onerror = () => {
    el.connDot.classList.add("error");
    addEvent("Connection error", "The call socket dropped.", "err");
  };
}

function handle(data) {
  switch (data.type) {
    case "ready":
      el.badge.textContent = `call #${data.call_id}`;
      addTurn("assistant", data.greeting);
      speak(data.greeting);
      break;

    case "status":
      if (data.stage === "transcribing") setState("thinking", "Transcribing…");
      else if (data.stage === "thinking") setState("thinking", "Thinking…");
      else if (data.note === "no_speech") resumeListening();
      break;

    case "transcript":
      addTurn(data.role, data.text);
      break;

    case "speech":
      speak(data.text);
      break;

    case "tool": {
      const [name, detail, kind] = describeTool(data);
      addEvent(name, detail, kind);
      break;
    }

    case "turn_end":
      if (pendingSpeech === 0) resumeListening();
      break;

    case "error":
      addEvent("Error", data.message, "err");
      resumeListening();
      break;

    case "summary":
      if (data.summary) {
        addEvent("Call summary", data.summary, "ok");
        (data.follow_ups || []).forEach((f) => addEvent("Follow-up", f, "warn"));
      }
      break;

    default:
      break;
  }
}

/* ------------------------------------------------------------------ actions */

async function startCall() {
  el.start.disabled = true;
  setState("thinking", "Requesting microphone…");
  try {
    await openMic();
  } catch (err) {
    setState("idle", "Microphone blocked — you can still type below.");
    addEvent("Microphone", String(err && err.message ? err.message : err), "err");
    el.start.disabled = false;
  }
  connect();
  setState("thinking", "Calibrating background noise…");
  el.hangup.disabled = false;
  el.ptt.disabled = !stream;
  el.muteTts.disabled = false;
  el.typeInput.disabled = false;
  el.typeSend.disabled = false;
}

function endCall(notifyServer = true) {
  if (notifyServer && ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "end" }));
  }
  stopSpeaking();
  closeMic();
  if (ws && notifyServer) setTimeout(() => ws && ws.close(), 400);
  setState("idle", "Call ended");
  el.start.disabled = false;
  el.hangup.disabled = true;
  el.ptt.disabled = true;
  el.muteTts.disabled = true;
  el.typeInput.disabled = true;
  el.typeSend.disabled = true;
}

el.start.addEventListener("click", startCall);
el.hangup.addEventListener("click", () => endCall(true));

el.muteTts.addEventListener("click", () => {
  ttsMuted = !ttsMuted;
  el.muteTts.textContent = ttsMuted ? "Unmute voice" : "Mute voice";
  el.muteTts.classList.toggle("active", ttsMuted);
  if (ttsMuted) { stopSpeaking(); resumeListening(); }
});

el.ptt.addEventListener("pointerdown", () => {
  if (!stream) return;
  pushToTalk = true;
  stopSpeaking();
  el.ptt.classList.add("active");
  startRecording();
});

const releasePtt = () => {
  if (!pushToTalk) return;
  pushToTalk = false;
  el.ptt.classList.remove("active");
  stopRecording();
};
el.ptt.addEventListener("pointerup", releasePtt);
el.ptt.addEventListener("pointerleave", releasePtt);

el.typeForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = el.typeInput.value.trim();
  if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;
  el.typeInput.value = "";
  stopSpeaking();
  setState("thinking", "Thinking…");
  ws.send(JSON.stringify({ type: "text", text }));
});

if ("speechSynthesis" in window) {
  speechSynthesis.onvoiceschanged = () => pickVoice();
  pickVoice();
}

fetch("/api/config")
  .then((r) => r.json())
  .then((cfg) => {
    el.business.textContent = `${cfg.business_name} — ${cfg.receptionist_name}`;
    document.title = `${cfg.business_name} — AI Receptionist`;
    addEvent(
      "Knowledge base",
      `${cfg.knowledge.files} file(s), ${cfg.knowledge.chunks} passages indexed`,
      cfg.knowledge.chunks ? "ok" : "warn"
    );
  })
  .catch(() => {});

window.addEventListener("beforeunload", () => {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "end" }));
});
