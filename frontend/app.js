/* AI Receptionist — call console.
 *
 * Turn taking is done in the browser: an AnalyserNode watches the mic level,
 * MediaRecorder captures one utterance at a time, and the complete WebM blob is
 * pushed over the WebSocket when the caller stops speaking. Replies stream back
 * sentence by sentence. Each "speech" event carries the text; when the server
 * has Piper TTS enabled, an "audio" event with the synthesized WAV follows it
 * and that's what actually plays. If the server has TTS turned off, the
 * "ready" event says so and this falls back to the Web Speech API instead.
 */

const $ = (id) => document.getElementById(id);

const el = {
  orb: $("orb"),
  status: $("status"),
  level: $("level-bar"),
  call: $("call-btn"),
  transcript: $("transcript"),
  activity: $("activity"),
  badge: $("call-badge"),
  connDot: $("conn-dot"),
  business: $("business-name"),
};

const VAD = {
  // Whatever this is set to, recording only starts *after* startFrames
  // worth of confirmed speech — so every frame here is audio lost off the
  // front of the utterance. Kept short (5ms) so that onset delay
  // (startFrames * frameMs) stays low; the speech-band check below is what
  // actually keeps noise from opening the mic, not a long confirm run.
  frameMs: 5,            // 5ms per frame
  startFrames: 4,        // ~120ms of confirmed speech before we open the mic
  silenceMs: 1100,       // quiet run that ends an utterance
  minUtteranceMs: 650,
  maxUtteranceMs: 25000,
  floorFrames: 28,       // ~0.85s of initial calibration
  noiseEmaAlpha: 0.015,  // how fast the ambient floor keeps tracking a changing room
  speechBandMin: 0.42,   // min. fraction of energy in the 300-3400Hz voice band to open the mic
  bargeInFrames: 20,     // longer confirm run while our own TTS may be bleeding into the mic
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
let freqData = null;
let noiseSamples = [];
let noiseFloor = 0.02;
let threshold = 0.02;
let loudRun = 0;
let bargeInRun = 0;
let quietSince = 0;
let utteranceStart = 0;
let inCall = false;
let pendingSpeech = 0;
let serverTtsEnabled = false;
let audioQueue = [];
let currentAudio = null;

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
  if (!("speechSynthesis" in window)) return;
  const u = new SpeechSynthesisUtterance(text);
  const voice = pickVoice();
  if (voice) u.voice = voice;
  u.rate = 1.03;
  u.pitch = 1.0;
  pendingSpeech += 1;
  setState("speaking", "Speaking…");
  const done = () => {
    pendingSpeech = Math.max(0, pendingSpeech - 1);
    // Cancelling speechSynthesis for a barge-in fires this asynchronously —
    // by the time it lands we may already be recording the next utterance,
    // and this must not stomp that state back to "listening".
    if (pendingSpeech === 0 && ws && state === "speaking") resumeListening();
  };
  u.onend = done;
  u.onerror = done;
  speechSynthesis.speak(u);
}

function playServerAudio(base64) {
  audioQueue.push(base64);
  if (!currentAudio) playNextServerAudio();
}

function playNextServerAudio() {
  const b64 = audioQueue.shift();
  if (!b64) {
    currentAudio = null;
    if (pendingSpeech === 0 && ws) resumeListening();
    return;
  }
  pendingSpeech += 1;
  setState("speaking", "Speaking…");
  const audio = new Audio(`data:audio/wav;base64,${b64}`);
  currentAudio = audio;
  const done = () => {
    pendingSpeech = Math.max(0, pendingSpeech - 1);
    // .pause() (used to cut audio short on barge-in) doesn't fire onended,
    // so this only runs for a clip that actually finished or errored —
    // still, only advance the queue if nothing has moved us on already.
    if (state === "speaking") playNextServerAudio();
  };
  audio.onended = done;
  audio.onerror = done;
  audio.play().catch(done);
}

function stopSpeaking() {
  if ("speechSynthesis" in window) speechSynthesis.cancel();
  audioQueue = [];
  if (currentAudio) {
    currentAudio.pause();
    currentAudio = null;
  }
  pendingSpeech = 0;
}

function resumeListening() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  setState("listening", "Listening…");
  loudRun = 0;
  quietSince = 0;
}

function bargeIn() {
  bargeInRun = 0;
  stopSpeaking();
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "interrupt" }));
  }
  startRecording();
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
  // Voice-tuned bitrate — browsers often default Opus lower than this for
  // a generic recording, which can blur consonants the STT model then
  // can't recover.
  const opts = mimeType
    ? { mimeType, audioBitsPerSecond: 64000 }
    : { audioBitsPerSecond: 64000 };
  try {
    recorder = new MediaRecorder(stream, opts);
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

function speechBandRatio() {
  // Broadband noise (clicks, creaks, AC hum) can cross the RMS threshold
  // just as easily as speech — that's what was feeding Whisper garbage
  // audio it then hallucinated real-sounding replies for. Human speech
  // concentrates energy in the ~300-3400Hz band, so require a meaningful
  // share of it there before treating a loud frame as someone talking.
  analyser.getByteFrequencyData(freqData);
  const binHz = (audioCtx.sampleRate / 2) / freqData.length;
  let total = 0;
  let band = 0;
  for (let i = 0; i < freqData.length; i += 1) {
    const v = freqData[i];
    total += v;
    const hz = i * binHz;
    if (hz >= 300 && hz <= 3400) band += v;
  }
  return total > 0 ? band / total : 0;
}

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
      noiseFloor = noiseSamples.reduce((a, b) => a + b, 0) / noiseSamples.length;
      threshold = Math.max(0.014, noiseFloor * 2.6 + 0.006);
      resumeListening();
    }
    return;
  }

  const loud = rms > threshold;

  if (state === "thinking" || state === "speaking") {
    // Barge-in: let the caller talk over a reply instead of waiting it out.
    // While actually speaking, our own TTS can bleed into the mic even with
    // echo cancellation on, so require a longer confirmed run than opening
    // the mic fresh needs (headphones make this far more reliable — see the
    // on-page hint).
    const required = state === "speaking" ? VAD.bargeInFrames : VAD.startFrames;
    const speechLike = loud && speechBandRatio() > VAD.speechBandMin;
    bargeInRun = speechLike ? bargeInRun + 1 : 0;
    if (bargeInRun >= required) bargeIn();
    return;
  }

  if (state === "listening") {
    if (!loud) {
      // Keep tracking the ambient floor so the threshold adapts to a
      // room that gets noisier or quieter mid-call, not just at the start.
      noiseFloor = noiseFloor * (1 - VAD.noiseEmaAlpha) + rms * VAD.noiseEmaAlpha;
      threshold = Math.max(0.014, noiseFloor * 2.6 + 0.006);
      loudRun = 0;
      return;
    }
    loudRun = speechBandRatio() > VAD.speechBandMin ? loudRun + 1 : 0;
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
  freqData = new Uint8Array(analyser.frequencyBinCount);
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
      serverTtsEnabled = Boolean(data.tts_enabled);
      addTurn("assistant", data.greeting);
      // The greeting is fixed text sent before any per-sentence server
      // synthesis happens, so it always uses the browser's own voice.
      speak(data.greeting);
      break;

    case "status":
      if (data.stage === "transcribing") setState("thinking", "Transcribing…");
      else if (data.stage === "thinking") setState("thinking", "Thinking…");
      else if (data.note === "no_speech") resumeListening();
      else if (data.note === "interrupted") addEvent("Interrupted", "Caller spoke over the reply", "warn");
      break;

    case "transcript":
      addTurn(data.role, data.text);
      break;

    case "speech":
      // Only used as the spoken output when the server has no audio for this
      // sentence (TTS off, or Piper failed) — the "audio" event otherwise
      // arrives right after and is what actually plays.
      if (!serverTtsEnabled) speak(data.text);
      break;

    case "audio":
      playServerAudio(data.data);
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
  el.call.disabled = true;
  el.call.textContent = "Connecting…";
  setState("thinking", "Requesting microphone…");
  try {
    await openMic();
  } catch (err) {
    setState("idle", "Microphone blocked — this app is audio-only, so a call needs it.");
    addEvent("Microphone", String(err && err.message ? err.message : err), "err");
    el.call.disabled = false;
    el.call.textContent = "Start call";
    return;
  }
  connect();
  setState("thinking", "Calibrating background noise…");
  inCall = true;
  el.call.textContent = "End call";
  el.call.classList.remove("primary");
  el.call.classList.add("danger");
  el.call.disabled = false;
}

function endCall(notifyServer = true) {
  if (notifyServer && ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "end" }));
  }
  stopSpeaking();
  closeMic();
  if (ws && notifyServer) setTimeout(() => ws && ws.close(), 400);
  setState("idle", "Call ended");
  inCall = false;
  el.call.textContent = "Start call";
  el.call.classList.remove("danger");
  el.call.classList.add("primary");
  el.call.disabled = false;
}

el.call.addEventListener("click", () => {
  if (inCall) endCall(true);
  else startCall();
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
