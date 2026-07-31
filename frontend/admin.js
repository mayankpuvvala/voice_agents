/* Dashboard: calls, appointments, notes and the knowledge base. */

const view = document.getElementById("view");
const dialog = document.getElementById("call-dialog");
const detail = document.getElementById("call-detail");

let current = "calls";

const get = (url) => fetch(url).then((r) => r.json());

const esc = (v) =>
  String(v ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );

const when = (iso) => {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime())
    ? esc(iso)
    : d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
};

const empty = (msg) => `<p class="empty">${esc(msg)}</p>`;

/* ------------------------------------------------------------------- views */

async function renderCalls() {
  const calls = await get("/api/calls");
  if (!calls.length) return empty("No calls yet. Open the call console and say hello.");
  return `<table>
    <thead><tr>
      <th>#</th><th>Started</th><th>Caller</th><th>Summary</th>
      <th>Booked</th><th>Notes</th><th></th>
    </tr></thead>
    <tbody>${calls
      .map(
        (c) => `<tr>
          <td>${c.id}</td>
          <td>${when(c.started_at)}</td>
          <td>${esc(c.caller_name || "—")}<br><span class="muted">${esc(c.caller_phone || "")}</span></td>
          <td>${esc(c.summary || (c.ended_at ? "—" : "in progress"))}</td>
          <td>${c.appointment_count}</td>
          <td>${c.note_count}</td>
          <td><button class="btn ghost" data-call="${c.id}">Transcript</button></td>
        </tr>`
      )
      .join("")}</tbody></table>`;
}

async function renderAppointments() {
  const rows = await get("/api/appointments");
  if (!rows.length) return empty("Nothing booked yet.");
  return `<table>
    <thead><tr>
      <th>When</th><th>Name</th><th>Contact</th><th>Reason</th>
      <th>Length</th><th>Status</th><th></th>
    </tr></thead>
    <tbody>${rows
      .map(
        (a) => `<tr>
          <td>${when(a.starts_at)}</td>
          <td>${esc(a.name)}</td>
          <td>${esc(a.phone || "—")}<br><span class="muted">${esc(a.email || "")}</span></td>
          <td>${esc(a.reason || "—")}</td>
          <td>${a.duration_minutes} min</td>
          <td><span class="pill ${esc(a.status)}">${esc(a.status)}</span></td>
          <td>${
            a.status === "booked"
              ? `<button class="btn ghost" data-cancel="${a.id}">Cancel</button>`
              : ""
          }</td>
        </tr>`
      )
      .join("")}</tbody></table>`;
}

async function renderNotes() {
  const rows = await get("/api/notes");
  if (!rows.length) return empty("No notes recorded yet.");
  return `<table>
    <thead><tr><th>When</th><th>Category</th><th>Note</th><th>Call</th></tr></thead>
    <tbody>${rows
      .map(
        (n) => `<tr>
          <td>${when(n.created_at)}</td>
          <td><span class="pill ${esc(n.category)}">${esc(n.category)}</span></td>
          <td>${esc(n.content)}</td>
          <td>${n.call_id ? `<button class="btn ghost" data-call="${n.call_id}">#${n.call_id}</button>` : "—"}</td>
        </tr>`
      )
      .join("")}</tbody></table>`;
}

async function renderKnowledge() {
  const kb = await get("/api/knowledge");
  return `
    <div class="upload-row">
      <input type="file" id="kb-file" accept=".md,.txt,.pdf" />
      <button class="btn primary" id="kb-upload">Upload</button>
      <button class="btn ghost" id="kb-reindex">Re-index</button>
      <span class="muted">${kb.files} file(s), ${kb.chunks} passages indexed</span>
    </div>
    <p class="muted">
      Drop .md, .txt or .pdf files into the <code>knowledge/</code> folder or upload them here.
      The receptionist answers factual questions only from these documents.
    </p>
    ${
      kb.sources.length
        ? `<table><thead><tr><th>Indexed file</th></tr></thead><tbody>${kb.sources
            .map((s) => `<tr><td>${esc(s)}</td></tr>`)
            .join("")}</tbody></table>`
        : empty("No documents indexed — the receptionist has nothing to answer from.")
    }`;
}

const RENDERERS = {
  calls: renderCalls,
  appointments: renderAppointments,
  notes: renderNotes,
  knowledge: renderKnowledge,
};

async function render() {
  view.innerHTML = `<p class="empty">Loading…</p>`;
  try {
    view.innerHTML = await RENDERERS[current]();
  } catch (err) {
    view.innerHTML = empty(`Could not load: ${err}`);
  }
}

/* ------------------------------------------------------------------ details */

async function openCall(id) {
  const call = await get(`/api/calls/${id}`);
  detail.innerHTML = `
    <h2 style="margin-top:0">Call #${call.id}</h2>
    <p class="muted">${when(call.started_at)} → ${when(call.ended_at)}</p>
    ${call.summary ? `<p>${esc(call.summary)}</p>` : ""}
    ${
      call.follow_ups && call.follow_ups.length
        ? `<p><strong>Follow-ups</strong></p><ul>${call.follow_ups
            .map((f) => `<li>${esc(f)}</li>`)
            .join("")}</ul>`
        : ""
    }
    <div class="transcript" style="max-height:46vh">
      ${call.messages
        .map(
          (m) =>
            `<div class="turn ${esc(m.role)}"><span class="who">${
              m.role === "user" ? "Caller" : "Receptionist"
            }</span>${esc(m.content)}</div>`
        )
        .join("")}
    </div>`;
  dialog.showModal();
}

/* ------------------------------------------------------------------- events */

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    current = tab.dataset.view;
    render();
  });
});

document.getElementById("refresh").addEventListener("click", (e) => {
  e.preventDefault();
  render();
});

document.getElementById("close-dialog").addEventListener("click", () => dialog.close());

view.addEventListener("click", async (e) => {
  const target = e.target;

  if (target.dataset.call) {
    openCall(target.dataset.call);
    return;
  }

  if (target.dataset.cancel) {
    await fetch(`/api/appointments/${target.dataset.cancel}/cancel`, { method: "POST" });
    render();
    return;
  }

  if (target.id === "kb-reindex") {
    target.disabled = true;
    await fetch("/api/knowledge/reindex", { method: "POST" });
    render();
    return;
  }

  if (target.id === "kb-upload") {
    const input = document.getElementById("kb-file");
    if (!input.files.length) return;
    const body = new FormData();
    body.append("file", input.files[0]);
    target.disabled = true;
    const res = await fetch("/api/knowledge/upload", { method: "POST", body });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Upload failed" }));
      alert(err.detail || "Upload failed");
    }
    render();
  }
});

get("/api/config")
  .then((cfg) => {
    document.getElementById("business-name").textContent = `${cfg.business_name} — Dashboard`;
  })
  .catch(() => {});

render();
