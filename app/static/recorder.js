// Voice memo capture via the Web MediaRecorder API.
//
// UI is a modal with clear state:
//   idle       -> Start button (nothing recording yet)
//   recording  -> Pause + Stop, level meter live, duration counting up
//   paused     -> Resume + Stop, duration frozen
//   review     -> Playback preview + Upload/Retake
//
// Why not <input type=file accept=audio/* capture>? On iOS the bare
// `capture` attribute opens the camera-with-video. In-browser recording
// sidesteps that. Requires HTTPS on iOS (getUserMedia needs a secure
// context); the tailnet cert Caddy serves handles that.
//
// Wrapped in an IIFE because upload.js also declares module-level
// names. See top of upload.js for the full explanation.

(() => {
"use strict";

// ---------- DOM refs ----------

const recordBtn = document.getElementById("recordBtn");
const modal = document.getElementById("recorderModal");
const durationEl = document.getElementById("recorderDuration");
const levelCanvas = document.getElementById("recorderLevel");
const stateEl = document.getElementById("recorderState");
const previewEl = document.getElementById("recorderPreview");

const btnStart = document.getElementById("recorderStart");
const btnPause = document.getElementById("recorderPause");
const btnResume = document.getElementById("recorderResume");
const btnStop = document.getElementById("recorderStop");
const btnUpload = document.getElementById("recorderUpload");
const btnRetake = document.getElementById("recorderRetake");
const btnCancel = document.getElementById("recorderCancel");

const statusEl = document.getElementById("status");
const noteEl = document.getElementById("note");

if (!recordBtn) return;    // page doesn't have the recorder — bail cleanly

// ---------- state ----------

const _PREFERRED_MIMES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/mp4;codecs=mp4a.40.2",
  "audio/mp4",
  "",
];

let mediaRecorder = null;
let stream = null;
let audioCtx = null;
let analyser = null;
let levelRAF = null;
let chunks = [];
let recorderMime = "";
let startedAt = 0;
let elapsedBefore = 0;      // accumulated seconds from previous non-paused runs
let durationTimer = null;
let finalBlob = null;
let previewURL = null;

// ---------- UI state machine ----------

const STATES = {
  idle: {
    show: [btnStart, btnCancel],
    hide: [btnPause, btnResume, btnStop, btnUpload, btnRetake, previewEl],
    stateText: "Tap Start to begin.",
  },
  recording: {
    show: [btnPause, btnStop, btnCancel],
    hide: [btnStart, btnResume, btnUpload, btnRetake, previewEl],
    stateText: "Recording…",
  },
  paused: {
    show: [btnResume, btnStop, btnCancel],
    hide: [btnStart, btnPause, btnUpload, btnRetake, previewEl],
    stateText: "Paused.",
  },
  review: {
    show: [btnUpload, btnRetake, btnCancel, previewEl],
    hide: [btnStart, btnPause, btnResume, btnStop],
    stateText: "Review your recording, then upload or retake.",
  },
};

function setState(name) {
  const s = STATES[name];
  s.show.forEach((el) => (el.hidden = false));
  s.hide.forEach((el) => (el.hidden = true));
  stateEl.textContent = s.stateText;
  modal.dataset.state = name;
}

// ---------- open / close ----------

recordBtn.addEventListener("click", () => {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    if (statusEl) {
      statusEl.textContent =
        "In-page recording requires HTTPS. Open this app via the HTTPS URL.";
    }
    return;
  }
  openModal();
});

function openModal() {
  modal.hidden = false;
  finalBlob = null;
  chunks = [];
  elapsedBefore = 0;
  updateDuration(0);
  clearLevelCanvas();
  setState("idle");
}

function closeModal() {
  stopEverything();
  modal.hidden = true;
  if (previewURL) {
    URL.revokeObjectURL(previewURL);
    previewURL = null;
    previewEl.src = "";
  }
}

btnCancel.addEventListener("click", closeModal);

// ---------- start / pause / resume / stop ----------

btnStart.addEventListener("click", async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    stateEl.textContent = `Mic permission denied: ${e.message}`;
    return;
  }

  recorderMime = pickMimeType();
  try {
    mediaRecorder = recorderMime
      ? new MediaRecorder(stream, { mimeType: recorderMime })
      : new MediaRecorder(stream);
  } catch (e) {
    stateEl.textContent = `Recorder init failed: ${e.message}`;
    stopEverything();
    return;
  }

  chunks = [];
  mediaRecorder.ondataavailable = (ev) => {
    if (ev.data && ev.data.size > 0) chunks.push(ev.data);
  };
  mediaRecorder.onstop = () => {
    finalBlob = new Blob(chunks, {
      type: mediaRecorder.mimeType || recorderMime || "audio/webm",
    });
    previewURL = URL.createObjectURL(finalBlob);
    previewEl.src = previewURL;
    setState("review");
    stopLevelMeter();
    stopStream();
  };

  mediaRecorder.start(250);   // fire ondataavailable every 250ms so
                              // pause preserves buffered data
  elapsedBefore = 0;
  startedAt = Date.now();
  startDurationTimer();
  startLevelMeter();
  setState("recording");
});

btnPause.addEventListener("click", () => {
  if (!mediaRecorder || mediaRecorder.state !== "recording") return;
  mediaRecorder.pause();
  elapsedBefore += (Date.now() - startedAt) / 1000;
  stopDurationTimer();
  stopLevelMeter();
  drawFlatLevel();
  setState("paused");
});

btnResume.addEventListener("click", () => {
  if (!mediaRecorder || mediaRecorder.state !== "paused") return;
  mediaRecorder.resume();
  startedAt = Date.now();
  startDurationTimer();
  startLevelMeter();
  setState("recording");
});

btnStop.addEventListener("click", () => {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();   // triggers onstop -> review state
  }
  stopDurationTimer();
});

btnRetake.addEventListener("click", () => {
  finalBlob = null;
  chunks = [];
  elapsedBefore = 0;
  if (previewURL) {
    URL.revokeObjectURL(previewURL);
    previewURL = null;
    previewEl.src = "";
  }
  updateDuration(0);
  clearLevelCanvas();
  setState("idle");
});

btnUpload.addEventListener("click", async () => {
  if (!finalBlob) return;
  stateEl.textContent = "Uploading…";
  btnUpload.disabled = true;
  btnRetake.disabled = true;
  const ext = finalBlob.type.includes("mp4") ? "m4a"
             : finalBlob.type.includes("webm") ? "webm"
             : "audio";
  const fd = new FormData();
  fd.set("file", new File([finalBlob], `voice-memo.${ext}`, {
    type: finalBlob.type,
  }));
  if (noteEl && noteEl.value) fd.append("note", noteEl.value);
  try {
    const r = await fetch("/upload", { method: "POST", body: fd });
    if (!r.ok) throw new Error(`upload failed: ${r.status}`);
    const { id, status_url } = await r.json();
    if (statusEl) statusEl.textContent = "Uploaded. Processing…";
    if (noteEl) noteEl.value = "";
    closeModal();
    pollForCompletion(status_url);
  } catch (e) {
    stateEl.textContent = `Upload error: ${e.message}`;
    btnUpload.disabled = false;
    btnRetake.disabled = false;
  }
});

// ---------- duration counter ----------

function startDurationTimer() {
  stopDurationTimer();
  durationTimer = setInterval(() => {
    const now = (Date.now() - startedAt) / 1000 + elapsedBefore;
    updateDuration(now);
  }, 100);
}

function stopDurationTimer() {
  if (durationTimer) {
    clearInterval(durationTimer);
    durationTimer = null;
  }
}

function updateDuration(seconds) {
  const s = Math.floor(seconds);
  const mm = Math.floor(s / 60);
  const ss = s % 60;
  durationEl.textContent = `${mm}:${String(ss).padStart(2, "0")}`;
}

// ---------- level meter ----------

function startLevelMeter() {
  if (!stream) return;
  try {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    const source = audioCtx.createMediaStreamSource(stream);
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 512;
    source.connect(analyser);
    drawLevel();
  } catch (e) {
    // Level meter is nice-to-have. If AudioContext fails, keep recording.
    console.warn("level meter unavailable:", e);
  }
}

function drawLevel() {
  if (!analyser) return;
  const ctx = levelCanvas.getContext("2d");
  const w = levelCanvas.width;
  const h = levelCanvas.height;
  const data = new Uint8Array(analyser.frequencyBinCount);

  function tick() {
    analyser.getByteTimeDomainData(data);
    // Peak amplitude around 128 (silence) -> pushes to 0 or 255 (loud).
    let peak = 0;
    for (let i = 0; i < data.length; i++) {
      const v = Math.abs(data[i] - 128);
      if (v > peak) peak = v;
    }
    const pct = Math.min(1, peak / 128);

    ctx.fillStyle = getComputedStyle(document.body).getPropertyValue("--bg") || "#fff";
    ctx.fillRect(0, 0, w, h);

    // Filled bar from left, colored by level.
    const barW = w * pct;
    ctx.fillStyle = pct > 0.85 ? "#dc2626"
                     : pct > 0.4  ? "#eab308"
                     : "#22c55e";
    ctx.fillRect(0, h * 0.3, barW, h * 0.4);

    levelRAF = requestAnimationFrame(tick);
  }
  tick();
}

function stopLevelMeter() {
  if (levelRAF) {
    cancelAnimationFrame(levelRAF);
    levelRAF = null;
  }
  analyser = null;
}

function drawFlatLevel() {
  const ctx = levelCanvas.getContext("2d");
  const w = levelCanvas.width;
  const h = levelCanvas.height;
  ctx.fillStyle = getComputedStyle(document.body).getPropertyValue("--bg") || "#fff";
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = getComputedStyle(document.body).getPropertyValue("--muted") || "#888";
  ctx.fillRect(0, h * 0.48, w, h * 0.04);   // thin flat line
}

function clearLevelCanvas() {
  const ctx = levelCanvas.getContext("2d");
  ctx.clearRect(0, 0, levelCanvas.width, levelCanvas.height);
}

// ---------- cleanup ----------

function stopEverything() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    try { mediaRecorder.stop(); } catch { /* ignore */ }
  }
  stopDurationTimer();
  stopLevelMeter();
  stopStream();
}

function stopStream() {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  // AudioContext can be reused across sessions; keep it open.
}

function pickMimeType() {
  if (typeof MediaRecorder === "undefined" || !MediaRecorder.isTypeSupported) return "";
  for (const m of _PREFERRED_MIMES) {
    if (m === "") return "";
    if (MediaRecorder.isTypeSupported(m)) return m;
  }
  return "";
}

// ---------- status polling for a submitted memo ----------

async function pollForCompletion(url) {
  if (!statusEl) return;
  for (let i = 0; i < 60; i++) {
    await new Promise((r) => setTimeout(r, 5000));
    let j;
    try {
      const r = await fetch(url);
      if (!r.ok) continue;
      j = await r.json();
    } catch { continue; }
    if (j.status === "embedded") {
      statusEl.innerHTML = `Done: <strong>${escapeHtml(j.title || "(untitled)")}</strong>
        filed under <code>${escapeHtml(j.path)}</code>`;
      return;
    }
    if (j.status === "failed") {
      statusEl.textContent = `Failed: ${j.error_message}`;
      return;
    }
    statusEl.textContent = `Status: ${j.status}…`;
  }
  statusEl.textContent = "Still processing. Check the Inbox / Recent later.";
}

function escapeHtml(s) {
  return String(s || "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

})();
