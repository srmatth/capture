// Voice memo capture via the Web MediaRecorder API.
//
// Why not a <input type=file accept=audio/* capture>? On iOS the bare
// `capture` attribute is ambiguous — it defaults to "camera with video"
// alongside accept=audio/*, and even after picking an audio file the
// resulting mp4-container m4a fails the server's mime check. Recording
// in-browser sidesteps both problems.
//
// The recorder produces webm/opus on Chrome/Firefox and mp4/aac on
// Safari. Both are audio containers the server accepts (audio/* prefix
// matches on the mime check in routers/upload.py).

const recordBtn = document.getElementById("recordBtn");
const recordBtnLabel = document.getElementById("recordBtnLabel");
const status = document.getElementById("status");
const noteEl = document.getElementById("note");

// Diagnostic: surface load state in the status area so silent failures
// on iOS Safari (where the JS console isn't visible) become visible.
// This runs before we get to the getUserMedia check, so if the button
// tap doesn't produce this message we know the click handler itself
// never fires.
function dbg(msg) {
  if (status) status.textContent = msg;
  console.log("[recorder]", msg);
}

if (!recordBtn) {
  dbg("recorder: #recordBtn element not found");
}

// Try mime types in preference order — Safari doesn't support webm/opus
// so we fall through to mp4/aac. If the browser accepts NEITHER we let
// MediaRecorder pick its default with an empty string (edge browsers).
const _PREFERRED_MIMES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/mp4;codecs=mp4a.40.2",
  "audio/mp4",
  "",
];

let mediaRecorder = null;
let stream = null;
let chunks = [];
let recording = false;

if (recordBtn) {
  recordBtn.addEventListener("click", async () => {
    dbg("recorder: button clicked");
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      dbg("recorder: navigator.mediaDevices unavailable "
          + `(secureContext=${window.isSecureContext})`);
      return;
    }
    if (!recording) {
      await startRecording();
    } else {
      stopRecording();
    }
  });
}

async function startRecording() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    status.textContent = "This browser doesn't support in-page audio recording.";
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    status.textContent = `Mic permission denied: ${e.message}`;
    return;
  }

  const mimeType = pickMimeType();
  try {
    mediaRecorder = mimeType
      ? new MediaRecorder(stream, { mimeType })
      : new MediaRecorder(stream);
  } catch (e) {
    status.textContent = `Recorder init failed: ${e.message}`;
    stopStream();
    return;
  }

  chunks = [];
  mediaRecorder.ondataavailable = (ev) => {
    if (ev.data && ev.data.size > 0) chunks.push(ev.data);
  };
  mediaRecorder.onstop = async () => {
    stopStream();
    const blob = new Blob(chunks, { type: mediaRecorder.mimeType || "audio/webm" });
    await uploadRecording(blob);
  };
  mediaRecorder.start();
  recording = true;
  recordBtn.classList.add("recording");
  recordBtnLabel.textContent = "⏹ Stop and upload";
  status.textContent = "Recording… tap the button again to stop.";
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();  // triggers onstop -> uploadRecording
  }
  recording = false;
  recordBtn.classList.remove("recording");
  recordBtnLabel.textContent = "🎤 Record voice memo";
}

function stopStream() {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
}

function pickMimeType() {
  if (typeof MediaRecorder === "undefined" || !MediaRecorder.isTypeSupported) return "";
  for (const m of _PREFERRED_MIMES) {
    if (m === "") return "";
    if (MediaRecorder.isTypeSupported(m)) return m;
  }
  return "";
}

async function uploadRecording(blob) {
  status.textContent = "Uploading recording…";
  // Extension driven by mime type so the server-side kind sniff sees
  // the right hint. audio/webm -> .webm, audio/mp4 -> .m4a.
  const ext = blob.type.includes("mp4") ? "m4a"
             : blob.type.includes("webm") ? "webm"
             : "audio";
  const fd = new FormData();
  fd.set("file", new File([blob], `voice-memo.${ext}`, { type: blob.type }));
  if (noteEl && noteEl.value) fd.append("note", noteEl.value);
  try {
    const r = await fetch("/upload", { method: "POST", body: fd });
    if (!r.ok) throw new Error(`upload failed: ${r.status}`);
    const { id, status_url } = await r.json();
    status.textContent = "Uploaded. Processing…";
    if (noteEl) noteEl.value = "";
    // uploadSingle in upload.js has the poll loop; call the same
    // helper if it's on the page. The pattern is duplicated here to
    // keep the recorder self-contained.
    pollForCompletion(status_url);
  } catch (e) {
    status.textContent = `Upload error: ${e.message}`;
  }
}

async function pollForCompletion(url) {
  for (let i = 0; i < 60; i++) {
    await new Promise((r) => setTimeout(r, 5000));
    let j;
    try {
      const r = await fetch(url);
      if (!r.ok) continue;
      j = await r.json();
    } catch { continue; }
    if (j.status === "embedded") {
      status.innerHTML = `Done: <strong>${escapeHtml(j.title || "(untitled)")}</strong>
        filed under <code>${escapeHtml(j.path)}</code>`;
      return;
    }
    if (j.status === "failed") {
      status.textContent = `Failed: ${j.error_message}`;
      return;
    }
    status.textContent = `Status: ${j.status}…`;
  }
  status.textContent = "Still processing. Check the Inbox / Recent later.";
}

function escapeHtml(s) {
  return String(s || "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
