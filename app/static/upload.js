// Client-side capture flow. Every image gets a crop pass. Audio and
// existing-file uploads bypass the cropper and post directly.

const noteEl = document.getElementById("note");
const status = document.getElementById("status");
const cropStage = document.getElementById("cropStage");
const cropImage = document.getElementById("cropImage");
const batchTray = document.getElementById("batchTray");
const batchThumbs = document.getElementById("batchThumbs");
const batchSubmit = document.getElementById("batchSubmit");

let cropper = null;
let pendingCropFor = null;      // 'single' | 'batch'
let batchPages = [];            // {blob, url} in insertion order

document.querySelectorAll('input[type="file"]').forEach(input => {
  input.addEventListener("change", async () => {
    if (!input.files.length) return;
    const file = input.files[0];
    const flow = input.dataset.flow;
    // Reset so re-selecting the same file still fires 'change'.
    input.value = "";

    if (flow === "direct") {
      await uploadSingle(file);
      return;
    }
    if (flow === "single" || flow === "batch-add") {
      pendingCropFor = flow === "single" ? "single" : "batch";
      openCropper(file);
    }
  });
});

function openCropper(file) {
  const url = URL.createObjectURL(file);
  cropImage.src = url;
  cropStage.hidden = false;
  cropImage.onload = () => {
    if (cropper) cropper.destroy();
    cropper = new Cropper(cropImage, {
      viewMode: 1,
      dragMode: "move",
      autoCropArea: 1,
      background: false,
      responsive: true,
    });
  };
}

document.getElementById("cropCancel").onclick = closeCropper;
document.getElementById("cropReset").onclick = () => cropper && cropper.reset();
document.getElementById("cropRotate").onclick = () => cropper && cropper.rotate(90);
document.getElementById("cropAccept").onclick = async () => {
  if (!cropper) return;
  // Cap output to keep uploads bounded. Phone cameras produce 12MP+ images;
  // OCR quality doesn't benefit beyond ~2400px for typical page-sized content.
  const canvas = cropper.getCroppedCanvas({
    maxWidth: 2400, maxHeight: 2400, imageSmoothingQuality: "high",
  });
  const blob = await new Promise(r => canvas.toBlob(r, "image/jpeg", 0.92));
  const wasBatch = (pendingCropFor === "batch");
  closeCropper();
  if (wasBatch) {
    addPageToBatch(blob);
  } else {
    await uploadSingle(new File([blob], "capture.jpg", { type: "image/jpeg" }));
  }
};

function closeCropper() {
  if (cropper) { cropper.destroy(); cropper = null; }
  cropStage.hidden = true;
  cropImage.src = "";
  pendingCropFor = null;
}

// ---------- Batch flow ----------

function addPageToBatch(blob) {
  const url = URL.createObjectURL(blob);
  batchPages.push({ blob, url });
  renderBatchTray();
}

function renderBatchTray() {
  batchTray.hidden = batchPages.length === 0;
  batchThumbs.innerHTML = "";
  batchPages.forEach((p, i) => {
    const li = document.createElement("li");
    const img = document.createElement("img");
    img.src = p.url;
    const btn = document.createElement("button");
    btn.textContent = "×";
    btn.title = "Remove this page";
    btn.onclick = () => {
      URL.revokeObjectURL(p.url);
      batchPages.splice(i, 1);
      renderBatchTray();
    };
    li.append(img, btn);
    batchThumbs.append(li);
  });
  batchSubmit.textContent = `Submit ${batchPages.length} page${batchPages.length === 1 ? "" : "s"}`;
}

document.getElementById("batchAddMore").onclick = () => {
  document.querySelector('input[data-flow="batch-add"]').click();
};
document.getElementById("batchCancel").onclick = () => {
  batchPages.forEach(p => URL.revokeObjectURL(p.url));
  batchPages = [];
  renderBatchTray();
};
document.getElementById("batchSubmit").onclick = async () => {
  if (!batchPages.length) return;
  status.textContent = `Uploading ${batchPages.length} pages…`;
  const fd = new FormData();
  batchPages.forEach((p, i) => fd.append("files", p.blob, `page-${i + 1}.jpg`));
  if (noteEl.value) fd.append("note", noteEl.value);
  try {
    const r = await fetch("/upload_batch", { method: "POST", body: fd });
    if (!r.ok) throw new Error(`upload failed: ${r.status}`);
    const { id, status_url, pages } = await r.json();
    status.textContent = `Uploaded ${pages} pages. Processing…`;
    batchPages.forEach(p => URL.revokeObjectURL(p.url));
    batchPages = [];
    renderBatchTray();
    noteEl.value = "";
    poll(id, status_url);
  } catch (e) {
    status.textContent = `Error: ${e.message}`;
  }
};

// ---------- Single upload ----------

async function uploadSingle(file) {
  status.textContent = "Uploading…";
  const fd = new FormData();
  fd.set("file", file);
  if (noteEl.value) fd.append("note", noteEl.value);
  try {
    const r = await fetch("/upload", { method: "POST", body: fd });
    if (!r.ok) throw new Error(`upload failed: ${r.status}`);
    const { id, status_url } = await r.json();
    status.textContent = "Uploaded. Processing…";
    noteEl.value = "";
    poll(id, status_url);
  } catch (e) {
    status.textContent = `Error: ${e.message}`;
  }
}

// ---------- Status polling ----------

async function poll(id, url) {
  for (let i = 0; i < 60; i++) {              // ~5 minutes at 5s
    await new Promise(r => setTimeout(r, 5000));
    const r = await fetch(url);
    if (!r.ok) continue;
    const j = await r.json();
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
  return String(s || "").replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
