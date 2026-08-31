// Async retranscribe: click a .retranscribe-btn, POST to the endpoint,
// poll /jobs/<id> until status hits 'embedded' or 'failed', reload the
// page so the new transcript + source pill are visible.
//
// Wrapped in an IIFE because other page scripts (upload.js, recorder.js)
// also declare top-level names like `status`, and multiple <script>
// tags share global scope. See recorder.js for the diagnosis.

(() => {
"use strict";

const statusEl = document.getElementById("retranscribe-status");

document.querySelectorAll(".retranscribe-btn").forEach(btn => {
  btn.addEventListener("click", async () => {
    const itemId = btn.dataset.itemId;
    const withMethod = btn.dataset.with;
    disableAll(true);
    statusEl.textContent = `Queueing ${withMethod} retranscribe…`;
    try {
      const r = await fetch(
        `/item/${itemId}/retranscribe?with=${encodeURIComponent(withMethod)}`,
        { method: "POST" },
      );
      if (!r.ok) {
        const errText = await r.text();
        throw new Error(`${r.status}: ${errText.slice(0, 200)}`);
      }
      const { status_url } = await r.json();
      statusEl.textContent = "Queued. Waiting for transcribe → classify → embed…";
      await poll(status_url);
    } catch (e) {
      statusEl.textContent = `Failed: ${e.message}`;
      disableAll(false);
    }
  });
});

function disableAll(disabled) {
  document.querySelectorAll(".retranscribe-btn").forEach(b => {
    b.disabled = disabled;
  });
}

async function poll(url) {
  // Retranscribe on a long PDF can take 30–60s (force-OCR then Haiku
  // then embed). Poll every 3s for up to 5 minutes.
  for (let i = 0; i < 100; i++) {
    await new Promise(r => setTimeout(r, 3000));
    let j;
    try {
      const r = await fetch(url);
      if (!r.ok) continue;
      j = await r.json();
    } catch {
      continue;
    }
    if (j.status === "embedded" || j.status === "classified") {
      statusEl.textContent = "Done. Reloading…";
      // Reload so the item page re-fetches transcript + source pill.
      setTimeout(() => window.location.reload(), 500);
      return;
    }
    if (j.status === "failed" || j.status === "dead_letter") {
      statusEl.textContent = `Failed: ${j.error_message || "unknown error"}`;
      disableAll(false);
      return;
    }
    statusEl.textContent = `Status: ${j.status}…`;
  }
  statusEl.textContent = "Still processing after 5 minutes. Refresh the page manually to check.";
  disableAll(false);
}

})();
