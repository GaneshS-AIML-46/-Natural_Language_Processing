/* =============================================================
   PlagiaScope – script.js
   Handles: drag-and-drop, file management, API call, rendering
   ============================================================= */

"use strict";

// ── DOM refs ────────────────────────────────────────────────
const dropZone        = document.getElementById("drop-zone");
const fileInput       = document.getElementById("file-input");
const browseBtn       = document.getElementById("browse-btn");
const fileList        = document.getElementById("file-list");
const thresholdSlider = document.getElementById("threshold-slider");
const thresholdDisplay= document.getElementById("threshold-display");
const analyzeBtn      = document.getElementById("analyze-btn");
const errorBox        = document.getElementById("error-box");
const loadingOverlay  = document.getElementById("loading-overlay");
const resultsSection  = document.getElementById("results-section");
const statsBar        = document.getElementById("stats-bar");
const matrixWrapper   = document.getElementById("matrix-wrapper");
const flaggedBadge    = document.getElementById("flagged-badge");
const flaggedList     = document.getElementById("flagged-list");
const sentenceSection = document.getElementById("sentence-section");
const sentenceIntro   = document.getElementById("sentence-intro");
const sentenceList    = document.getElementById("sentence-list");
const heatmapImg      = document.getElementById("heatmap-img");
const downloadBtn     = document.getElementById("download-heatmap-btn");
const resetBtn        = document.getElementById("reset-btn");

// ── State ───────────────────────────────────────────────────
let selectedFiles = [];   // Array of File objects

// ── Helpers ─────────────────────────────────────────────────
const fmt  = n  => `${n.toFixed(1)}%`;
const fmtN = n  => n.toFixed(0);
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

function showError(msg) {
  errorBox.textContent = "⚠ " + msg;
  errorBox.classList.remove("hidden");
  setTimeout(() => errorBox.classList.add("hidden"), 7000);
}

function hideError() { errorBox.classList.add("hidden"); }

function setLoading(on) {
  loadingOverlay.classList.toggle("hidden", !on);
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + " B";
  return (bytes / 1024).toFixed(1) + " KB";
}

// ── File selection management ────────────────────────────────
function addFiles(fileArr) {
  const existing = new Set(selectedFiles.map(f => f.name + f.size));
  for (const f of fileArr) {
    const ext = f.name.split(".").pop().toLowerCase();
    if (!["txt","md"].includes(ext)) {
      showError(`"${f.name}" is not supported. Only .txt and .md files.`);
      continue;
    }
    if (!existing.has(f.name + f.size)) {
      selectedFiles.push(f);
      existing.add(f.name + f.size);
    }
  }
  renderFileList();
  updateAnalyzeBtn();
}

function removeFile(index) {
  selectedFiles.splice(index, 1);
  renderFileList();
  updateAnalyzeBtn();
}

function renderFileList() {
  fileList.innerHTML = "";
  selectedFiles.forEach((file, i) => {
    const li = document.createElement("li");
    li.className = "file-item";
    li.innerHTML = `
      <span class="file-icon">📄</span>
      <span class="file-name" title="${file.name}">${file.name}</span>
      <span class="file-size">${formatBytes(file.size)}</span>
      <button class="file-remove" title="Remove" data-index="${i}">✕</button>
    `;
    fileList.appendChild(li);
  });
  fileList.querySelectorAll(".file-remove").forEach(btn => {
    btn.addEventListener("click", () => removeFile(parseInt(btn.dataset.index)));
  });
}

function updateAnalyzeBtn() {
  analyzeBtn.disabled = selectedFiles.length < 2;
}

// ── Drag and Drop ────────────────────────────────────────────
dropZone.addEventListener("dragover", e => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  addFiles([...e.dataTransfer.files]);
});
dropZone.addEventListener("click", e => {
  if (e.target !== browseBtn) fileInput.click();
});
browseBtn.addEventListener("click", e => {
  e.stopPropagation();
  fileInput.click();
});
fileInput.addEventListener("change", () => {
  addFiles([...fileInput.files]);
  fileInput.value = "";
});

// ── Threshold Slider ─────────────────────────────────────────
thresholdSlider.addEventListener("input", () => {
  thresholdDisplay.textContent = thresholdSlider.value + "%";
});

// ── Analyze ──────────────────────────────────────────────────
analyzeBtn.addEventListener("click", runAnalysis);

async function runAnalysis() {
  hideError();
  if (selectedFiles.length < 2) { showError("Please add at least 2 files."); return; }

  const formData = new FormData();
  selectedFiles.forEach(f => formData.append("files[]", f));
  formData.append("threshold", thresholdSlider.value);

  setLoading(true);
  try {
    const resp = await fetch("/analyze", { method: "POST", body: formData });
    const data = await resp.json();

    if (!resp.ok || data.error) {
      showError(data.error || "Server error. Please try again.");
      return;
    }

    renderResults(data);
    resultsSection.classList.remove("hidden");
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });

  } catch (err) {
    showError("Could not connect to the server. Is app.py running?");
  } finally {
    setLoading(false);
  }
}

// ── Render Results ───────────────────────────────────────────
function renderResults(data) {
  renderStats(data.stats);
  renderMatrix(data.matrix, data.names);
  renderFlagged(data.flagged, data.stats.threshold_pct);
  renderSentences(data.sentence_results, data.top_pair);
  renderHeatmap(data.heatmap);
}

// Stats Bar
function renderStats(stats) {
  const items = [
    { label: "Documents",      value: stats.total_docs,    cls: "info"    },
    { label: "Total Pairs",    value: stats.total_pairs,   cls: "accent"  },
    { label: "Flagged",        value: stats.flagged_count, cls: stats.flagged_count > 0 ? "danger" : "success" },
    { label: "Clean",          value: stats.clean_count,   cls: "success" },
    { label: "Max Similarity", value: stats.max_score.toFixed(1) + "%", cls: stats.max_score >= stats.threshold_pct ? "danger" : "info", raw: true },
    { label: "Avg Similarity", value: stats.avg_score.toFixed(1) + "%", cls: "warning", raw: true },
  ];
  statsBar.innerHTML = items.map(it => `
    <div class="stat-card">
      <span class="stat-label">${it.label}</span>
      <span class="stat-value ${it.cls}">${it.raw ? it.value : it.value}</span>
    </div>
  `).join("");
}

// Similarity Matrix
function renderMatrix(matrix, names) {
  const shortName = n => n.length > 14 ? n.slice(0,13) + "…" : n;

  let html = `<table class="matrix-table">
    <thead><tr>
      <th class="row-header"></th>
      ${names.map(n => `<th title="${n}">${shortName(n)}</th>`).join("")}
    </tr></thead><tbody>`;

  matrix.forEach((row, i) => {
    html += `<tr><th class="row-header" title="${names[i]}">${shortName(names[i])}</th>`;
    row.cells.forEach(cell => {
      let cls = "cell-low";
      if (cell.is_self)          cls = "cell-self";
      else if (cell.is_flagged)  cls = "cell-flagged";
      else if (cell.score >= 50) cls = "cell-high";
      else if (cell.score >= 25) cls = "cell-mid";

      const label = cell.is_self ? "—" : fmt(cell.score);
      html += `<td class="matrix-cell ${cls}" title="${cell.score.toFixed(1)}%">${label}</td>`;
    });
    html += `</tr>`;
  });

  html += `</tbody></table>`;
  matrixWrapper.innerHTML = html;
}

// Flagged Pairs
function renderFlagged(flagged, thresholdPct) {
  flaggedBadge.textContent = flagged.length + (flagged.length === 1 ? " pair" : " pairs");
  flaggedBadge.className = "badge " + (flagged.length > 0 ? "badge-danger" : "badge-success");

  if (flagged.length === 0) {
    flaggedList.innerHTML = `
      <div class="flagged-empty">
        <span class="check">✅</span>
        No document pairs exceed the ${thresholdPct}% threshold.<br>
        <small style="color:var(--text-muted);font-size:13px;font-weight:400">All documents appear to be original.</small>
      </div>`;
    return;
  }

  flaggedList.innerHTML = flagged.map((pair, i) => `
    <div class="flagged-pair">
      <div class="pair-rank">#${i + 1}</div>
      <div class="pair-docs">
        <div class="pair-names">
          <span>${pair.doc1}</span>
          <span class="pair-arrow">↔</span>
          <span>${pair.doc2}</span>
        </div>
        <div class="pair-label">⚠ Potential Plagiarism Detected</div>
        <div class="pair-bar-wrap">
          <div class="pair-bar-bg">
            <div class="pair-bar-fill" style="width:${clamp(pair.similarity,0,100)}%"></div>
          </div>
          <div class="pair-pct">${pair.similarity.toFixed(1)}%</div>
        </div>
      </div>
    </div>
  `).join("");
}

// Sentence Comparison
function renderSentences(sentences, topPair) {
  if (!sentences || sentences.length === 0) {
    sentenceSection.classList.add("hidden");
    return;
  }
  sentenceSection.classList.remove("hidden");
  sentenceIntro.textContent =
    `Showing the most similar sentences between "${topPair.doc1}" and "${topPair.doc2}" (overall similarity: ${topPair.similarity.toFixed(1)}%).`;

  sentenceList.innerHTML = sentences.map(s => `
    <div class="sent-pair">
      <div class="sent-score-row">
        <span class="sent-score-label">Match strength</span>
        <span class="sent-score-badge">${s["Similarity (%)"].toFixed(1)}%</span>
      </div>
      <div class="sent-row">
        <span class="sent-tag sent-tag-a">Doc A</span>
        <span class="sent-text">${escHtml(s["Sentence (Doc A)"])}</span>
      </div>
      <div class="sent-row">
        <span class="sent-tag sent-tag-b">Doc B</span>
        <span class="sent-text">${escHtml(s["Sentence (Doc B)"])}</span>
      </div>
    </div>
  `).join("");
}

// Heatmap
function renderHeatmap(b64) {
  heatmapImg.src = b64;
  downloadBtn.onclick = () => {
    const a = document.createElement("a");
    a.href = b64;
    a.download = "plagiarism_heatmap.png";
    a.click();
  };
}

// ── Reset ────────────────────────────────────────────────────
resetBtn.addEventListener("click", () => {
  selectedFiles = [];
  renderFileList();
  updateAnalyzeBtn();
  resultsSection.classList.add("hidden");
  document.getElementById("upload-section").scrollIntoView({ behavior: "smooth" });
});

// ── Escape HTML ──────────────────────────────────────────────
function escHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
