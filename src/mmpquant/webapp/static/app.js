/* MMP-Quant web UI — vanilla JS, no build step, so `mmpquant serve` is the
 * only thing anyone needs to run this.
 *
 * Score-bar colour is magnitude, so it uses ONE hue (blue) from the dataviz
 * skill's validated ramp (references/palette.md), snapped to documented steps
 * rather than interpolated. On the dark theme the ramp's anchor flips —
 * brighter = more — and only steps clearing 3:1 against the chart track are
 * used (step 500 and lighter; checked with the skill's contrast()). The value
 * is always printed next to the bar, so colour is never the only channel.
 */

const SEQ_RAMP_DARK = [
  "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb",
];

const MAIN = document.getElementById("main");
const VIDEO_LIST = document.getElementById("video-list");
const DROPZONE = document.getElementById("dropzone");
const FILE_INPUT = document.getElementById("file-input");

let STAGES = [];
let activeVideoId = null;
let pollTimer = null;
let CURRENT_REPORT = null;
let OPEN_SEGMENT = null;
let ACTIVE_PRINCIPLE = "redundancy";   // which tab of a report is showing
let DRAWER_PRINCIPLE = "redundancy";   // which principle the open drawer explains

// ------------------------------------------------------------------ helpers

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s ?? "";
  return d.innerHTML;
}
function seqColor(score) {
  const n = SEQ_RAMP_DARK.length - 1;
  return SEQ_RAMP_DARK[Math.max(0, Math.min(n, Math.round(((score || 0) / 100) * n)))];
}
function timecode(seconds) {
  const m = Math.floor(seconds / 60);
  return `${m}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
}
const fmtScore = (x) => (x === null || x === undefined ? "N/A" : Number(x).toFixed(2));
const f2 = (x) => (x === null || x === undefined ? "—" : Number(x).toFixed(2));
const f4 = (x) => (x === null || x === undefined ? "—" : Number(x).toFixed(4));
const pct = (x) => (x === null || x === undefined ? "—" : `${Math.round(x * 100)}%`);
const cap = (s) => (s || "").replace(/^./, (c) => c.toUpperCase());

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

// ------------------------------------------------------------------ icons & illustrations
//
// Inline SVG only (no assets to host). Icons draw with currentColor so they
// take the colour of whatever chip they sit in.

const ICONS = {
  eye: '<path d="M2 12s3.6-6.5 10-6.5S22 12 22 12s-3.6 6.5-10 6.5S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
  mic: '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21"/>',
  link: '<path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1"/><path d="M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1"/>',
  gauge: '<path d="M4 17a8 8 0 1 1 16 0"/><path d="M12 17l4-5"/><circle cx="12" cy="17" r="1.2"/>',
  doc: '<path d="M7 3h7l5 5v13H7z"/><path d="M14 3v5h5M10 13h6M10 17h4"/>',
  clock: '<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>',
  check: '<path d="M5 12.5l4.2 4.2L19 7"/>',
  x: '<path d="M6 6l12 12M18 6L6 18"/>',
  chevron: '<path d="M9 6l6 6-6 6"/>',
  left: '<path d="M15 6l-6 6 6 6"/>',
  sigma: '<path d="M17 5H7l6 7-6 7h10"/>',
  timeline: '<path d="M4 6h10M4 12h16M4 18h7"/>',
  spark: '<path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5L18 18M18 6l-2.5 2.5M8.5 15.5L6 18"/>',
  warn: '<path d="M12 4l9 16H3z"/><path d="M12 10v4M12 17v.5"/>',
  info: '<circle cx="12" cy="12" r="8.5"/><path d="M12 11v5M12 8v.5"/>',
  layers: '<path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/>',
  music: '<path d="M9 18V5l11-2v13"/><circle cx="6.5" cy="18" r="2.5"/><circle cx="17.5" cy="16" r="2.5"/>',
  target: '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="4.5"/><circle cx="12" cy="12" r="1"/>',
  copy: '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M4 16V6a2 2 0 0 1 2-2h10"/>',
};
function icon(name, size = 18) {
  return `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name]}</svg>`;
}

// A screen showing text, a voice saying it, and the link between them.
const HERO_ILLUSTRATION = `
<svg class="float" width="300" height="170" viewBox="0 0 300 170" aria-hidden="true">
  <defs>
    <linearGradient id="hg" x1="0" x2="1"><stop offset="0" stop-color="#7c6cff"/><stop offset="1" stop-color="#5fd8ef"/></linearGradient>
  </defs>
  <rect x="18" y="22" width="132" height="92" rx="12" fill="#181f37" stroke="#36406a"/>
  <rect x="62" y="114" width="44" height="8" rx="3" fill="#283152"/>
  <rect x="34" y="40" width="80" height="8" rx="4" fill="#5fd8ef"/>
  <rect x="34" y="56" width="98" height="6" rx="3" fill="#36406a"/>
  <rect x="34" y="68" width="88" height="6" rx="3" fill="#36406a"/>
  <rect x="34" y="80" width="64" height="6" rx="3" fill="#36406a"/>
  <rect x="34" y="94" width="40" height="8" rx="4" fill="#a397ff" opacity=".8"/>
  <path d="M152 68 C 178 40, 200 40, 214 58" fill="none" stroke="url(#hg)" stroke-width="2.5" stroke-dasharray="5 6"/>
  <circle cx="214" cy="58" r="4" fill="#5fd8ef"/>
  <rect x="196" y="62" width="88" height="62" rx="16" fill="#181f37" stroke="#36406a"/>
  <path d="M214 124 l-6 14 16-14z" fill="#181f37" stroke="#36406a"/>
  <g stroke-width="4" stroke-linecap="round">
    <path d="M214 90v6" stroke="#7c6cff"/><path d="M224 85v16" stroke="#8d8afa"/><path d="M234 78v30" stroke="#9fa9f5"/>
    <path d="M244 86v14" stroke="#7fc0f2"/><path d="M254 80v26" stroke="#6fcdf0"/><path d="M264 89v8" stroke="#5fd8ef"/>
  </g>
  <g fill="#ffc864"><circle cx="168" cy="26" r="2.5"/><circle cx="276" cy="40" r="2"/><circle cx="30" cy="140" r="2"/></g>
  <path d="M176 18l2 5 5 2-5 2-2 5-2-5-5-2 5-2z" fill="#ff7ab6"/>
</svg>`;

const STAGE_ICON = { queued: "clock", asr: "mic", ocr: "eye", audio: "music", aligning: "link", scoring: "gauge", writing_report: "doc" };

// ------------------------------------------------------------------ sidebar

async function loadStages() {
  STAGES = await api("/api/stages");
}

async function loadVideoList() {
  const videos = await api("/api/videos");
  if (videos.length === 0) {
    VIDEO_LIST.innerHTML = `<div class="muted">None scored yet.</div>`;
    return;
  }
  VIDEO_LIST.innerHTML = videos
    .map((v) => {
      const na = v.score === null;
      return `<div class="video-item ${v.video_id === activeVideoId ? "active" : ""}" data-video-id="${escapeHtml(v.video_id)}"
          role="button" tabindex="0">
        <span class="vid-name">${escapeHtml(v.video_id)}</span>
        <span class="vid-mini">${na ? "" : `<span style="width:${Math.max(3, v.score)}%; background:${seqColor(v.score)}"></span>`}</span>
        <span class="vid-score tabular">${na ? "N/A" : v.score.toFixed(1)}</span>
      </div>`;
    })
    .join("");
  VIDEO_LIST.querySelectorAll(".video-item").forEach((el) => {
    el.addEventListener("click", () => selectVideo(el.dataset.videoId));
    el.addEventListener("keydown", (e) => { if (e.key === "Enter") selectVideo(el.dataset.videoId); });
  });
}

async function selectVideo(videoId) {
  activeVideoId = videoId;
  closeDrawer();
  loadVideoList();
  MAIN.innerHTML = `<div class="loading"><span class="spinner"></span>&nbsp; Loading report…</div>`;
  try {
    renderReport(await api(`/api/videos/${encodeURIComponent(videoId)}/report`));
  } catch (e) {
    MAIN.innerHTML = `<div class="error-box">Could not load report: ${escapeHtml(e.message)}</div>`;
  }
}

// ------------------------------------------------------------------ empty state

function renderWelcome() {
  const steps = [
    ["eye", "chip-cyan", "Read the screen", "OCR finds the on-screen text and keeps only full sentences — titles and labels are signaling, not redundancy."],
    ["mic", "chip-pink", "Listen to the narration", "Speech recognition writes down what is said, with a time for every word."],
    ["link", "chip-violet", "Compare sentence by sentence", "A sentence counts only if the narration restates it — more than chance, at that moment."],
    ["gauge", "chip-amber", "Score 0–100", "Higher means more on-screen text that duplicates the narration. Every number is explained."],
  ];
  MAIN.innerHTML = `
    <div class="hero">
      ${HERO_ILLUSTRATION}
      <h1>Is the screen just <span>repeating the narration</span>?</h1>
      <p>Upload an educational video, or pick one on the left, to see how much of its on-screen
        text duplicates what is being said — Mayer's redundancy principle, measured.</p>
      <div class="how">
        ${steps.map(([ic, chip, t, d], i) => `<div class="how-step">
          <span class="icon-chip ${chip}">${icon(ic, 20)}</span>
          <div class="title">${t}</div><div class="desc">${d}</div>
          <div class="num" style="margin-top:8px">STEP ${i + 1}</div></div>`).join("")}
      </div>
    </div>`;
}

// ------------------------------------------------------------------ upload

DROPZONE.addEventListener("click", () => FILE_INPUT.click());
DROPZONE.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") FILE_INPUT.click(); });
FILE_INPUT.addEventListener("change", () => { if (FILE_INPUT.files[0]) uploadVideo(FILE_INPUT.files[0]); });
["dragover", "dragenter"].forEach((evt) =>
  DROPZONE.addEventListener(evt, (e) => { e.preventDefault(); DROPZONE.classList.add("dragover"); }));
["dragleave", "drop"].forEach((evt) =>
  DROPZONE.addEventListener(evt, (e) => { e.preventDefault(); DROPZONE.classList.remove("dragover"); }));
DROPZONE.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) uploadVideo(f); });

async function uploadVideo(file) {
  activeVideoId = null;
  closeDrawer();
  MAIN.innerHTML = `<div class="card"><h2><span class="card-icon">${icon("layers")}</span>Uploading ${escapeHtml(file.name)}</h2>
    <p class="card-sub"><span class="spinner"></span>&nbsp; Sending the file — large videos take a moment.</p></div>`;
  const form = new FormData();
  form.append("file", file);
  let job;
  try {
    const res = await fetch("/api/videos", { method: "POST", body: form });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
    job = await res.json();
  } catch (e) {
    MAIN.innerHTML = `<div class="error-box">Upload failed: ${escapeHtml(e.message)}</div>`;
    return;
  }
  activeVideoId = job.video_id;
  renderStepper(job);
  pollJob(job.job_id);
}

function pollJob(jobId) {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    let job;
    try { job = await api(`/api/jobs/${jobId}`); } catch { return; }
    renderStepper(job);
    if (job.status === "done") {
      clearInterval(pollTimer); pollTimer = null;
      renderReport(await api(`/api/jobs/${jobId}/report`));
      loadVideoList();
    } else if (job.status === "failed") {
      clearInterval(pollTimer); pollTimer = null;
      MAIN.innerHTML = `<div class="card"><h2>Scoring failed — ${escapeHtml(job.video_id)}</h2>
        <div class="error-box">${escapeHtml(job.error || "unknown error")}</div></div>`;
    }
  }, 2000);
}

function renderStepper(job) {
  const steps = STAGES.filter((s) => s.key !== "done").map((s, i) => {
    const state = job.stage_index > i ? "done" : job.stage_index === i ? "active" : "";
    return `<div class="step ${state}">
      <span class="icon-chip">${icon(state === "done" ? "check" : STAGE_ICON[s.key] || "clock", 19)}</span>
      <div><div class="step-label">${escapeHtml(s.label)}</div><div class="step-desc">${escapeHtml(s.description)}</div></div>
    </div>`;
  }).join("");
  MAIN.innerHTML = `<div class="card">
      <h2><span class="card-icon">${icon("spark")}</span>Scoring ${escapeHtml(job.video_id)}</h2>
      <p class="card-sub">Speech recognition and OCR each take one to several minutes, depending on video length.</p>
      <div class="stepper">${steps}</div>
    </div>`;
}

// ------------------------------------------------------------------ report

function renderReport(data) {
  CURRENT_REPORT = data;
  if (!data.coherence) ACTIVE_PRINCIPLE = "redundancy";
  const sections = ACTIVE_PRINCIPLE === "coherence" ? coherenceSections(data) : redundancySections(data);
  MAIN.innerHTML = principleTabs(data) + sections.filter(Boolean).join("\n");

  MAIN.querySelectorAll("[data-principle]").forEach((b) => b.addEventListener("click", () => {
    if (b.disabled) return;
    ACTIVE_PRINCIPLE = b.dataset.principle;
    renderReport(CURRENT_REPORT);
  }));
  MAIN.querySelectorAll("[data-segment]").forEach((el) => {
    const open = () => openDrawer(Number(el.dataset.segment), el.dataset.principleOf || "redundancy");
    el.addEventListener("click", open);
    el.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); }
    });
  });
  // animate the gauge in
  requestAnimationFrame(() => {
    const arc = MAIN.querySelector(".ring-arc");
    if (arc) arc.style.strokeDashoffset = arc.dataset.target;
  });
}

function redundancySections(data) {
  const scored = data.score !== null;
  return [
    reportHeader(data),
    reliabilitySection(data),
    scored ? segmentChart(data) : "",
    scored ? worstSegmentsSection(data) : "",
    scored ? exclusionsSection(data) : "",
    methodologySection(data),
  ];
}

function principleTabs(data) {
  const c = data.coherence;
  const tab = (key, label, ic, score, hint) => `<button class="principle-tab ${ACTIVE_PRINCIPLE === key ? "active" : ""}"
      data-principle="${key}" ${score === undefined ? "disabled" : ""} role="tab" aria-selected="${ACTIVE_PRINCIPLE === key}">
      <span class="pt-icon">${icon(ic, 18)}</span>
      <span class="pt-text"><span class="pt-label">${label}</span><span class="pt-hint">${hint}</span></span>
      <span class="pt-score tabular">${score === undefined ? "—" : fmtScore(score)}</span></button>`;
  return `<div class="principle-tabs" role="tablist" aria-label="Mayer principle">
    ${tab("redundancy", "Redundancy", "copy", data.score, "on-screen text repeating the narration")}
    ${tab("coherence", "Coherence", "music", c ? c.score : undefined,
      c ? "background music &amp; off-topic narration" : "not scored in this run — re-score the video")}
  </div>`;
}

function scoreRing(score) {
  const r = 62, c = 2 * Math.PI * r;
  const target = score === null ? c : c * (1 - Math.max(0, Math.min(100, score)) / 100);
  return `<div class="ring">
    <svg width="148" height="148" viewBox="0 0 148 148" aria-hidden="true">
      <defs><linearGradient id="rg" x1="0" x2="1" y1="0" y2="1">
        <stop offset="0" stop-color="#7c6cff"/><stop offset="1" stop-color="#5fd8ef"/></linearGradient></defs>
      <circle cx="74" cy="74" r="${r}" fill="none" stroke="#181f37" stroke-width="12"/>
      <circle class="ring-arc" cx="74" cy="74" r="${r}" fill="none" stroke="url(#rg)" stroke-width="12"
        stroke-linecap="round" stroke-dasharray="${c}" stroke-dashoffset="${c}" data-target="${target}"/>
    </svg>
    <div class="ring-value"><div><strong class="tabular">${score === null ? "N/A" : score.toFixed(1)}</strong>
      <span>out of 100</span></div></div>
  </div>`;
}

function reportHeader(data) {
  const isNA = data.score === null;
  const warnings = (data.warnings || [])
    .map((w) => `<div class="warning-item">${icon("info", 16)}<div>${escapeHtml(w)}</div></div>`).join("");
  return `<div class="card">
    <div class="score-hero">
      ${scoreRing(data.score)}
      <div class="score-meta">
        <h2>${escapeHtml(data.video_id)}
          <span class="badge ${isNA ? "badge-na" : "badge-ok"}">${icon(isNA ? "warn" : "check", 13)} ${isNA ? "N/A" : "Scored"}</span></h2>
        <div class="score-meaning">Redundancy score — higher means more on-screen text duplicating the narration (worse, per Mayer).</div>
        ${isNA ? `<p class="muted" style="margin-top:10px">${escapeHtml(data.na_reason || "")}<br/>
          <strong>N/A is not zero:</strong> zero means assessed and found free of redundancy; N/A means it could not be assessed.</p>` : ""}
        <dl class="stat-row">
          <div class="stat"><dt>Duration</dt><dd class="tabular">${timecode(data.duration)}</dd></div>
          <div class="stat"><dt>Segments</dt><dd>${data.segments.length}</dd></div>
          <div class="stat"><dt>Windows assessed</dt><dd class="tabular">${data.n_windows_qualifying} / ${data.n_windows_total}</dd></div>
          ${!isNA && data.coverage != null ? `<div class="stat"><dt>Time with text to assess</dt><dd class="tabular">${pct(data.coverage)}</dd></div>` : ""}
          ${!isNA && data.assessed_score != null ? `<div class="stat"><dt>Score where text was on screen</dt><dd class="tabular">${fmtScore(data.assessed_score)}</dd></div>` : ""}
        </dl>
        ${!isNA && data.assessed_score != null && data.coverage != null ? `<p class="muted score-explain">Video score = average over every 10-second window, a window with nothing on screen to assess counting as 0
          = ${fmtScore(data.assessed_score)} × ${(data.coverage * 100).toFixed(1)}% = <strong>${fmtScore(data.score)}</strong>.</p>` : ""}
      </div>
    </div>
    ${warnings}
  </div>`;
}

function reliabilitySection(data) {
  const rel = data.reliability || {};
  if (!rel.reliability || rel.reliability === "ok") return "";
  if (rel.reliability === "no_text_detected") {
    return `<div class="card"><div class="banner banner-warning">${icon("info", 20)}
      <div><strong>No on-screen text detected</strong>${escapeHtml(rel.note || "")}</div></div></div>`;
  }
  return `<div class="card"><div class="banner banner-critical">${icon("warn", 20)}
    <div><strong>OCR unreliable — this score is not trustworthy</strong>
      Text detected across ${pct(rel.detection_share)} of the video (${escapeHtml(rel.basis_phrase || "")}); only
      ${pct(rel.detection_recovery_rate)} of detected regions could be read. A low score means the text could not be
      read, <strong>not</strong> that the video is free of redundancy. Inspect frames before reporting it.</div></div></div>`;
}

function segmentChart(data) {
  const rows = data.segments.map((s) => {
    const score = s.score ?? 0;
    return `<div class="chart-row" data-segment="${s.index}" role="button" tabindex="0"
        aria-label="Show how segment ${s.index} (${escapeHtml(s.timecode)}) was scored">
      <div class="chart-label tabular">${escapeHtml(s.timecode)}</div>
      <div class="chart-track">
        <div class="chart-bar" style="width:${Math.max(1, score)}%; background:${seqColor(score)}; animation-delay:${s.index * 60}ms"></div>
        <span class="chart-bar-value tabular">${fmtScore(s.score)}</span>
      </div>
      <span class="chart-go">${icon("chevron", 16)}</span>
    </div>`;
  }).join("");
  return `<div class="card">
    <h2><span class="card-icon">${icon("timeline")}</span>Segment timeline</h2>
    <p class="card-sub">Segments are the unit your raters score. <strong style="color:var(--ink)">Click any segment</strong>
      to see its on-screen text, the narration, and the full calculation.</p>
    ${rows}
    <div class="chart-axis"><span></span><div><span>0</span><span>25</span><span>50</span><span>75</span><span>100</span></div><span></span></div>
  </div>`;
}

function windowEvidence(w) {
  return `<div class="window-evidence">
    <div class="window-evidence-meta tabular"><strong>${escapeHtml(w.timecode)}</strong> · score ${fmtScore(w.score)}
      ${(w.restated_sentences || []).length ? ` · restates <strong>${escapeHtml(w.restated_sentences.map((s) => `“${s}”`).join("; "))}</strong>` : ""}</div>
    <div class="evidence-grid">
      <div class="evidence-col screen"><div class="evidence-col-title">${icon("eye", 14)} On screen</div>${escapeHtml(w.onscreen_text) || "<em>(none)</em>"}</div>
      <div class="evidence-col voice"><div class="evidence-col-title">${icon("mic", 14)} Narration</div>${escapeHtml(w.narration_text) || "<em>(none)</em>"}</div>
    </div>
    ${(w.shared_words || []).length ? `<div class="excluded-note">Narration repeated: ${escapeHtml(w.shared_words.join(", "))}</div>` : ""}
  </div>`;
}

function worstSegmentsSection(data) {
  if (!data.worst_segments.length) {
    return `<div class="card"><h2><span class="card-icon">${icon("check")}</span>Worst-offending segments</h2>
      <p class="card-sub" style="margin:0">No segment carried measurable redundancy — no on-screen sentence was restated by the narration.</p></div>`;
  }
  const blocks = data.worst_segments.map((s) => `<div class="segment-block">
      <button class="chip-btn tabular" data-segment="${s.segment_index}">Segment ${s.segment_index} · ${escapeHtml(s.timecode)} · ${fmtScore(s.score)} →</button>
      ${s.windows.map(windowEvidence).join("")}
    </div>`).join("");
  return `<div class="card"><h2><span class="card-icon">${icon("spark")}</span>Worst-offending segments</h2>
    <p class="card-sub">The evidence behind the highest scores. Read the two columns and judge whether the narration duplicates the screen.</p>
    ${blocks}</div>`;
}

function exclusionsSection(data) {
  if (!data.exclusions.length) {
    return `<div class="card"><h2><span class="card-icon">${icon("layers")}</span>Windows not assessed</h2>
      <p class="card-sub" style="margin:0">Every window was assessed.</p></div>`;
  }
  return `<div class="card"><h2><span class="card-icon">${icon("layers")}</span>Windows not assessed</h2>
    <table class="plain"><thead><tr><th>Reason</th><th class="num">Windows</th></tr></thead><tbody>
    ${data.exclusions.map((e) => `<tr><td>${escapeHtml(cap(e.label))}</td><td class="num tabular">${e.count}</td></tr>`).join("")}
    </tbody></table></div>`;
}

function methodologySection(data) {
  const m = data.methodology;
  const rows = [
    ["Window size", `${m.window_size_s} s`], ["Segment length", `${m.segment_length_s} s`],
    ["Alignment tolerance", `±${m.alignment_tolerance_s} s`], ["Frame sampling", `${m.frame_fps} fps`],
    ["Sentence must be restated", `more than ${pct(m.min_sentence_coverage)} of its content words`],
    ["Minimum words said", m.min_shared_content_words], ["Chance baseline", `narration > ${m.null_exclusion_s} s away`],
    ["Word-match weight", m.weight_verbatim], ["Meaning weight", m.weight_semantic],
    ["Embedding model", m.embedding_model], ["Speech recognition", m.asr_backend], ["OCR", m.ocr_backend],
  ].map(([k, v]) => `<tr><td>${escapeHtml(k)}</td><td class="num tabular">${escapeHtml(String(v))}</td></tr>`).join("");
  return `<div class="card"><details class="methodology">
    <summary><span class="card-icon">${icon("doc")}</span>Methodology appendix ${icon("chevron", 16)}</summary>
    <p class="muted">Only text classified as a <strong>sentence</strong> is compared — labels, key terms, titles and goal
      statements are signaling under Mayer, not redundancy.</p>
    <table class="plain"><tbody>${rows}</tbody></table></details></div>`;
}

// ------------------------------------------------------------------ segment drawer
//
// A side panel with its own scroll area: the header (segment, score, close)
// and the segment tabs stay put while the calculation scrolls beneath them,
// and every segment is one click (or ←/→) away without closing anything.
// Replaces the first version, a centred pop-up that was 3,000+ px tall per
// segment and froze the page behind it.

function drawerSegments(principle) {
  const data = CURRENT_REPORT;
  if (!data) return [];
  return principle === "coherence" ? (data.coherence?.segments || []) : data.segments;
}

function openDrawer(index, principle = DRAWER_PRINCIPLE) {
  const data = CURRENT_REPORT;
  if (!data) return;
  DRAWER_PRINCIPLE = principle;
  const segments = drawerSegments(principle);
  const seg = segments.find((s) => s.index === index);
  if (!seg) return;
  OPEN_SEGMENT = index;

  let drawer = document.querySelector(".drawer");
  if (!drawer) {
    const backdrop = document.createElement("div");
    backdrop.className = "drawer-backdrop";
    backdrop.addEventListener("click", closeDrawer);
    drawer = document.createElement("aside");
    drawer.className = "drawer";
    drawer.setAttribute("role", "dialog");
    drawer.setAttribute("aria-modal", "true");
    drawer.setAttribute("aria-label", "Segment calculation");
    document.body.append(backdrop, drawer);
    document.documentElement.classList.add("drawer-open");
  }

  drawer.dataset.principle = principle;
  const pos = segments.findIndex((s) => s.index === index);
  const prev = segments[pos - 1], next = segments[pos + 1];
  drawer.innerHTML = `
    <div class="drawer-head">
      <div class="drawer-title-row">
        <button class="icon-btn" data-nav="${prev ? prev.index : ""}" ${prev ? "" : "disabled"} aria-label="Previous segment">${icon("left")}</button>
        <button class="icon-btn" data-nav="${next ? next.index : ""}" ${next ? "" : "disabled"} aria-label="Next segment">${icon("chevron")}</button>
        <div>
          <div class="drawer-kicker">${escapeHtml(data.video_id)} · ${principle} · segment ${pos + 1} of ${segments.length}</div>
          <div class="drawer-title tabular">${escapeHtml(seg.timecode)}</div>
        </div>
        <div class="drawer-score"><strong class="tabular">${fmtScore(seg.score)}</strong><span>segment score</span></div>
        <button class="icon-btn" data-close aria-label="Close">${icon("x")}</button>
      </div>
    </div>
    <nav class="seg-tabs" aria-label="Segments">
      ${segments.map((s) => `<button class="seg-tab ${s.index === index ? "active" : ""}" data-nav="${s.index}">
        <span class="t tabular">${escapeHtml(s.timecode)}</span>
        <span class="bar"><i style="width:${Math.max(2, s.score || 0)}%; background:${seqColor(s.score || 0)}"></i></span>
        <span class="s tabular">${fmtScore(s.score)}</span></button>`).join("")}
    </nav>
    <div class="drawer-body">${principle === "coherence" ? coherenceSegmentDetail(seg, data.coherence) : segmentDetail(seg, data.methodology)}</div>`;

  drawer.querySelectorAll("[data-nav]").forEach((b) => b.addEventListener("click", () => {
    if (b.dataset.nav !== "") openDrawer(Number(b.dataset.nav));
  }));
  drawer.querySelector("[data-close]").addEventListener("click", closeDrawer);
  drawer.querySelector(".seg-tab.active")?.scrollIntoView({ block: "nearest", inline: "center" });
  drawer.querySelector(".drawer-body").scrollTop = 0;
  drawer.querySelector("[data-close]").focus({ preventScroll: true });
}

function closeDrawer() {
  document.querySelectorAll(".drawer, .drawer-backdrop").forEach((el) => el.remove());
  document.documentElement.classList.remove("drawer-open");
  OPEN_SEGMENT = null;
}

document.addEventListener("keydown", (e) => {
  if (OPEN_SEGMENT === null) return;
  if (e.key === "Escape") closeDrawer();
  const segs = drawerSegments(DRAWER_PRINCIPLE);
  const pos = segs.findIndex((s) => s.index === OPEN_SEGMENT);
  if (e.key === "ArrowRight" && segs[pos + 1]) openDrawer(segs[pos + 1].index);
  if (e.key === "ArrowLeft" && segs[pos - 1]) openDrawer(segs[pos - 1].index);
});

function segmentDetail(seg, m) {
  if (!seg.windows) {
    return `<div class="error-box">This report came from an older server that does not send the window breakdown.
      Restart it (<code>mmpquant serve</code>) and reload the page.</div>`;
  }
  const firstScored = seg.windows.findIndex((w) => w.gate === "passed");
  return segmentSummary(seg) +
    `<div class="drawer-section-title" style="margin-top:22px"><span class="card-icon">${icon("layers", 16)}</span>
      Window by window <span class="muted" style="font-weight:400">— click a window to open its calculation</span></div>` +
    seg.windows.map((w, i) => windowCalculation(w, m, i === (firstScored === -1 ? 0 : firstScored))).join("");
}

function segmentSummary(seg) {
  const rows = seg.windows.map((w) => {
    const contrib = w.qualifies ? w.score * w.duration : 0;
    return `<tr class="${w.qualifies ? "" : "row-muted"}">
      <td class="tabular">${escapeHtml(w.timecode)}</td>
      <td>${w.qualifies ? (w.gate === "passed" ? "Restates a sentence" : "Assessed — nothing restated") : "Nothing to assess — " + escapeHtml(w.exclusion_label || "") + " (counts 0)"}</td>
      <td class="num tabular">${w.qualifies ? fmtScore(w.score) : "0"}</td>
      <td class="num tabular">${f2(w.duration)} s</td>
      <td class="num tabular">${f2(contrib)}</td></tr>`;
  }).join("");
  let formula;
  if (!seg.windows.length) formula = "No measurement window falls in this segment.";
  else {
    const num = seg.windows.reduce((a, w) => a + (w.qualifies ? w.score * w.duration : 0), 0);
    const den = seg.windows.reduce((a, w) => a + w.duration, 0);
    formula = `(${seg.windows.map((w) => `${w.qualifies ? f2(w.score) : "0"} × ${f2(w.duration)}`).join(" + ")}) ÷ (${seg.windows.map((w) => f2(w.duration)).join(" + ")})
      <br/>= ${f2(num)} ÷ ${f2(den)} = <strong>${fmtScore(seg.score)}</strong>`;
  }
  return `<div class="drawer-section-title"><span class="card-icon">${icon("sigma", 16)}</span>How this segment's score is calculated</div>
    <div class="note">The segment score is the duration-weighted average of <strong>all</strong> its 10-second windows.
      A window with no on-screen sentence or no narration had nothing on screen to duplicate the narration, so it counts as 0;
      so does a window that was assessed but restates nothing. The whole video's score is the same average over every window.</div>
    <div class="table-scroll"><table class="plain"><thead><tr><th>Window</th><th>Status</th><th class="num">Score</th>
      <th class="num">Duration</th><th class="num">Score × duration</th></tr></thead><tbody>${rows}</tbody></table></div>
    <div class="formula tabular">Segment score = ${formula}</div>`;
}

function checkResult(c, m) {
  const need = m.min_sentence_coverage ?? 0.5, minShared = m.min_shared_content_words ?? 2;
  if (c.passed) return `<span class="result-pass">${icon("check", 14)} Restated</span>`;
  if (c.is_question && !c.shared_words.length) return `<span class="result-fail">✗ Not asked aloud as a question</span>`;
  if (c.shared_words.length < minShared) return `<span class="result-fail">✗ Fewer than ${minShared} words said</span>`;
  if (c.coverage <= need) return `<span class="result-fail">✗ Not more than ${pct(need)}</span>`;
  return `<span class="result-fail">✗ Not above chance</span>`;
}

function chanceRow(name, raw, chance, adj) {
  if (raw === null || raw === undefined) return "";
  const working = raw <= chance
    ? `${f4(raw)} is not above chance ${f4(chance)} → <strong>0</strong>`
    : `(${f4(raw)} − ${f4(chance)}) ÷ (1 − ${f4(chance)}) = <strong>${f4(adj)}</strong>`;
  return `<tr><td>${name}</td><td class="num tabular">${f4(raw)}</td><td class="num tabular">${f4(chance)}</td><td class="tabular">${working}</td></tr>`;
}

function windowCalculation(w, m, open) {
  const status = !w.qualifies ? ["badge-na", "Left out"] : w.gate === "passed" ? ["badge-scored", "Restates a sentence"] : ["badge-zero", "Nothing restated"];
  const texts = `<div class="evidence-grid" style="margin-top:12px">
      <div class="evidence-col screen"><div class="evidence-col-title">${icon("eye", 14)} On-screen sentences</div>${escapeHtml(w.onscreen_compared) || "<em>(no sentence on screen)</em>"}</div>
      <div class="evidence-col voice"><div class="evidence-col-title">${icon("mic", 14)} Narration (± ${escapeHtml(String(m.alignment_tolerance_s ?? 3))} s)</div>${escapeHtml(w.narration) || "<em>(no narration)</em>"}</div>
    </div>
    ${w.labels_excluded.length ? `<div class="note" style="margin:10px 0 0">Set aside as labels / key terms — signaling, never redundancy:</div>
      <div class="labels">${w.labels_excluded.map((l) => `<span>${escapeHtml(l)}</span>`).join("")}${w.n_labels_excluded > w.labels_excluded.length ? `<span>+${w.n_labels_excluded - w.labels_excluded.length} more</span>` : ""}</div>` : ""}`;

  let body;
  if (!w.qualifies) {
    body = `${texts}<div class="verdict">${icon("info", 18)}<div><strong>Not assessed.</strong> ${escapeHtml(cap(w.exclusion_label || ""))}.
      A window needs at least ${m.min_onscreen_words} words of on-screen sentence text and some narration, so there is nothing here to assess — it counts as 0 in the segment average.</div></div>`;
  } else {
    const step1 = `<div class="step-block"><div class="step-title"><span class="step-num">1</span>Does the narration restate an on-screen sentence?</div>
      <div class="note">Restated = the narration says at least ${m.min_shared_content_words ?? 2} of the sentence's content words,
        <strong>more than ${pct(m.min_sentence_coverage ?? 0.5)}</strong> of them, and more than narration elsewhere in the video
        does (chance: median over ${w.null_windows || 0} windows &gt; ${m.null_exclusion_s ?? 30} s away${w.null_windows ? "" : " — too few, so taken as 0"}).
        Content words skip "the", "is", "of"…; plurals and tenses match. A question counts only if it is asked aloud.</div>
      ${w.sentence_checks.length ? `<div class="table-scroll"><table class="plain"><thead><tr><th>On-screen sentence</th><th>Its content words</th>
        <th>Narration said</th><th class="num">Coverage</th><th class="num">Chance</th><th>Result</th></tr></thead><tbody>
        ${w.sentence_checks.map((c) => `<tr class="${c.passed ? "row-pass" : ""}"><td>${escapeHtml(c.sentence)}${c.is_question ? ' <span class="muted">(question)</span>' : ""}</td>
          <td>${escapeHtml(c.content_words.join(", "))}</td><td>${escapeHtml(c.shared_words.join(", ") || "—")}</td>
          <td class="num tabular">${c.shared_words.length}/${c.content_words.length} = ${pct(c.coverage)}</td>
          <td class="num tabular">${pct(c.chance)}</td><td>${checkResult(c, m)}</td></tr>`).join("")}
        </tbody></table></div>` : `<div class="note"><em>No full sentence on screen in this window to test.</em></div>`}</div>`;

    if (w.gate !== "passed") {
      body = `${texts}${step1}<div class="verdict">${icon("check", 18)}<div>No on-screen sentence is restated →
        <strong>window score = 0</strong> (assessed, no redundancy). It still counts as 0 in the segment average.</div></div>`;
    } else {
      const wv = m.weight_verbatim, ws = m.weight_semantic;
      body = `${texts}${step1}
        <div class="step-block"><div class="step-title"><span class="step-num">2</span>How closely does the narration match the restated sentence?</div>
          <div class="note">Word match = ROUGE-L F1 over content words (order matters). Meaning = cosine similarity of sentence
            embeddings (${escapeHtml(m.embedding_model)}). Only what is above chance counts: (raw − chance) ÷ (1 − chance).</div>
          <div class="table-scroll"><table class="plain"><thead><tr><th>Measure</th><th class="num">Raw</th><th class="num">Chance</th><th>Above chance</th></tr></thead><tbody>
            ${chanceRow("Word match", w.verbatim_raw, w.verbatim_chance, w.verbatim)}
            ${chanceRow("Meaning", w.semantic_raw, w.semantic_chance, w.semantic)}
          </tbody></table></div></div>
        <div class="step-block"><div class="step-title"><span class="step-num">3</span>Window score</div>
          <div class="formula tabular">100 × (${wv} × word match + ${ws} × meaning)<br/>
            = 100 × (${wv} × ${f4(w.verbatim)} + ${ws} × ${f4(w.semantic)})<br/>
            = 100 × (${f4(wv * w.verbatim)} + ${f4(ws * w.semantic)})<br/>
            = <strong>${fmtScore(w.score)}</strong></div></div>`;
    }
  }

  return `<details class="win" ${open ? "open" : ""}>
    <summary><span class="win-chev">${icon("chevron", 16)}</span><span class="win-time tabular">${escapeHtml(w.timecode)}</span>
      <span class="badge ${status[0]}">${status[1]}</span>
      <span class="win-score tabular">${w.qualifies ? fmtScore(w.score) : "—"}</span></summary>
    <div class="win-body">${body}</div></details>`;
}

// ------------------------------------------------------------------ coherence
//
// Same layout language as redundancy: a score ring, a clickable segment chart
// that opens the same drawer, and every number shown with its working. Bars are
// magnitude, so they use the same single blue ramp. The sound timeline is
// identity (speech vs music), so it uses categorical hues with a legend and
// lane labels — colour is never the only channel.

function coherenceSections(data) {
  const c = data.coherence;
  if (c.score === null) {
    return [coherenceHeader(data, c), coherenceMethod(c)];
  }
  return [coherenceHeader(data, c), coherenceChart(c), soundTimeline(data, c), offTopicCard(c), coherenceMethod(c)];
}

function partRow(label, ic, value, weight) {
  const v = value ?? 0;
  const experimental = Number(weight) === 0;
  return `<div class="part-row ${experimental ? "part-experimental" : ""}">
    <div class="part-label">${icon(ic, 16)}<span>${label}</span>
      ${experimental ? `<span class="badge badge-na">${icon("info", 12)} experimental · not in score</span>` : ""}
      <span class="muted">weight ${weight}</span></div>
    <div class="chart-track"><div class="chart-bar" style="width:${Math.max(1, v)}%; background:${seqColor(v)}"></div>
      <span class="chart-bar-value tabular">${fmtScore(value)}</span></div>
  </div>`;
}

function coherenceHeader(data, c) {
  const isNA = c.score === null;
  const m = c.methodology, w = m.weights;
  const warnings = (c.warnings || [])
    .map((x) => `<div class="warning-item">${icon("info", 16)}<div>${escapeHtml(x)}</div></div>`).join("");
  const parts = c.parts || {};
  return `<div class="card">
    <div class="score-hero">
      ${scoreRing(c.score)}
      <div class="score-meta">
        <h2>${escapeHtml(data.video_id)}
          <span class="badge ${isNA ? "badge-na" : "badge-ok"}">${icon(isNA ? "warn" : "check", 13)} ${isNA ? "N/A" : "Scored"}</span></h2>
        <div class="score-meaning">Coherence score — higher means more extraneous material competing with the lesson (worse, per Mayer).</div>
        ${isNA ? `<p class="muted" style="margin-top:10px">${escapeHtml(c.na_reason || "")}<br/>
          <strong>N/A is not zero:</strong> it means the video could not be assessed.</p>` : `
        <div class="parts">
          ${partRow("Background music under narration", "music", parts.background_music, w.background_music)}
          ${partRow("Off-topic narration", "target", parts.off_topic, w.off_topic)}
        </div>
        <p class="muted score-explain">Coherence = ${w.background_music} × ${fmtScore(parts.background_music)} + ${w.off_topic} × ${fmtScore(parts.off_topic)}
          = <strong>${fmtScore(c.score)}</strong>. Each part is averaged over every 10-second window; a window with no narration counts as 0.
          ${c.alternative && c.alternative.score != null ? `<br/>Pre-registered alternative (${c.alternative.weights.background_music} / ${c.alternative.weights.off_topic}, off-topic included):
            <strong>${fmtScore(c.alternative.score)}</strong> — kept to compare against the professor's scores, not the reported score.` : ""}</p>`}
      </div>
    </div>
    ${warnings}
  </div>`;
}

function coherenceChart(c) {
  const rows = c.segments.map((s) => {
    const score = s.score ?? 0;
    return `<div class="chart-row" data-segment="${s.index}" data-principle-of="coherence" role="button" tabindex="0"
        aria-label="Show how coherence segment ${s.index} (${escapeHtml(s.timecode)}) was scored">
      <div class="chart-label tabular">${escapeHtml(s.timecode)}</div>
      <div class="chart-track">
        <div class="chart-bar" style="width:${Math.max(1, score)}%; background:${seqColor(score)}; animation-delay:${s.index * 60}ms"></div>
        <span class="chart-bar-value tabular">${fmtScore(s.score)}</span>
      </div>
      <span class="chart-go">${icon("chevron", 16)}</span>
    </div>`;
  }).join("");
  return `<div class="card">
    <h2><span class="card-icon">${icon("timeline")}</span>Coherence segment timeline</h2>
    <p class="card-sub"><strong style="color:var(--ink)">Click any segment</strong> to see, window by window, how much of the
      speech had music under it, whether the narration stayed on topic, and the full calculation.</p>
    ${rows}
    <div class="chart-axis"><span></span><div><span>0</span><span>25</span><span>50</span><span>75</span><span>100</span></div><span></span></div>
  </div>`;
}

// Categorical dark slots 3 (aqua) and 2 (orange) of the dataviz reference
// palette — blue is skipped because it already means magnitude on this page.
// Validated on the chart surface #181f37 (11 Oct 2026): CVD ΔE 9.4, normal
// ΔE 26.5, both >= 3:1. "Music, no speech" is texture (hatching), not a hue.
const LANE = { speech: "#199e70", music: "#d95926", alone: "#5d6788" };

function laneSvg(duration, seconds, color, label, hatch) {
  const W = 1000, H = 18;
  const rects = seconds.map((t) => `<rect x="${(t / duration) * W}" y="0" width="${Math.max(1.2, W / duration)}" height="${H}"
      fill="${hatch ? "url(#hatch)" : color}"/>`).join("");
  return `<div class="lane"><div class="lane-label">${label}</div>
    <svg class="lane-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="${label}">
      <defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <rect width="6" height="6" fill="#1f2744"/><rect width="2.4" height="6" fill="${LANE.alone}"/></pattern></defs>
      <rect width="${W}" height="${H}" fill="#181f37"/>${rects}</svg></div>`;
}

function soundTimeline(data, c) {
  const all = (k) => c.windows.flatMap((w) => w[k] || []);
  const spoken = all("spoken_at"), under = all("music_under_speech_at"), alone = all("music_without_speech_at");
  const heard = {};
  c.windows.forEach((w) => (w.heard || []).forEach((h) => { heard[h] = (heard[h] || 0) + 1; }));
  const spokenS = c.windows.reduce((a, w) => a + w.narrated_s, 0);
  const underS = c.windows.reduce((a, w) => a + w.music_under_speech_s, 0);
  const m = c.methodology;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => `<span>${timecode(f * data.duration)}</span>`).join("");
  return `<div class="card">
    <h2><span class="card-icon">${icon("music")}</span>Background music</h2>
    <p class="card-sub">Each second of the soundtrack, as the sound classifier heard it. Only music <strong style="color:var(--ink)">under speech</strong>
      counts — music with nobody speaking (an intro jingle) does not compete with narration and is shown hatched.</p>
    <div class="lanes">
      ${laneSvg(data.duration, spoken, LANE.speech, "Speech")}
      ${laneSvg(data.duration, under, LANE.music, "Music under speech")}
      ${laneSvg(data.duration, alone, LANE.alone, "Music, no speech", true)}
      <div class="lane"><div class="lane-label"></div><div class="lane-axis tabular">${ticks}</div></div>
    </div>
    <div class="legend">
      <span><i style="background:${LANE.speech}"></i>Speech</span>
      <span><i style="background:${LANE.music}"></i>Music under speech — counted</span>
      <span><i class="hatched"></i>Music, no speech — not counted</span>
    </div>
    <dl class="stat-row">
      <div class="stat"><dt>Spoken time</dt><dd class="tabular">${Math.round(spokenS)} s</dd></div>
      <div class="stat"><dt>With music under it</dt><dd class="tabular">${Math.round(underS)} s (${spokenS ? pct(underS / spokenS) : "—"})</dd></div>
      <div class="stat"><dt>Music threshold</dt><dd class="tabular">${escapeHtml(m.music_classes.join("/"))} ≥ ${m.music_threshold}</dd></div>
    </dl>
    ${Object.keys(heard).length ? `<p class="muted" style="margin:10px 0 0">The classifier heard: ${escapeHtml(Object.entries(heard).sort((a, b) => b[1] - a[1]).slice(0, 6).map((e) => e[0]).join(", "))}.</p>` : ""}
  </div>`;
}

function offTopicCard(c) {
  const lv = c.off_topic_levels || {};
  const judged = c.windows.filter((w) => w.off_topic_check && w.off_topic_check.judged);
  if (lv.cut == null || !judged.length) {
    const why = (c.warnings || []).find((x) => x.startsWith("off-topic part not assessed"));
    return `<div class="card"><h2><span class="card-icon">${icon("target")}</span>Off-topic narration</h2>
      <p class="card-sub" style="margin:0">${escapeHtml(why || "Not assessed for this video.")}</p></div>`;
  }
  const vals = judged.map((w) => w.off_topic_check.relevance).concat([lv.unrelated, lv.typical, lv.cut]);
  const lo = Math.min(...vals) - 0.05, hi = Math.max(...vals) + 0.05;
  const x = (v) => ((v - lo) / (hi - lo)) * 100;
  const dots = judged.map((w) => {
    const off = w.off_topic_check.off_topic;
    return `<span class="dot ${off ? "dot-off" : ""}" style="left:${x(w.off_topic_check.relevance)}%"
      title="${escapeHtml(w.timecode)} · relevance ${w.off_topic_check.relevance.toFixed(3)}${off ? " · OFF-TOPIC" : ""}">${off ? "×" : ""}</span>`;
  }).join("");
  const marker = (v, label, cls) => `<span class="marker ${cls}" style="left:${x(v)}%"><span>${label}<br/><b class="tabular">${v.toFixed(3)}</b></span></span>`;
  const off = judged.filter((w) => w.off_topic_check.off_topic);
  const expNote = Number(c.methodology.weights.off_topic) === 0
    ? `<div class="banner banner-warning" style="margin-bottom:14px">${icon("info", 20)}<div><strong>Experimental — not part of the score.</strong>
        Read by hand on this corpus, most windows this test flags are short remarks about the board or on-topic talk in everyday
        words, not real digressions. It is shown so you can judge it; the professor's scores will decide whether it earns a weight.</div></div>` : "";
  return `<div class="card"><h2><span class="card-icon">${icon("target")}</span>Off-topic narration</h2>
    ${expNote}
    <p class="card-sub">Each dot is one 10-second window, placed by how close its narration is (in meaning) to the rest of the video.
      A window left of the <strong style="color:var(--ink)">cut-off</strong> — nearer to unrelated everyday text than to this video's
      typical narration — is off-topic (×). Hover a dot for its time.</p>
    <div class="relevance-strip">${marker(lv.unrelated, "unrelated text", "m-unrel")}${marker(lv.cut, "cut-off", "m-cut")}${marker(lv.typical, "typical", "m-typ")}
      <div class="strip-line"></div>${dots}</div>
    ${off.length ? `<div class="table-scroll" style="margin-top:14px"><table class="plain"><thead><tr><th>Window</th><th class="num">Relevance</th><th>Narration</th></tr></thead><tbody>
      ${off.map((w) => `<tr><td class="tabular">${escapeHtml(w.timecode)} <span class="badge badge-scored">${icon("warn", 12)} off-topic</span></td>
        <td class="num tabular">${w.off_topic_check.relevance.toFixed(3)}</td><td>${escapeHtml(w.narration)}</td></tr>`).join("")}
      </tbody></table></div>` : `<p class="muted" style="margin:14px 0 0">${icon("check", 14)} No window's narration was judged off-topic.</p>`}
  </div>`;
}

function coherenceMethod(c) {
  const m = c.methodology;
  const rows = [
    ["Background-music weight", m.weights.background_music], ["Off-topic weight", m.weights.off_topic],
    ["Sound classifier", m.audio_model], ["Counts as music", `${m.music_classes.join("/")} probability ≥ ${m.music_threshold}, under speech`],
    ["Embedding model", m.embedding_model], ["Off-topic cut-off", `unrelated + ${m.decision_point} × (typical − unrelated)`],
    ["Minimum words to judge topic", m.min_narration_words], ["Unrelated reference", `${m.reference_texts} (${m.n_reference_texts} paragraphs)`],
  ].map(([k, v]) => `<tr><td>${escapeHtml(k)}</td><td class="num tabular">${escapeHtml(String(v))}</td></tr>`).join("");
  return `<div class="card"><details class="methodology">
    <summary><span class="card-icon">${icon("doc")}</span>Coherence methodology ${icon("chevron", 16)}</summary>
    <p class="muted">Limitations: the off-topic test is relative to the video itself, so a video that is off-topic throughout
      cannot be detected, and a legitimate change of subtopic can look like a digression. Weights are untuned until the
      professor's coherence scores exist.</p>
    <table class="plain"><tbody>${rows}</tbody></table></details></div>`;
}

function coherenceSegmentDetail(seg, c) {
  const m = c.methodology, wm = m.weights.background_music, wo = m.weights.off_topic;
  const rows = seg.windows.map((w) => `<tr class="${w.qualifies ? "" : "row-muted"}">
      <td class="tabular">${escapeHtml(w.timecode)}</td>
      <td class="num tabular">${w.qualifies ? pct(w.background_music) : "—"}</td>
      <td>${w.qualifies ? (w.off_topic ? "off-topic" : (w.off_topic_check.judged ? "on topic" : "not judged")) : "no narration (counts 0)"}</td>
      <td class="num tabular">${w.qualifies ? fmtScore(w.score) : "0"}</td>
      <td class="num tabular">${f2(w.duration)} s</td>
      <td class="num tabular">${f2(w.qualifies ? w.score * w.duration : 0)}</td></tr>`).join("");
  const num = seg.windows.reduce((a, w) => a + (w.qualifies ? w.score * w.duration : 0), 0);
  const den = seg.windows.reduce((a, w) => a + w.duration, 0);
  const formula = seg.windows.length ? `(${seg.windows.map((w) => `${w.qualifies ? f2(w.score) : "0"} × ${f2(w.duration)}`).join(" + ")}) ÷ (${seg.windows.map((w) => f2(w.duration)).join(" + ")})
      <br/>= ${f2(num)} ÷ ${f2(den)} = <strong>${fmtScore(seg.score)}</strong>` : "No measurement window falls in this segment.";
  const firstScored = seg.windows.findIndex((w) => w.qualifies && w.score > 0);
  return `<div class="drawer-section-title"><span class="card-icon">${icon("sigma", 16)}</span>How this segment's coherence score is calculated</div>
    <div class="note">The segment score is the duration-weighted average of all its 10-second windows; a window with no narration counts as 0.</div>
    <div class="table-scroll"><table class="plain"><thead><tr><th>Window</th><th class="num">Music under speech</th><th>Topic</th>
      <th class="num">Score</th><th class="num">Duration</th><th class="num">Score × duration</th></tr></thead><tbody>${rows}</tbody></table></div>
    <div class="formula tabular">Segment score = ${formula}</div>
    <div class="drawer-section-title" style="margin-top:22px"><span class="card-icon">${icon("layers", 16)}</span>
      Window by window <span class="muted" style="font-weight:400">— click a window to open its calculation</span></div>
    ${seg.windows.map((w, i) => coherenceWindow(w, m, wm, wo, i === (firstScored === -1 ? 0 : firstScored))).join("")}`;
}

function coherenceWindow(w, m, wm, wo, open) {
  const ot = w.off_topic_check || {};
  const status = !w.qualifies ? ["badge-na", "No narration"] : w.score > 0 ? ["badge-scored", [w.background_music > 0 ? "music" : "", w.off_topic ? "off-topic" : ""].filter(Boolean).join(" + ")] : ["badge-zero", "Nothing extraneous"];
  const narration = `<div class="evidence-col voice" style="margin-top:12px"><div class="evidence-col-title">${icon("mic", 14)} Narration in this window</div>${escapeHtml(w.narration) || "<em>(no narration)</em>"}</div>`;
  let body;
  if (!w.qualifies) {
    body = `${narration}<div class="verdict">${icon("info", 18)}<div><strong>Nobody speaks in this window.</strong>
      Neither part can be judged without narration, so it counts as 0 in the segment average.
      ${w.music_without_speech_s ? `Music without speech here: ${Math.round(w.music_without_speech_s)} s — recorded, not counted.` : ""}</div></div>`;
  } else {
    const step1 = `<div class="step-block"><div class="step-title"><span class="step-num">1</span>How much of the speech has music under it?</div>
      <div class="note">A second counts when someone is speaking and the classifier gives ${escapeHtml(m.music_classes.join("/"))} a probability of at least ${m.music_threshold}.
        ${w.heard.length ? `Heard: ${escapeHtml(w.heard.join(", "))}.` : ""}</div>
      <div class="formula tabular">music share = ${f2(w.music_under_speech_s)} s with music ÷ ${f2(w.narrated_s)} s spoken = <strong>${f4(w.background_music)}</strong></div>
      ${w.music_without_speech_s ? `<div class="note">Plus ${f2(w.music_without_speech_s)} s of music with nobody speaking — not counted.</div>` : ""}</div>`;
    const step2 = ot.judged
      ? `<div class="step-block"><div class="step-title"><span class="step-num">2</span>Does the narration stay on topic?</div>
        <div class="note">Relevance = cosine similarity between this window's narration and the rest of the video's narration
          (${escapeHtml(m.embedding_model)}). Cut-off = unrelated + ${m.decision_point} × (typical − unrelated).</div>
        <div class="formula tabular">cut-off = ${f4(ot.unrelated)} + ${m.decision_point} × (${f4(ot.typical)} − ${f4(ot.unrelated)}) = ${f4(ot.cut)}<br/>
          relevance ${f4(ot.relevance)} ${ot.off_topic ? "&lt;" : "≥"} ${f4(ot.cut)} → <strong>${ot.off_topic ? "off-topic (1)" : "on topic (0)"}</strong></div></div>`
      : `<div class="step-block"><div class="step-title"><span class="step-num">2</span>Does the narration stay on topic?</div>
        <div class="note">Not judged: ${escapeHtml(ot.reason_not_judged || "")}. Counts as on topic (0).</div></div>`;
    const step3 = `<div class="step-block"><div class="step-title"><span class="step-num">3</span>Window score</div>
      <div class="formula tabular">100 × (${wm} × music share + ${wo} × off-topic)<br/>
        = 100 × (${wm} × ${f4(w.background_music)} + ${wo} × ${w.off_topic ? 1 : 0})<br/>
        = <strong>${fmtScore(w.score)}</strong></div></div>`;
    body = narration + step1 + step2 + step3;
  }
  return `<details class="win" ${open ? "open" : ""}>
    <summary><span class="win-chev">${icon("chevron", 16)}</span><span class="win-time tabular">${escapeHtml(w.timecode)}</span>
      <span class="badge ${status[0]}">${escapeHtml(status[1])}</span>
      <span class="win-score tabular">${w.qualifies ? fmtScore(w.score) : "—"}</span></summary>
    <div class="win-body">${body}</div></details>`;
}

// ------------------------------------------------------------------ boot

(async function init() {
  renderWelcome();
  await loadStages();
  await loadVideoList();
})();
