"use strict";

const $ = (id) => document.getElementById(id);
const state = { images: [], image: null, features: [], presets: [], variants: [], tool: "brush",
                history: [], drawing: false, start: null, snapshot: null, pollTimer: null, modalVariant: null };

// ---------------------------------------------------------------- helpers
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

let toastTimer;
function toast(msg, isError = false) {
  const t = $("toast");
  t.textContent = msg;
  t.className = "toast" + (isError ? " error" : "");
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), isError ? 7000 : 3500);
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail
        : Array.isArray(body.detail) ? body.detail.map((d) => `${(d.loc || []).join(".")}: ${d.msg}`).join("; ") : detail;
    } catch (_) { /* non-JSON error */ }
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}

const jsonPost = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

function fmt(v, digits = 3) { return v === null || v === undefined ? "–" : Number(v).toFixed(digits); }

// ---------------------------------------------------------------- init
async function init() {
  bindRanges();
  bindUpload();
  bindCanvas();
  bindActions();
  try {
    const f = await api("/api/features");
    state.features = f.features;
    state.presets = f.region_presets;
    applyDefaults(f.defaults);
    fillFeatureSelect();
    fillPresetSelect();
  } catch (e) { toast("Could not load feature catalogue: " + e.message, true); }
  loadHealth();
  await loadImages();
}

async function loadHealth() {
  const chip = $("engineChip");
  try {
    const h = await api("/api/health");
    const e = h.engine;
    if (e.dependencies_installed) { chip.textContent = `Engine ready · ${e.device}`; chip.className = "chip ok"; }
    else { chip.textContent = "Install torch + diffusers to generate"; chip.className = "chip warn"; }
  } catch (_) { chip.textContent = "API offline"; chip.className = "chip warn"; }
}

function applyDefaults(d) {
  const set = (id, v) => { $(id).value = v; $(id).dispatchEvent(new Event("input")); };
  set("steps", d.steps); set("guidance", d.guidance_scale); set("strength", d.strength);
  set("cnScale", d.controlnet_scale); set("dilate", d.mask_dilate_px); set("feather", d.mask_feather_px);
}

function bindRanges() {
  document.querySelectorAll('input[type="range"]').forEach((input) => {
    const out = input.parentElement.querySelector("output");
    const update = () => { if (out) out.textContent = input.value; };
    input.addEventListener("input", update);
    update();
  });
  $("modeSelect").addEventListener("change", () => {
    const cn = $("modeSelect").value === "controlnet";
    document.querySelectorAll(".cn-only").forEach((el) => (el.style.opacity = cn ? 1 : 0.4));
  });
  $("modeSelect").dispatchEvent(new Event("change"));
}

// ---------------------------------------------------------------- features
function fillFeatureSelect() {
  $("featureSelect").innerHTML = state.features.map((f) => `<option value="${esc(f.id)}">${esc(f.label)}</option>`).join("");
  $("featureSelect").onchange = onFeatureChange;
  onFeatureChange();
}

function onFeatureChange() {
  const f = state.features.find((x) => x.id === $("featureSelect").value);
  if (!f) return;
  const custom = f.id === "custom";
  $("optionSelect").innerHTML = custom ? `<option value="">(describe below)</option>`
    : f.options.map((o) => `<option value="${esc(o)}">${esc(o)}</option>`).join("");
  $("optionSelect").disabled = custom;
  $("customPrompt").placeholder = custom ? "Describe the feature, e.g. matte black front splitter" : "e.g. 18-inch, gloss finish";
}

function fillPresetSelect() {
  $("presetSelect").innerHTML = `<option value="">Choose…</option>` +
    state.presets.map((p) => `<option value="${esc(p.id)}">${esc(p.label)}</option>`).join("");
  $("presetSelect").onchange = async () => {
    const p = state.presets.find((x) => x.id === $("presetSelect").value);
    $("presetSelect").value = "";
    if (!p || !requireImage()) return;
    if (p.region.type === "auto_wheels") return detectWheels();
    const [x0, y0, x1, y1] = p.region.box;
    const c = $("maskCanvas");
    pushHistory();
    const ctx = c.getContext("2d");
    ctx.globalCompositeOperation = "source-over";
    ctx.fillStyle = MASK_COLOR;
    ctx.fillRect(x0 * c.width, y0 * c.height, (x1 - x0) * c.width, (y1 - y0) * c.height);
  };
}

// ---------------------------------------------------------------- images
async function loadImages(selectId) {
  state.images = await api("/api/images");
  const sel = $("imageSelect");
  sel.innerHTML = state.images.length
    ? state.images.map((i) => `<option value="${esc(i.id)}">${esc(i.filename)} (${i.width}×${i.height})</option>`).join("")
    : `<option value="">No images yet</option>`;
  sel.onchange = () => selectImage(sel.value);
  const target = selectId || (state.image && state.images.find((i) => i.id === state.image.id) ? state.image.id : state.images[0]?.id);
  if (target) { sel.value = target; await selectImage(target); } else { selectImage(null); }
}

async function selectImage(id) {
  state.image = state.images.find((i) => i.id === id) || null;
  const img = $("baseImg"), canvas = $("maskCanvas");
  if (!state.image) {
    img.hidden = canvas.hidden = true; $("emptyState").hidden = false; $("imageMeta").textContent = "";
    state.variants = []; renderGallery(); return;
  }
  $("emptyState").hidden = true;
  await new Promise((resolve) => { img.onload = resolve; img.onerror = resolve; img.src = state.image.url; });
  img.hidden = canvas.hidden = false;
  canvas.width = state.image.width; canvas.height = state.image.height;
  state.history = [];
  $("viewSelect").value = state.image.view;
  $("imageMeta").textContent = `${state.image.filename} · ${state.image.width}×${state.image.height} (original ${state.image.orig_width}×${state.image.orig_height})`;
  await loadVariants();
}

function bindUpload() {
  const dz = $("dropzone"), input = $("fileInput");
  dz.onclick = () => input.click();
  dz.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } };
  input.onchange = () => input.files[0] && upload(input.files[0]);
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => e.dataTransfer.files[0] && upload(e.dataTransfer.files[0]));
}

async function upload(file) {
  const form = new FormData();
  form.append("file", file);
  form.append("view", $("viewSelect").value);
  try {
    const img = await api("/api/images", { method: "POST", body: form });
    toast("Image uploaded");
    await loadImages(img.id);
  } catch (e) { toast(e.message, true); }
  $("fileInput").value = "";
}

// ---------------------------------------------------------------- mask canvas
const MASK_COLOR = "#ff3c5a"; // drawn opaque; the canvas element is shown at 60% opacity via CSS

function requireImage() {
  if (!state.image) { toast("Select or upload an image first", true); return false; }
  return true;
}

function pushHistory() {
  const c = $("maskCanvas");
  state.history.push(c.getContext("2d").getImageData(0, 0, c.width, c.height));
  if (state.history.length > 25) state.history.shift();
}

function canvasPos(e) {
  const c = $("maskCanvas"), r = c.getBoundingClientRect();
  return { x: ((e.clientX - r.left) * c.width) / r.width, y: ((e.clientY - r.top) * c.height) / r.height };
}

function bindCanvas() {
  const c = $("maskCanvas"), ctx = c.getContext("2d");
  const brush = () => Number($("brushSize").value) * (c.width / 768);

  c.addEventListener("pointerdown", (e) => {
    if (!state.image) return;
    c.setPointerCapture(e.pointerId);
    pushHistory();
    state.drawing = true;
    state.start = canvasPos(e);
    state.snapshot = ctx.getImageData(0, 0, c.width, c.height);
    if (state.tool !== "rect") dot(state.start, brush());
  });
  c.addEventListener("pointermove", (e) => {
    if (!state.drawing) return;
    const p = canvasPos(e);
    if (state.tool === "rect") {
      ctx.putImageData(state.snapshot, 0, 0);
      ctx.globalCompositeOperation = "source-over";
      ctx.fillStyle = MASK_COLOR;
      ctx.fillRect(state.start.x, state.start.y, p.x - state.start.x, p.y - state.start.y);
    } else {
      stroke(state.start, p, brush());
      state.start = p;
    }
  });
  const end = () => { state.drawing = false; };
  c.addEventListener("pointerup", end);
  c.addEventListener("pointercancel", end);

  function setMode() { ctx.globalCompositeOperation = state.tool === "erase" ? "destination-out" : "source-over"; }
  function dot(p, size) {
    setMode(); ctx.fillStyle = "#ff3c5a"; ctx.beginPath(); ctx.arc(p.x, p.y, size / 2, 0, Math.PI * 2); ctx.fill();
  }
  function stroke(a, b, size) {
    setMode(); ctx.strokeStyle = "#ff3c5a"; ctx.lineWidth = size; ctx.lineCap = "round"; ctx.lineJoin = "round";
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
  }

  document.querySelectorAll(".tool").forEach((btn) => btn.addEventListener("click", () => {
    state.tool = btn.dataset.tool;
    document.querySelectorAll(".tool").forEach((b) => b.classList.toggle("active", b === btn));
  }));
  $("brushSize").addEventListener("input", () => ($("brushOut").textContent = $("brushSize").value));
  $("undoBtn").onclick = () => { const h = state.history.pop(); if (h) c.getContext("2d").putImageData(h, 0, 0); };
  $("clearBtn").onclick = () => { if (state.image) { pushHistory(); ctx.clearRect(0, 0, c.width, c.height); } };
  $("wheelsBtn").onclick = detectWheels;
}

async function detectWheels() {
  if (!requireImage()) return;
  try {
    const res = await api(`/api/images/${state.image.id}/detect-wheels`, { method: "POST" });
    const c = $("maskCanvas"), ctx = c.getContext("2d");
    pushHistory();
    ctx.globalCompositeOperation = "source-over";
    ctx.fillStyle = MASK_COLOR;
    res.circles.forEach((w) => {
      ctx.beginPath(); ctx.arc(w.cx * c.width, w.cy * c.height, w.r * c.width * res.scale, 0, Math.PI * 2); ctx.fill();
    });
    toast(`Detected ${res.count} wheel${res.count === 1 ? "" : "s"} - adjust with brush/eraser if needed`);
  } catch (e) { toast(e.message, true); }
}

/** White-on-black PNG of the painted region, as a data URL. Returns null when nothing is painted. */
function exportMask() {
  const c = $("maskCanvas"), src = c.getContext("2d").getImageData(0, 0, c.width, c.height);
  const out = document.createElement("canvas");
  out.width = c.width; out.height = c.height;
  const octx = out.getContext("2d"), dst = octx.createImageData(c.width, c.height);
  let painted = 0;
  for (let i = 0; i < src.data.length; i += 4) {
    const on = src.data[i + 3] > 16;
    const v = on ? 255 : 0;
    painted += on ? 1 : 0;
    dst.data[i] = dst.data[i + 1] = dst.data[i + 2] = v; dst.data[i + 3] = 255;
  }
  if (!painted) return null;
  octx.putImageData(dst, 0, 0);
  return out.toDataURL("image/png");
}

// ---------------------------------------------------------------- generate
function buildRequest() {
  if (!requireImage()) return null;
  const mask = exportMask();
  if (!mask) { toast("Paint the region you want to change first", true); return null; }
  const feature = $("featureSelect").value;
  const custom = $("customPrompt").value.trim();
  if (feature === "custom" && !custom) { toast("Describe the custom feature", true); return null; }
  const seed = $("seed").value;
  return {
    image_id: state.image.id, feature,
    option: feature === "custom" ? null : $("optionSelect").value,
    custom_prompt: custom || null,
    mode: $("modeSelect").value,
    regions: [{ type: "mask_png", data: mask }],
    steps: Number($("steps").value), guidance_scale: Number($("guidance").value),
    strength: Number($("strength").value), controlnet_scale: Number($("cnScale").value),
    keep_inner_edges: $("keepInner").checked,
    mask_dilate_px: Number($("dilate").value), mask_feather_px: Number($("feather").value),
    seed: seed === "" ? null : Number(seed),
  };
}

function bindActions() {
  $("diceBtn").onclick = () => ($("seed").value = Math.floor(Math.random() * 2147483647));
  $("generateBtn").onclick = () => submit("/api/variants", "Variant queued");
  $("compareBtn").onclick = () => submit("/api/compare", "Both pipelines queued (same seed & mask)");
  $("refreshBtn").onclick = loadVariants;
  $("deleteImageBtn").onclick = async () => {
    if (!requireImage() || !confirm("Delete this image and all its variants?")) return;
    try { await api(`/api/images/${state.image.id}`, { method: "DELETE" }); state.image = null; await loadImages(); }
    catch (e) { toast(e.message, true); }
  };
  $("modalClose").onclick = closeModal;
  $("modal").addEventListener("click", (e) => { if (e.target === $("modal")) closeModal(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });
  $("cmpSlider").addEventListener("input", () => {
    $("cmpAfter").style.clipPath = `inset(0 0 0 ${$("cmpSlider").value}%)`;
  });
}

async function submit(url, okMessage) {
  const body = buildRequest();
  if (!body) return;
  const btns = [$("generateBtn"), $("compareBtn")];
  btns.forEach((b) => (b.disabled = true));
  try {
    await jsonPost(url, body);
    toast(okMessage);
    await loadVariants();
  } catch (e) { toast(e.message, true); }
  finally { btns.forEach((b) => (b.disabled = false)); }
}

// ---------------------------------------------------------------- gallery
async function loadVariants() {
  if (!state.image) return;
  try { state.variants = await api(`/api/variants?image_id=${encodeURIComponent(state.image.id)}`); }
  catch (e) { toast(e.message, true); return; }
  renderGallery();
  const active = state.variants.some((v) => v.status === "queued" || v.status === "running");
  clearTimeout(state.pollTimer);
  if (active) state.pollTimer = setTimeout(loadVariants, 2500);
  if (state.modalVariant) { const v = state.variants.find((x) => x.id === state.modalVariant); if (v) renderModalInfo(v); }
}

function qcBadge(v) {
  if (v.status === "queued") return `<span class="badge">queued</span>`;
  if (v.status === "running") return `<span class="badge warn">generating…</span>`;
  if (v.status === "failed") return `<span class="badge bad">failed</span>`;
  return v.qc.passed ? `<span class="badge ok">QC pass</span>` : `<span class="badge bad">QC: ${v.qc.failed.length} issue${v.qc.failed.length === 1 ? "" : "s"}</span>`;
}

function reviewBadge(v) {
  return v.review_decision === "approved" ? `<span class="badge ok">approved</span>`
    : v.review_decision === "rejected" ? `<span class="badge bad">rejected</span>` : "";
}

function renderGallery() {
  $("variantCount").textContent = state.variants.length ? `(${state.variants.length})` : "";
  const g = $("gallery");
  if (!state.variants.length) { g.innerHTML = `<p class="hint">No variants yet. Paint a region and press Generate.</p>`; return; }
  g.innerHTML = state.variants.map((v) => {
    const thumb = v.status === "done" ? `<img loading="lazy" src="${v.files.output}" alt="Generated variant">`
      : v.status === "failed" ? `<div class="error-text">${esc(v.error)}</div>` : `<div class="spinner" role="status" aria-label="Generating"></div>`;
    const m = v.metrics || {};
    return `<article class="card" data-id="${esc(v.id)}">
      <div class="thumb" data-open="${esc(v.id)}">${thumb}</div>
      <div class="body">
        <div class="title">${esc(v.feature)}${v.option ? " · " + esc(v.option) : ""}</div>
        <div class="badges"><span class="badge">${v.mode === "controlnet" ? "ControlNet" : "Inpaint"}</span>${qcBadge(v)}${reviewBadge(v)}</div>
        <div class="meta">seed ${v.seed}${v.status === "done" ? ` · SSIM ${fmt(m.raw_outside_ssim, 2)} · edge F1 ${fmt(m.raw_edge_f1, 2)}` : ""}${v.duration_s ? ` · ${v.duration_s}s` : ""}</div>
        <div class="btn-row">
          ${v.status === "done" ? `<button class="btn small" data-open="${esc(v.id)}" type="button">Inspect</button>` : ""}
          <button class="btn small danger ghost" data-del="${esc(v.id)}" type="button">Delete</button>
        </div>
      </div></article>`;
  }).join("");
  g.querySelectorAll("[data-open]").forEach((el) => el.addEventListener("click", () => openModal(el.dataset.open)));
  g.querySelectorAll("[data-del]").forEach((el) => el.addEventListener("click", async () => {
    try { await api(`/api/variants/${el.dataset.del}`, { method: "DELETE" }); await loadVariants(); } catch (e) { toast(e.message, true); }
  }));
}

// ---------------------------------------------------------------- modal
const METRIC_ROWS = [
  ["coverage", "Edited area"], ["raw_outside_ssim", "Background SSIM (raw)"], ["raw_outside_mae", "Background MAE (raw)"],
  ["raw_edge_f1", "Structure edge F1 (raw)"], ["final_background_ssim", "Background SSIM (final)"],
  ["inside_change", "Feature change"], ["halo_ssim", "Halo SSIM"], ["clip_score", "CLIP similarity"],
];

function openModal(id) {
  const v = state.variants.find((x) => x.id === id);
  if (!v || v.status !== "done") return;
  state.modalVariant = id;
  $("modal").hidden = false;
  $("cmpBefore").src = state.image.url;
  const tabs = [["output", "Final"], ["raw", "Raw model output"], ["mask", "Mask"], ["control", "ControlNet edges"]]
    .filter(([k]) => v.files[k]);
  $("viewTabs").innerHTML = tabs.map(([k, l], i) => `<button type="button" data-kind="${k}" class="${i === 0 ? "active" : ""}">${l}</button>`).join("");
  $("viewTabs").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    $("viewTabs").querySelectorAll("button").forEach((x) => x.classList.toggle("active", x === b));
    $("cmpAfter").src = v.files[b.dataset.kind];
  }));
  $("cmpAfter").src = v.files.output;
  $("cmpSlider").value = 50; $("cmpAfter").style.clipPath = "inset(0 0 0 50%)";
  renderModalInfo(v);
}

function closeModal() { $("modal").hidden = true; state.modalVariant = null; }

function renderModalInfo(v) {
  const m = v.metrics || {};
  const rows = METRIC_ROWS.map(([k, l]) => `<tr><td>${l}</td><td>${fmt(m[k])}</td></tr>`).join("");
  const checks = (v.qc?.checks || []).map((c) => `<tr><td class="${c.passed ? "pass" : "fail"}">${c.passed ? "✔" : "✘"} ${esc(c.label)}${c.passed ? "" : `<br><small>${esc(c.help)}</small>`}</td>
    <td>${fmt(c.value)} ${c.op} ${c.threshold}</td></tr>`).join("");
  $("modalInfo").innerHTML = `
    <h3>${esc(v.feature)}${v.option ? " · " + esc(v.option) : ""}</h3>
    <div class="badges">${qcBadge(v)}${reviewBadge(v)}<span class="badge">${v.mode}</span><span class="badge">seed ${v.seed}</span></div>
    <p class="prompt">${esc(v.prompt)}</p>
    <h3>Quality-control checkpoints</h3><table>${checks}</table>
    <h3>Metrics</h3><table>${rows}</table>
    <label>Reviewer note<input id="reviewNote" type="text" maxlength="1000" value="${esc(v.review_note || "")}"></label>
    <div class="btn-row" style="margin-top:10px">
      <button class="btn primary" data-review="approved" type="button">Approve</button>
      <button class="btn danger" data-review="rejected" type="button">Reject</button>
      <a class="btn ghost" href="${v.files.output}" download>Download</a>
    </div>`;
  $("modalInfo").querySelectorAll("[data-review]").forEach((b) => b.addEventListener("click", async () => {
    try {
      await jsonPost(`/api/variants/${v.id}/review`, { decision: b.dataset.review, note: $("reviewNote").value || null });
      toast(`Marked ${b.dataset.review}`);
      await loadVariants();
    } catch (e) { toast(e.message, true); }
  }));
}

init();
