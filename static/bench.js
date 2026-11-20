"use strict";

const mount = document.getElementById("content");
const modal = document.getElementById("modal");
const notice = document.getElementById("notice");
const recent = document.getElementById("recent");
const roles = {
  transmitted: "Transmitted light", reflected: "Reflected front", rake_left: "Raking from left",
  rake_right: "Raking from right", incident: "Incident reference", dark: "Dark reference"
};
const categories = {
  pane: "Glass piece", came: "Came or solder", fracture: "Observed fracture", paint: "Paint detail",
  occluded: "Occluded region", ignore: "Exclude from measurement", reference: "Reference patch"
};
const states = { observed: "Observed", review: "Needs review", resolved: "Resolved" };
const tabs = {
  capture: "Photographs", align: "Register", mark: "Markings", measure: "Examinations",
  compare: "Compare", evidence: "Evidence", visits: "Visit log"
};
const model = {
  list: [], item: null, capture: null, tab: "capture", tool: "exposure", display: null, marks: true,
  plane: true, active: null, pointMode: null, points: [], blobUrls: [], editingMark: null, opened: null,
  tools: {}, selectedPair: "", probe: null, epoch: 0
};
const today = () => new Date().toISOString().slice(0, 10);
const title = text => String(text || "").replaceAll("_", " ").replace(/^./, c => c.toUpperCase());
const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, c => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
})[c]);
const attr = escapeHtml;
const idOf = node => node.closest("[data-id]")?.dataset.id;
const value = (form, key) => form.elements.namedItem(key)?.value ?? "";
const number = (form, key) => Number(value(form, key));
const checked = (form, key) => !!form.elements.namedItem(key)?.checked;
const currentWindow = () => model.item?.window;
const photo = () => model.item?.captures.find(item => item.id === model.capture);
const imageSource = (key, tag) => "/api/records/" + key + "/media/" + tag;
const routeTo = key => { location.hash = key ? "#window/" + key : "#collection"; };
const selected = yes => yes ? " selected" : "";
const current = yes => yes ? ' aria-current="true"' : "";
const textFormat = v => typeof v === "number" ? v.toLocaleString(undefined, { maximumFractionDigits: 5 })
  : typeof v === "boolean" ? v ? "Yes" : "No" : v === null ? "Unavailable" : String(v);
const scalar = value => value === null || ["number", "string", "boolean"].includes(typeof value);
const optionList = (labels, mark) => Object.entries(labels)
  .map(([key, label]) => '<option value="' + attr(key) + '"' + selected(mark === key) + ">" + escapeHtml(label) + "</option>").join("");
const button = (label, action, style = "", id = "") =>
  '<button type="button" class="btn ' + style + '" data-action="' + action + '"' + (id ? ' data-id="' + attr(id) + '"' : "") + ">" + escapeHtml(label) + "</button>";
const link = (label, href, style = "") =>
  '<a class="btn ' + style + '" href="' + attr(href) + '">' + escapeHtml(label) + "</a>";
const submit = label => '<button class="btn primary" type="submit">' + escapeHtml(label) + "</button>";
const input = (label, name, type = "text", contents = "", extras = "") =>
  '<label>' + escapeHtml(label) + '<input type="' + type + '" name="' + name + '" value="' + attr(contents) + '" ' + extras + "></label>";
const selector = (label, name, options, mark = "") =>
  '<label>' + escapeHtml(label) + '<select name="' + name + '">' + optionList(options, mark) + "</select></label>";
const textarea = (label, name, contents = "", extras = "") =>
  '<label>' + escapeHtml(label) + '<textarea name="' + name + '" ' + extras + ">" + escapeHtml(contents) + "</textarea></label>";
const checkbox = (label, name, enabled = false, contents = "") =>
  '<label class="check"><input type="checkbox" name="' + name + '" value="' + attr(contents) + '"' + (enabled ? " checked" : "") + ">" + escapeHtml(label) + "</label>";
const controls = (...items) => '<div class="actions">' + items.join("") + "</div>";
const row = (...items) => '<div class="row">' + items.join("") + "</div>";
const empty = message => '<p class="empty">' + escapeHtml(message) + "</p>";
const form = (action, content) => '<form data-form="' + action + '">' + content + "</form>";

async function request(path, method = "GET", body = null, binary = false) {
  const options = { method, headers: { Accept: binary ? "image/png" : "application/json" } };
  if (body instanceof FormData) {
    options.body = body;
  } else if (body !== null) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  const response = await fetch("/api" + path, options);
  if (!response.ok) {
    const info = await response.json().catch(() => ({}));
    throw new Error(info.error || "The library could not complete this request.");
  }
  return binary ? response.blob() : response.json();
}

function flash(msg, error = false) {
  clearTimeout(flash.timer);
  notice.textContent = msg;
  notice.classList.toggle("error", error);
  notice.hidden = false;
  flash.timer = setTimeout(() => { notice.hidden = true; }, error ? 7200 : 3600);
}

function openModal(label, content, wide = false) {
  modal.classList.toggle("wide", wide);
  document.getElementById("modal-content").innerHTML = '<div class="section-head"><h2 id="modal-title">' +
    escapeHtml(label) + "</h2>" + button("Close", "close") + "</div>" + content;
  if (!modal.open) modal.showModal();
}

function showError(error, content) {
  if (content) content.innerHTML = empty(error.message);
  flash(error.message, true);
}

function clearPhoto() {
  model.active = null;
  model.probe = null;
  model.points = [];
  for (const url of model.blobUrls) URL.revokeObjectURL(url);
  model.blobUrls = [];
}

function usePicture(blob) {
  const url = URL.createObjectURL(blob);
  model.blobUrls.push(url);
  model.active = url;
  syncBoard();
}

function pageHeading(name, description, action = "", crumb = "") {
  return '<header class="page-head"><div>' + (crumb ? '<p class="crumb">' + crumb + "</p>" : "") + "<h1>" +
    escapeHtml(name) + "</h1>" + (description ? '<p class="subtitle">' + escapeHtml(description) + "</p>" : "") +
    "</div>" + action + "</header>";
}

function refreshNav(context) {
  for (const key of ["collection", "received"]) {
    const node = document.getElementById("nav-" + key);
    if (key === context) node.setAttribute("aria-current", "page");
    else node.removeAttribute("aria-current");
  }
  recent.innerHTML = model.list.length ? "<small>Cataloged windows</small>" + model.list.slice(0, 6).map(item =>
    '<a href="#window/' + item.id + '"' + current(item.id === currentWindow()?.id) + ">" + escapeHtml(item.title) + "</a>").join("") : "";
}

function collectionView() {
  refreshNav("collection");
  const grid = model.list.map(item => {
    const visual = item.thumb ? '<img loading="lazy" src="' + imageSource(item.thumb, "thumbnail") + '" alt="">' :
      '<span class="tile-placeholder" aria-hidden="true">▥</span>';
    return '<a href="#window/' + item.id + '" class="tile">' + visual + '<div class="tile-body"><h3>' +
      escapeHtml(item.title) + "</h3><p>" + escapeHtml([item.code, item.location].filter(Boolean).join(" · ") || "Window record") +
      "</p><p>" + item.photo_count + " photographs · " + item.archive_count + " evidence records" +
      (item.synthetic ? " · Synthetic" : "") + "</p></div></a>";
  }).join("");
  mount.innerHTML = pageHeading("Window collection", "Catalog photographs, examine visible glass, and gather evidence.",
      button("New window", "new-window", "primary")) +
    '<section class="hero"><div class="hero-inner"><h2>Examine glass under changing light</h2>' +
    "<p>Load a synthetic panel with multiple lighting views, marked observations, and two inspection dates.</p>" +
    button("Load demonstration", "demo", "brass") + "</div></section>" +
    '<section class="collection-section"><div class="section-head"><h2>Cataloged windows</h2><span>' +
    model.list.length + " records</span></div>" + (grid ? '<div class="deck">' + grid + "</div>" :
      empty("Create a window record to begin, or load the demonstration above.")) + "</section>";
}

function listCards(items, renderer, fallback) {
  return items.length ? '<div class="record-list">' + items.map(renderer).join("") + "</div>" : empty(fallback);
}

function captureLabel(capture) {
  return capture.caption || capture.file_name;
}

function photoOptions(captures, fallback = "Choose a photograph") {
  return { "": fallback, ...Object.fromEntries(captures.map(item => [item.id, captureLabel(item) + (item.visit ? " · " + item.visit : "")])) };
}

function capturePicture() {
  const currentPhoto = photo();
  if (!currentPhoto) return "";
  const settings = { plane: model.plane, marks: model.marks };
  if (model.display) settings.display = model.display;
  return "/api/captures/" + currentPhoto.id + "/picture?view=" + encodeURIComponent(JSON.stringify(settings)) +
    "&rev=" + currentPhoto.version + "&a=" + (model.item.marks[0]?.updated || currentPhoto.alignment?.id || "");
}

function boardView(item) {
  if (!photo()) return '<section class="panel">' + empty("Add a photograph to inspect this window.") + "</section>";
  const selection = photo();
  const strip = item.captures.map(capture => '<button type="button" data-action="select-photo" data-id="' + capture.id +
    '"' + current(capture.id === model.capture) + "><strong>" + escapeHtml(roles[capture.role]) +
    "</strong><small>" + escapeHtml(capture.visit || capture.day || captureLabel(capture)) + "</small></button>").join("");
  return '<div><section class="panel photo" aria-label="Photograph viewer"><div class="photo-toolbar"><strong>' +
    escapeHtml(captureLabel(selection)) + "</strong>" +
    controls(button(model.plane ? "Source view" : "Window plane", "plane"),
      button(model.marks ? "Hide marks" : "Show marks", "overlay"), button("Reset view", "reset-view")) +
    '</div><div class="photo-board"><div class="photo-area" id="photo-area">' +
    '<img id="active-picture" alt="' + attr(captureLabel(selection)) + '"><canvas id="point-layer"></canvas></div></div>' +
    '<div class="photo-note"><span id="basis-note"></span><span>' + escapeHtml(selection.visit || "Capture") +
    (selection.day ? " · " + escapeHtml(selection.day) : "") + (selection.synthetic ? " · Synthetic" : "") +
    '</span></div></section><div class="strip" aria-label="Photographs">' + strip + "</div>" +
    '<div id="bench-result"></div></div>';
}

function windowView() {
  const item = model.item;
  const win = item.window;
  refreshNav("window");
  const description = [win.code, win.location, win.physical_mm?.join(" × ") && win.physical_mm.join(" × ") + " mm",
    win.synthetic ? "Synthetic demonstration" : ""].filter(Boolean).join(" · ");
  mount.innerHTML = pageHeading(win.title, description,
    controls(button("Edit window", "edit-window"), button("Photo intake", "intake"), button("Add photograph", "add-photo", "primary")),
    '<a href="#collection">Window collection</a> / Examination desk') +
    '<nav class="tabs" aria-label="Examination desk">' +
    Object.entries(tabs).map(([key, label]) => '<button type="button" data-action="tab" data-id="' + key +
      '"' + current(key === model.tab) + ">" + label + "</button>").join("") + "</nav>" +
    '<div class="workspace">' + boardView(item) + '<aside class="panel inspect" id="inspector"></aside></div>';
  const view = photo();
  const image = document.getElementById("active-picture");
  if (image) image.addEventListener("load", drawPoints);
  setInspector();
  syncBoard();
  if (view && model.opened) {
    renderResult(model.opened);
  }
}

function syncBoard() {
  const area = document.getElementById("photo-area");
  const image = document.getElementById("active-picture");
  const basis = document.getElementById("basis-note");
  if (!area || !image || !basis) return;
  area.toggleAttribute("data-cursor", !!model.pointMode);
  const url = model.active || capturePicture();
  if (image.getAttribute("src") !== url) image.setAttribute("src", url);
  basis.textContent = model.active ? "Examination preview" : model.plane && photo()?.alignment ? "Registered window plane" : "Original photograph";
  drawPoints();
}

function drawPoints() {
  const canvas = document.getElementById("point-layer");
  const image = document.getElementById("active-picture");
  if (!canvas || !image?.clientWidth || !image.clientHeight) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(image.clientWidth * dpr);
  canvas.height = Math.round(image.clientHeight * dpr);
  const context = canvas.getContext("2d");
  if (!model.points.length) return;
  context.lineWidth = 2 * dpr;
  context.strokeStyle = "#fff1a8";
  context.fillStyle = "#aa562a";
  context.setLineDash([5 * dpr, 3 * dpr]);
  const points = model.points.map(([x, y]) => [x * canvas.width, y * canvas.height]);
  context.beginPath();
  points.forEach(([x, y], idx) => idx ? context.lineTo(x, y) : context.moveTo(x, y));
  context.stroke();
  context.setLineDash([]);
  for (const [x, y] of points) {
    context.beginPath();
    context.arc(x, y, 5 * dpr, 0, Math.PI * 2);
    context.fill();
    context.stroke();
  }
}

function setInspector() {
  const holder = document.getElementById("inspector");
  if (!holder) return;
  const general = { evidence: evidencePane, visits: visitPane };
  const photographic = { capture: capturePane, align: alignmentPane, mark: markPane, measure: measurePane, compare: comparePane };
  holder.innerHTML = "<h2>" + escapeHtml(tabs[model.tab]) + "</h2>" +
    (general[model.tab]?.() ?? (photo() ? photographic[model.tab]?.() : empty("Add a photograph to use this workspace.")));
}

function capturePane() {
  const item = photo();
  return "<h3>" + escapeHtml(roles[item.role]) + '</h3><p class="help">' + escapeHtml(item.notes || captureLabel(item)) + "</p>" +
    controls(button("Edit capture", "edit-photo"), link("Original file", imageSource(item.id, "original")),
      button("Inspect source file", "read-source")) +
    form("tone", '<h3>Display levels</h3>' + row(input("Black point", "low", "number", model.display?.low ?? 0, 'step="0.001" min="0" max=".99" required'),
      input("White point", "high", "number", model.display?.high ?? 1, 'step="0.001" min=".01" max="1" required')) +
      selector("Display channel", "channel", { rgb: "Color", luminance: "Luminance", red: "Red", green: "Green", blue: "Blue" },
        model.display?.channel || "rgb") + submit("Apply levels")) +
    form("probe", '<h3>Original pixel probe</h3><p class="help">Choose a point in the displayed photo. The readout uses unadjusted linear values.</p>' +
      row(input("X position", "x", "number", model.points[0]?.[0] ?? .5, 'min="0" max="1" step="any" required'),
        input("Y position", "y", "number", model.points[0]?.[1] ?? .5, 'min="0" max="1" step="any" required')) +
      input("Radius (pixels)", "radius", "number", 4, 'min="0" max="35"') + controls(button("Pick position", "pick-probe"), submit("Read probe"))) +
    '<div id="probe-reading">' + (model.probe ? details(model.probe) : "") + "</div>";
}

function alignmentPane() {
  const setup = photo();
  if (["incident", "dark"].includes(setup.role)) return empty("Reference captures are sampled in the camera coordinates.");
  return '<p class="help">Use the source image to pick the four corners clockwise from upper left.</p>' +
    form("alignment", textarea("Corners [x, y]", "points", formatPoints(setup.alignment?.capture_corners ?? [[0,0],[1,0],[1,1],[0,1]]), "required") +
      controls(button("Pick four corners", "pick-align"), button("Clear points", "clear-points")) +
      controls(button("Preview alignment", "preview-align"), submit("Save alignment")) +
      '<p id="align-note" class="status">' + (setup.alignment ? "Alignment saved · " + Math.round(setup.alignment.support_fraction * 100) + "% photographed support" :
        "No saved alignment for this photograph.") + "</p>") + (setup.alignment ?
       controls(button("Check alignment landmarks", "new-check")) + listCards(model.item.alignment_checks.filter(item => item.capture === setup.id),
         item => '<article class="list-item"><strong>' + escapeHtml(item.title) + "</strong>" + controls(button("Open landmark check", "view-record", "",item.id)) + "</article>", "") : "");
}

function formatPoints(points) {
  return points ? JSON.stringify(points) : "";
}

function markPane() {
  const existing = model.item.marks.find(item => item.id === model.editingMark);
  return '<p class="help">Draw a region or trace on ' + (photo().alignment ? "the registered window." : "the photograph.") + "</p>" +
    form("mark", input("Observation name", "title", "text", existing?.title, 'required maxlength="100"') +
      selector("Observation kind", "category", categories, existing?.category || "pane") +
      row(selector("Shape", "shape", { polygon: "Closed region", line: "Open trace" }, existing?.shape || "polygon"),
        selector("Status", "state", states, existing?.state || "observed")) +
      textarea("Vertices [x, y]", "points", formatPoints(existing?.points || model.points), "required") +
      textarea("Notes", "notes", existing?.notes) +
      '<details><summary>Holes and archive</summary>' +
      textarea("Contained outlines", "holes", formatPoints(existing?.holes || [])) + checkbox("Archived observation", "retired", existing?.retired) + "</details>" +
      controls(button("Draw on photo", "pick-mark"), button("Clear points", "clear-points")) +
      controls(submit(existing ? "Save changes" : "Save observation"), existing ? button("New observation", "new-mark") : "")) +
    listCards(model.item.marks, item => '<article class="list-item"><strong>' + escapeHtml(item.title) + "</strong><small>" +
      escapeHtml(categories[item.category] + " · " + (item.retired ? "Archived" : states[item.state])) + "</small>" +
      controls(button("Edit mark", "edit-mark", "", item.id), button("Inspect", "view-record", "", item.id)) + "</article>", "No marked observations.");
}

const methods = {
  exposure: "Exposure and readability", panes: "Visible glass inventory", rim: "Inside edge condition",
  neighbours: "Adjacent pane marks", came: "Came strip examination", fracture: "Fine fracture candidates",
  "trace-log": "Pinned fracture traces", profile: "Optical line profile", grain: "Glass texture spectrum",
  transmission: "Transmitted-light ratio", paired_detail: "Surface detail across lights",
  color: "Photographed glass color", visit: "Registered visit comparison"
};
const numericOption = (label, name, value, min, max, step = "any") =>
  input(label, name, "number", value, 'data-option="number" step="' + step + '" min="' + min + '" max="' + max + '" required');
const boxOption = () => input("Area [x₀, y₀, x₁, y₁]", "box", "text", "", 'data-option="json" placeholder="[0.15, 0.20, 0.75, 0.80]"');
const markOptions = kind => Object.fromEntries(model.item.marks.filter(item => !item.retired && (!kind || item.category === kind))
  .map(item => [item.id, item.title]));

function measureOptions() {
  const method = model.tool;
  switch (method) {
    case "exposure":
      return row(numericOption("Dark floor", "dark_floor", .018, .0005, .3), numericOption("Bright ceiling", "bright_ceiling", .95, .35, 1)) + boxOption();
    case "rim":
      return selector("Glass piece", "pane", { "": "Choose a piece", ...markOptions("pane") }) + numericOption("Rim depth (pixels)", "rim_width", 10, 2, 80);
    case "neighbours":
      return numericOption("Edge tolerance (pixels)", "distance", 4, 1, 24);
    case "came":
      return numericOption("Strip half width (pixels)", "half_width", 12, 2, 35) +
        numericOption("Sampling step (pixels)", "step", 5, 1, 35);
    case "fracture":
    case "grain":
      return boxOption();
    case "profile":
      return textarea("Profile line [x, y]", "points", formatPoints(model.points.length ? model.points : [[.16,.42],[.84,.55]]),
        'data-option="json" required') + numericOption("Band half width", "half_width", 4, 0, 40, 1) +
        controls(button("Draw profile line", "pick-profile"), button("Clear points", "clear-points"));
    case "transmission":
      return selector("Reference method", "method", { field: "Per-pixel field", uniform: "Uniform reference", patch: "Reference patch" }, "field") +
        selector("Incident light reference", "reference", photoOptions(model.item.captures.filter(item => item.role === "incident"))) +
        selector("Dark reference", "dark", photoOptions(model.item.captures.filter(item => item.role === "dark"), "No dark reference")) +
        input("Reference patch [x₀, y₀, x₁, y₁]", "reference_box", "text", "", 'data-option="json"') + boxOption();
    case "paired_detail":
      return selector("Other lighting capture", "comparison",
        photoOptions(model.item.captures.filter(item => item.alignment && item.id !== model.capture))) + boxOption();
    case "color":
      return selector("Neutral patch", "neutral_mark", { "": "As photographed", ...markOptions("reference") }) +
        numericOption("Swatches", "swatches", 7, 2, 12, 1) + boxOption();
    case "visit":
      return selector("Comparison capture", "comparison",
        photoOptions(model.item.captures.filter(item => item.alignment && item.role === photo()?.role && item.id !== model.capture)), model.selectedPair) +
        selector("Measurement", "difference_kind", { luminance: "Linear luminance", color: "Color distance" }) +
        selector("Stable patch", "reference_mark", { "": "Use exposure as recorded", ...markOptions("reference") }) + boxOption();
    default:
      return "";
  }
}

function measurePane() {
  return form("report", selector("Examination tool", "tool", { ...methods, ...model.tools }, model.tool) +
    '<div id="measure-options">' + measureOptions() + "</div>" +
    input("Examination title", "title", "text", "", 'maxlength="100"') +
    controls(button("Preview map", "preview-report"), submit("Save examination"))) +
    listCards(model.item.reports.slice(0, 14), item => '<article class="list-item"><strong>' + escapeHtml(item.title) +
      "</strong><small>" + escapeHtml(methods[item.tool] || item.tool) + "</small>" +
      controls(button("Open result", "view-record", "", item.id), link("CSV", "/api/reports/" + item.id + "/data.csv")) +
      "</article>", "No saved examinations for this window.");
}

function comparePane() {
  if (!photo()?.alignment) return empty("Register this capture before comparing it with another view.");
  const list = model.item.captures.filter(item => item.alignment && item.id !== model.capture);
  return '<p class="help">Place the two registered photographs over the same window coordinates.</p>' +
    form("wipe", selector("Comparison capture", "comparison", photoOptions(list), model.selectedPair) +
      selector("Wipe mode", "wipe", { vertical: "Vertical divider", horizontal: "Horizontal divider", checker: "Checkerboard" }) +
      input("Divider position", "boundary", "number", .5, 'min=".05" max=".95" step=".05" required') +
      controls(submit("Show two captures"), button("Measure visit change", "visit-tool")));
}

function checklist(label, name, items) {
  return "<strong>" + escapeHtml(label) + '</strong><div class="check-list">' +
    (items.length ? items.map(item => checkbox(item.title || item.caption || item.id, name, false, item.id)).join("") :
      '<p class="help">No records available.</p>') + "</div>";
}

function evidencePane() {
  const item = model.item;
  return '<p class="help">Gather selected photographs and saved examinations into a contact sheet or evidence packet.</p>' +
    controls(button("Create evidence", "package"), button("Print detail at scale", "proof")) +
    listCards(item.packets.slice(0, 18), record => '<article class="list-item"><strong>' + escapeHtml(record.title) +
      "</strong><small>" + escapeHtml(title(record.packet_kind)) + "</small>" +
      controls(button("Open evidence", "view-record", "", record.id), link("PDF", imageSource(record.id, "pdf")),
        record.packet_kind === "frozen evidence" ? link("ZIP", imageSource(record.id, "zip")) : "") + "</article>",
      "No exported evidence.") +
    (item.references.length ? '<h3>Approved swatches</h3>' + listCards(item.references, record =>
      '<article class="list-item"><strong>' + escapeHtml(record.name) + "</strong><small>" + escapeHtml(title(record.state)) +
      "</small>" + controls(button("Open sample", "view-record", "", record.id)) + "</article>", "") : "");
}

function visitPane() {
  return '<p class="help">Keep dated notes with the exact version of each cited examination.</p>' +
    controls(button("Log an inspection", "new-visit", "primary"), button("Create condition sheet", "new-condition")) +
    listCards([...model.item.visits].sort((a,b) => b.day.localeCompare(a.day)), item => '<article class="list-item"><strong>' +
      escapeHtml(item.title) + "</strong><small>" + escapeHtml([item.day, title(item.visit_type), item.archived ? "Archived" : ""].filter(Boolean).join(" · ")) +
      "</small><p>" + escapeHtml(item.notes) + "</p>" +
      controls(button("Open log entry", "view-record", "", item.id), button("Edit entry", "edit-visit", "", item.id)) +
      "</article>", "No dated entries.") + (model.item.conditions.length ? '<h3>Condition sheets</h3>' +
      listCards(model.item.conditions, item => '<article class="list-item"><strong>' + escapeHtml(item.title) + "</strong>" +
        controls(button("Open condition sheet", "view-record", "", item.id)) + "</article>", "") : "");
}

function details(data) {
  return '<dl class="summary-grid">' + Object.entries(data).filter(([,v]) => scalar(v)).slice(0, 24)
    .map(([key, val]) => "<dt>" + escapeHtml(title(key)) + "</dt><dd>" + escapeHtml(textFormat(val)) + "</dd>").join("") + "</dl>";
}

function pictureAssets(item) {
  return (item.assets || []).map(asset => {
    const href = imageSource(item.id, asset.label);
    if (asset.mime === "image/png" && ["map", "chart", "thumbnail", "plane", "support", "diff", "photo"].includes(asset.label)) {
      return '<figure class="panel"><img loading="lazy" src="' + href + '" alt="' + attr(title(asset.label)) + '"><figcaption>' +
        escapeHtml(title(asset.label)) + "</figcaption></figure>";
    }
    return "";
  }).join("");
}

function showRecord(item) {
  const overview = item.reading || item.measure || item;
  const resources = (item.assets || []).filter(asset => asset.mime !== "image/png" || asset.label === "original")
    .map(asset => link("Download " + title(asset.label), imageSource(item.id, asset.label)));
  const csvPaths = { reports: "reports", alignment_checks: "alignment-checks", condition_sheets: "condition-sheets" };
  if (csvPaths[item.kind]) resources.push(link("Download CSV", "/api/" + csvPaths[item.kind] + "/" + item.id + "/data.csv"));
  if (item.tool === "color") resources.push(button("Approve as swatch", "approve", "", item.id));
  if (item.kind === "references") resources.push(button(item.state === "retired" ? "Restore reference" : "Retire reference", "toggle-ref", "", item.id));
  openModal(item.title || item.name || item.caption || "Record", controls(...resources, button("Check stored evidence", "audit", "", item.id)) +
    details(overview) + '<div class="reader">' + pictureAssets(item) + '</div><div id="audit-result"></div><details><summary>Recorded values and provenance</summary><pre>' +
    escapeHtml(JSON.stringify(item, null, 2)) + "</pre></details>", true);
}

function renderResult(item) {
  const result = document.getElementById("bench-result");
  if (!result || !item) return;
  result.innerHTML = '<section class="panel record"><div class="section-head"><h3>' + escapeHtml(item.title) + "</h3>" +
    button("Open result", "view-record", "", item.id) + "</div>" + details(item.reading || item.measure || item) + "</section>";
}

function receivedView(items) {
  refreshNav("received");
  mount.innerHTML = pageHeading("Received evidence", "Verify and open a LumePane evidence ZIP from the local library.") +
    '<section class="panel">' + form("receive", '<label>Evidence ZIP<input type="file" name="packet" accept=".zip,application/zip" required></label>' +
      controls(submit("Verify and receive packet"))) + "</section>" +
    '<section class="collection-section"><div class="section-head"><h2>Received packets</h2></div>' +
    listCards(items, item => '<article class="panel list-item"><strong>' + escapeHtml(item.title) + "</strong>" +
      controls(button("Open received views", "received-open", "", item.id), button("Check stored copy", "view-record", "", item.id)) +
      "</article>", "No received evidence yet.") + "</section>";
}

function windowForm(item = {}) {
  return input("Window name", "title", "text", item.title, 'required maxlength="120"') +
    row(input("Reference code", "code", "text", item.code), input("Location", "location", "text", item.location)) +
    row(input("Physical width (mm)", "width", "number", item.physical_mm?.[0], 'min="5" max="12000" step="any"'),
      input("Physical height (mm)", "height", "number", item.physical_mm?.[1], 'min="5" max="12000" step="any"')) +
    textarea("Window notes", "notes", item.notes) + submit(item.id ? "Save window" : "Create window");
}

function photoForm(item = {}, upload = false) {
  const fileInput = upload ? '<label>Image file<input type="file" name="image" accept="image/png,image/jpeg,image/tiff,image/webp,.tiff,.tif" required></label>' : "";
  return fileInput + input("Caption", "caption", "text", item.caption) +
    (upload ? selector("Lighting", "role", roles, "transmitted") + input("File page", "page", "number", 1, 'min="1" max="50"') : "") +
    row(input("Inspection label", "visit", "text", item.visit), input("Photographed on", "day", "date", item.day)) +
    textarea("Capture notes", "notes", item.notes) + submit(upload ? "Add to window" : "Save capture");
}

function fieldsForWindow(element) {
  const original = element.dataset.form === "edit-window" ? currentWindow() : {};
  const data = { title: value(element, "title"), code: value(element, "code"), location: value(element, "location"),
    notes: value(element, "notes"), working_extent: original.working_extent, version: original.version };
  if (value(element, "width") || value(element, "height")) {
    data.physical_mm = [number(element, "width"), number(element, "height")];
  }
  return data;
}

function selectMany(node, key) {
  return [...node.querySelectorAll('input[name="' + key + '"]:checked')].map(input => input.value);
}

function findForm(key) {
  return document.querySelector('[data-form="' + key + '"]');
}

function parsePoints(form, key, fallback) {
  const content = value(form, key).trim();
  if (!content && fallback !== undefined) return fallback;
  try {
    return JSON.parse(content);
  } catch {
    throw new Error("Enter coordinates as JSON lists of numbers.");
  }
}

function measureData(element) {
  const options = {};
  for (const node of element.querySelectorAll("#measure-options input, #measure-options textarea, #measure-options select")) {
    if (node.value === "") continue;
    if (node.name === "pane") {
      options.panes = [node.value];
      continue;
    }
    options[node.name] = node.dataset.option === "json" ? parsePoints(element, node.name) :
      node.dataset.option === "number" ? Number(node.value) : node.value;
  }
  return {
    tool: model.tool, capture: model.capture, native: !model.plane,
    title: value(element, "title") || undefined, options
  };
}

function alignData(element) {
  return { corners: parsePoints(element, "points"), capture_version: photo().version, window_version: currentWindow().version };
}

function underWindow(path = "") {
  return "/windows/" + currentWindow().id + path;
}

function underCapture(path) {
  return "/captures/" + model.capture + path;
}

async function refreshWindow() {
  const result = await request(underWindow());
  if (result.window.id === currentWindow()?.id) {
    model.item = result;
    windowView();
  }
}

function pointStart(mode) {
  model.active = null;
  model.points = [];
  model.pointMode = mode;
  const wantedPlane = mode === "mark" ? (model.editingMark ?
    model.item.marks.find(mark => mark.id === model.editingMark)?.coordinate_basis === "window" : true) :
    mode === "align" || mode === "survey" ? false : model.plane;
  if (wantedPlane !== model.plane) model.plane = wantedPlane;
  const element = findForm(mode === "align" ? "alignment" : mode === "profile" ? "report" : mode === "survey" ? "survey" : mode);
  if (element?.elements.points) element.elements.points.value = "";
  syncBoard();
  flash(mode === "align" ? "Click four corners, beginning at the upper left."
    : mode === "survey" ? "Click the planned survey position."
    : mode === "probe" ? "Click the point to read." : "Click points on the photograph to draw.");
}

function pointClick(event) {
  if (!model.pointMode) return;
  const image = document.getElementById("active-picture");
  const frame = image.getBoundingClientRect();
  const clamp = value => Math.round(Math.max(0, Math.min(1, value)) * 100000) / 100000;
  const point = [clamp((event.clientX - frame.left) / frame.width), clamp((event.clientY - frame.top) / frame.height)];
  if (model.pointMode === "probe" || model.pointMode === "survey") {
    const form = findForm(model.pointMode);
    form.elements.x.value = point[0];
    form.elements.y.value = point[1];
    model.points = [point];
    model.pointMode = null;
  } else {
    const key = { align: "alignment", profile: "report", mark: "mark" }[model.pointMode];
    model.points.push(point);
    const form = findForm(key);
    form.elements.points.value = formatPoints(model.points);
    if (model.pointMode === "align" && model.points.length === 4) model.pointMode = null;
  }
  syncBoard();
}

function exportForm() {
  const item = model.item;
  return form("packet", input("Evidence title", "title", "text", "", 'required maxlength="90"') +
    selector("Export", "kind", { packets: "Evidence ZIP and PDF", contact: "Contact sheet PDF" }) +
    row(input("Columns", "columns", "number", 2, 'min="1" max="4" required'),
      input("Rows", "rows", "number", 2, 'min="1" max="4" required')) +
    selector("Photograph view", "view", { plane: "Window plane", capture: "Original photograph" }) +
    checkbox("Show marked observations", "marks", true) +
    checklist("Include dated photographs and saved examinations", "records", [...item.captures, ...item.reports]) +
    submit("Create packet"));
}

function proofForm() {
  return form("proof", input("Proof title", "title", "text", "", 'required maxlength="85"') +
    selector("Capture", "capture", photoOptions(model.item.captures.filter(item => item.alignment)), photo()?.id) +
    input("Crop [x₀, y₀, x₁, y₁]", "box", "text", "[0.20, 0.25, 0.45, 0.48]", "required") +
    submit("Create scale proof"));
}

function visitForm(item) {
  const data = model.item;
  const type = {
    initial: "Initial examination", review: "Condition review", followup: "Follow-up photography",
    treatment_record: "Recorded intervention", handover: "Documentation handover"
  };
  return input("Entry title", "title", "text", item?.title, item ? "disabled" : 'required maxlength="100"') +
    row(input("Inspection date", "day", "date", item?.day || today(), item ? "disabled" : "required"),
      selector("Visit type", "visit_type", item ? { [item.visit_type]: type[item.visit_type] } : type, item?.visit_type || "review")) +
    (item ? "" : input("Operator", "operator", "text", "") +
      input("Labels, comma separated", "tags", "text", "") +
      checklist("Pin evidence references", "refs", [...data.captures, ...data.marks, ...data.reports, ...data.packets])) +
    textarea("Inspection notes", "notes", item?.notes) +
    (item ? checkbox("Archive this entry", "archived", item.archived) : "") + submit("Save log entry");
}

async function openExisting(key) {
  const item = await request("/records/" + key);
  model.opened = item;
  showRecord(item);
}

const clicks = {
  close() { modal.close(); },
  async demo() {
    await request("/demonstration", "POST", {});
    await refreshRoute();
    flash("Synthetic windows are ready.");
  },
  "new-window"() { openModal("New window", form("new-window", windowForm())); },
  "edit-window"() { openModal("Edit window", form("edit-window", windowForm(currentWindow()))); },
  "add-photo"() { openModal("Add photograph", form("add-photo", photoForm({}, true))); },
  "edit-photo"() { openModal("Edit capture", form("edit-photo", photoForm(photo()))); },
  "select-photo"(key) {
    clearPhoto();
    model.capture = key;
    model.plane = true;
    model.pointMode = null;
    model.editingMark = null;
    model.opened = null;
    windowView();
  },
  tab(key) {
    clearPhoto();
    model.tab = key;
    model.pointMode = null;
    model.plane = key !== "align";
    windowView();
  },
  plane() {
    model.plane = !model.plane;
    clearPhoto();
    model.pointMode = null;
    windowView();
  },
  overlay() {
    model.marks = !model.marks;
    windowView();
  },
  "reset-view"() {
    clearPhoto();
    model.plane = model.tab !== "align";
    model.marks = true;
    model.display = null;
    model.opened = null;
    model.pointMode = null;
    windowView();
  },
  "pick-align"() { pointStart("align"); },
  "pick-probe"() { pointStart("probe"); },
  "pick-profile"() { pointStart("profile"); },
  "pick-mark"() { pointStart("mark"); },
  "pick-survey"() { pointStart("survey"); },
  "clear-points"() {
    model.points = [];
    const key = { mark: "mark", measure: "report", align: "alignment" }[model.tab];
    const el = findForm(key)?.elements.points;
    if (el) el.value = "";
    drawPoints();
  },
  "new-mark"() {
    model.active = null;
    model.editingMark = null;
    model.points = [];
    setInspector();
    syncBoard();
  },
  "edit-mark"(key) {
    model.active = null;
    const mark = model.item.marks.find(item => item.id === key);
    model.capture = mark.source_capture;
    model.plane = mark.coordinate_basis === "window";
    model.points = mark.points;
    model.editingMark = key;
    windowView();
  },
  async "preview-align"() {
    const element = findForm("alignment");
    if (!element.reportValidity()) return;
    const blob = await request(underCapture("/align/preview"), "POST", alignData(element), true);
    model.pointMode = null;
    model.points = [];
    usePicture(blob);
    document.getElementById("align-note").textContent = "Preview ready. Save alignment to use this window plane.";
  },
  async "preview-report"() {
    const element = findForm("report");
    if (!element.reportValidity()) return;
    const blob = await request(underWindow("/reports/preview"), "POST", measureData(element), true);
    model.pointMode = null;
    model.points = [];
    usePicture(blob);
    flash("Examination preview ready.");
  },
  "visit-tool"() {
    model.selectedPair = value(findForm("wipe"), "comparison");
    model.tool = "visit";
    model.active = null;
    model.tab = "measure";
    windowView();
  },
  async "view-record"(key) { await openExisting(key); },
  async audit(key) {
    const result = await request("/records/" + key + "/check");
    document.getElementById("audit-result").innerHTML = "<h3>Stored evidence check</h3>" + details(result) +
      (result.assets?.length ? '<div class="record-list">' + result.assets.map(asset =>
        '<p class="list-item"><strong>' + escapeHtml(title(asset.label)) + "</strong><small>" +
        escapeHtml(asset.hash_matches ? "Stored bytes match the recorded digest" : "Stored bytes could not be verified") +
        "</small></p>").join("") + "</div>" : "");
  },
  approve(key) {
    const label = model.opened?.title;
    const element = form("approve", input("Sample name", "title", "text", label, 'required maxlength="100"') +
      input("Glass series", "family", "text") + textarea("Notes", "notes") + submit("Approve sample"));
    openModal("Approve color sample", element);
    findForm("approve").dataset.id = key;
  },
  async "toggle-ref"(key) {
    const item = model.opened;
    const result = await request("/references/" + key, "PUT", {
      version: item.version, state: item.state === "active" ? "retired" : "active"
    });
    await refreshWindow();
    model.opened = result;
    showRecord(result);
  },
  package() { openModal("Create evidence", exportForm()); },
  proof() { openModal("Print detail at scale", proofForm()); },
  "new-visit"() { openModal("Log an inspection", form("new-visit", visitForm(null))); },
  "edit-visit"(key) {
    const existing = model.item.visits.find(item => item.id === key);
    openModal("Edit log entry", form("edit-visit", visitForm(existing)));
    findForm("edit-visit").dataset.id = key;
  },
  async "received-open"(key) {
    const item = await request("/records/" + key);
    const views = item.views || [];
    openModal(item.title, '<div class="reader">' + views.map((view, index) =>
      '<figure class="panel"><img loading="lazy" src="/api/receive/' + item.id + "/view/" + index +
      '" alt="' + attr(view.caption) + '"><figcaption>' + escapeHtml(view.caption) + "</figcaption></figure>").join("") + "</div>" +
      (views.length ? "" : empty("No preview pictures are attached."))
      + controls(button("Check stored evidence", "audit", "", key)) + '<div id="audit-result"></div>', true);
  }
};

const forms = {
  async "new-window"(el) {
    const item = await request("/windows", "POST", fieldsForWindow(el));
    modal.close();
    routeTo(item.id);
    return "Window created.";
  },
  async "edit-window"(el) {
    await request(underWindow(), "PUT", fieldsForWindow(el));
    modal.close();
    await refreshWindow();
    return "Window updated.";
  },
  async "add-photo"(el) {
    const result = await request(underWindow("/captures"), "POST", new FormData(el));
    modal.close();
    clearPhoto();
    model.capture = result.capture.id;
    model.tab = "capture";
    model.plane = true;
    await refreshWindow();
    return result.duplicate ? "This photograph was already in the window." : "Photograph added.";
  },
  async "edit-photo"(el) {
    await request(underCapture(""), "PATCH", { caption: value(el, "caption"), visit: value(el, "visit"),
      day: value(el, "day"), notes: value(el, "notes"), version: photo().version });
    modal.close();
    await refreshWindow();
    return "Capture updated.";
  },
  tone(el) {
    const display = { low: number(el, "low"), high: number(el, "high"), channel: value(el, "channel") };
    if (display.high <= display.low) throw new Error("White point must be higher than black point.");
    model.display = display;
    model.active = null;
    syncBoard();
    return "Display levels applied.";
  },
  async probe(el) {
    model.probe = await request(underCapture("/probe"), "POST",
      { x: number(el, "x"), y: number(el, "y"), radius: number(el, "radius"), native: !model.plane });
    document.getElementById("probe-reading").innerHTML = details(model.probe) +
      '<details><summary>Probe coordinates and channels</summary><pre>' + escapeHtml(JSON.stringify(model.probe, null, 2)) + "</pre></details>";
    return "Original pixel values read.";
  },
  async alignment(el) {
    await request(underCapture("/align"), "POST", alignData(el));
    clearPhoto();
    model.plane = true;
    await refreshWindow();
    return "Alignment saved.";
  },
  async mark(el) {
    const old = model.item.marks.find(item => item.id === model.editingMark);
    const data = { title: value(el, "title"), category: value(el, "category"), shape: value(el, "shape"), notes: value(el, "notes"),
      state: value(el, "state"), points: parsePoints(el, "points"), holes: parsePoints(el, "holes", []), retired: checked(el, "retired"),
      capture: model.capture, native: !model.plane, version: old?.version };
    const item = await request(underWindow("/marks" + (old ? "/" + old.id : "")), old ? "PATCH" : "POST", data);
    clearPhoto();
    model.editingMark = null;
    model.opened = item;
    await refreshWindow();
    return "Observation saved.";
  },
  async report(el) {
    const item = await request(underWindow("/reports"), "POST", measureData(el));
    clearPhoto();
    model.opened = item;
    await refreshWindow();
    showRecord(item);
    return "Examination saved.";
  },
  async wipe(el) {
    model.selectedPair = value(el, "comparison");
    const blob = await request(underCapture("/wipe"), "POST",
      { comparison: model.selectedPair, wipe: value(el, "wipe"), boundary: number(el, "boundary") }, true);
    usePicture(blob);
    return "Comparison ready.";
  },
  async approve(el) {
    const reference = await request("/reports/" + idOf(el) + "/approve", "POST", {
      title: value(el, "title"), family: value(el, "family"), notes: value(el, "notes")
    });
    await refreshWindow();
    model.opened = reference;
    showRecord(reference);
    return "Sample approved.";
  },
  async packet(el) {
    const selected = selectMany(el, "records");
    const result = await request(underWindow("/" + value(el, "kind")), "POST", { title: value(el, "title"), records: selected,
      columns: number(el, "columns"), rows: number(el, "rows"), marks: checked(el, "marks"), view: value(el, "view") });
    await refreshWindow();
    model.opened = result;
    showRecord(result);
    return "Evidence packet ready.";
  },
  async proof(el) {
    const result = await request(underWindow("/scaled-proof"), "POST",
      { title: value(el, "title"), capture: value(el, "capture"), box: parsePoints(el, "box") });
    await refreshWindow();
    model.opened = result;
    showRecord(result);
    return "Scale proof ready.";
  },
  async receive(el) {
    await request("/receive", "POST", new FormData(el));
    await refreshRoute();
    return "Evidence received and verified.";
  },
  async "new-visit"(el) {
    await request(underWindow("/visits"), "POST", { title: value(el, "title"), day: value(el, "day"),
      visit_type: value(el, "visit_type"), operator: value(el, "operator"), refs: selectMany(el, "refs"),
      tags: value(el, "tags").split(",").map(tag => tag.trim()).filter(Boolean), notes: value(el, "notes") });
    modal.close();
    await refreshWindow();
    return "Inspection logged.";
  },
  async "edit-visit"(el) {
    const key = idOf(el);
    const old = model.item.visits.find(item => item.id === key);
    await request("/visits/" + key, "PATCH", { version: old.version, notes: value(el, "notes"), archived: checked(el, "archived") });
    modal.close();
    await refreshWindow();
    return "Inspection log updated.";
  }
};

async function refreshRoute() {
  const tick = ++model.epoch;
  const fragment = location.hash.slice(1);
  const key = fragment.match(/^window\/([a-f0-9]{24})$/)?.[1];
  if (key) {
    const [result, list, methods] = await Promise.all([request("/windows/" + key), request("/windows"), request("/tools")]);
    if (tick !== model.epoch) return;
    const current = currentWindow()?.id;
    model.list = list.windows;
    model.item = result;
    model.tools = methods;
    if (key !== current) {
      clearPhoto();
      model.capture = result.captures.find(item => item.role === "transmitted")?.id ?? result.captures[0]?.id;
      model.tab = "capture";
      model.plane = true;
      model.opened = null;
      model.display = null;
      model.selectedPair = "";
      model.editingMark = null;
      model.pointMode = null;
    }
    windowView();
  } else {
    const [list, received] = await Promise.all([request("/windows"), fragment === "received" ? request("/receive") : null]);
    if (tick !== model.epoch) return;
    model.list = list.windows;
    model.item = null;
    clearPhoto();
    if (received) receivedView(received.packets);
    else collectionView();
  }
}

document.addEventListener("click", async event => {
  if (event.target.closest("#photo-area")) pointClick(event);
  const item = event.target.closest("button[data-action]");
  const callback = item ? clicks[item.dataset.action] : null;
  if (!callback) return;
  item.disabled = true;
  try {
    await callback(idOf(item));
  } catch (error) {
    showError(error);
  } finally {
    item.disabled = false;
  }
});

document.addEventListener("submit", async event => {
  const element = event.target.closest("form[data-form]");
  const callback = element ? forms[element.dataset.form] : null;
  if (!callback) return;
  event.preventDefault();
  const button = event.submitter;
  if (button) button.disabled = true;
  try {
    const message = await callback(element);
    if (message) flash(message);
  } catch (error) {
    showError(error);
  } finally {
    if (button) button.disabled = false;
  }
});

document.addEventListener("change", event => {
  if (event.target.matches('[data-form="report"] select[name="tool"]')) {
    model.tool = event.target.value;
    model.active = null;
    model.points = [];
    document.getElementById("measure-options").innerHTML = measureOptions();
    syncBoard();
  }
});
window.addEventListener("hashchange", () => refreshRoute().catch(error => showError(error, mount)));
window.addEventListener("resize", drawPoints);
mount.innerHTML = '<p class="status">Opening the local image library…</p>';
refreshRoute().catch(error => showError(error, mount));

function intakeDialog() {
  openModal("Photo intake", '<p class="help">Inspect local files and individual pages before adding them to the selected window.</p>' +
    form("stage-intake", input("Intake name", "title", "text", "Photo intake selection", 'maxlength="100"') +
      '<label>Local photographs<input type="file" name="images" accept="image/png,image/jpeg,image/tiff,image/webp,.tiff,.tif" multiple required></label>' +
      submit("Check file pages")) + listCards(model.item.intake_sessions, item =>
        '<article class="list-item"><strong>' + escapeHtml(item.title) + "</strong><small>" +
        escapeHtml(title(item.state)) + "</small>" + controls(button("Open intake", "open-intake", "", item.id)) + "</article>", "No staging sessions."));
}

function intakeView(item) {
  const files = item.files.map(record => {
    const read = record.pages.map(page => {
      const tag = record.id + "/" + page.page;
      const name = "Page " + page.page + " · " + page.original_extent.join(" × ");
      return page.available
        ? controls(checkbox(name, "pages", false, tag), link("Preview page " + page.page, "/api/intake/files/" + tag))
        : '<p class="help">' + escapeHtml(name + " · " + page.outcome) + "</p>";
    }).join("");
    return '<article class="list-item"><h3>' + escapeHtml(record.file_name) + "</h3><small>" +
      escapeHtml([record.encoding, record.state, record.same_content_as_earlier_row ? "Repeats an earlier file" : ""].filter(Boolean).join(" · ")) +
      "</small>" + read + (record.explanation ? '<p class="help">' + escapeHtml(record.explanation) + "</p>" : "") + "</article>";
  }).join("");
  const receipts = item.imported.map(photo => '<article class="list-item"><strong>' + escapeHtml(photo.caption) +
    "</strong><small>" + escapeHtml(photo.existing ? "Reused existing photograph" : "Imported") + "</small></article>").join("");
  openModal(item.title, details(item.totals) + controls(link("Intake CSV", "/api/intake/" + item.id + "/data.csv")) +
    (item.state === "staged" ? form("import-intake", files + selector("Assign light to selected pages", "role", roles) +
      row(input("Inspection label", "visit", "text"), input("Photographed on", "day", "date")) +
      submit("Import selected pages")) : receipts), true);
  const element = findForm("import-intake");
  if (element) {
    element.dataset.id = item.id;
    element.dataset.version = item.version;
  }
}

clicks.intake = intakeDialog;
clicks["open-intake"] = async key => { intakeView(await request("/intake/" + key)); };
clicks["read-source"] = async () => {
  const result = await request(underCapture("/source-check"));
  openModal("Original photograph check", details(result.identity) + "<h3>Working dimensions</h3>" + details(result.dimensions) +
    controls(link("Source check CSV", "/api" + underCapture("/source-check/data.csv"))) +
    "<details><summary>Original photograph support</summary><pre>" + escapeHtml(JSON.stringify(result.visibility, null, 2)) + "</pre></details>", true);
};
clicks["new-check"] = () => {
  openModal("Check alignment landmarks", '<p class="help">Enter independent positions for the same feature in each image, using normalized window and original coordinates.</p>' +
    form("new-check", input("Check title", "title", "text", "Alignment landmark check", 'maxlength="100"') +
      textarea("Landmarks with plane and capture coordinates", "landmarks",
        '[{"name":"Upper lead joint","plane":[0.24,0.32],"capture":[0.24,0.32]},{"name":"Lower lead joint","plane":[0.69,0.74],"capture":[0.69,0.74]}]', "required") +
      input("Tolerance in original pixels", "threshold", "number", 4, 'min=".25" max="100" step="any"') +
      submit("Save landmark check")));
};
clicks["new-condition"] = () => {
  openModal("Create condition sheet", '<p class="help">Review observation versions cited in selected visit entries.</p>' +
    form("new-condition", input("Sheet title", "title", "text", "Condition observations", 'maxlength="100"') +
      row(input("Start date", "start", "date"), input("End date", "end", "date")) +
      selector("Photograph for context", "display_photo", photoOptions(model.item.captures, "No context photograph"), photo()?.id) +
      checkbox("Include archived visits", "include_archived") +
      checklist("Specific visits (empty uses date range)", "visits", model.item.visits) + submit("Save condition sheet")));
};
forms["stage-intake"] = async element => {
  const result = await request(underWindow("/intake"), "POST", new FormData(element));
  await refreshWindow();
  intakeView(result);
  return "Staged photographs inspected.";
};
forms["import-intake"] = async element => {
  const pages = selectMany(element, "pages").map(tag => {
    const [file, page] = tag.split("/");
    return { file, page: Number(page) };
  });
  const result = await request("/intake/" + idOf(element) + "/import", "POST", { pages, version: Number(element.dataset.version),
    role: value(element, "role"), visit: value(element, "visit"), day: value(element, "day") });
  if (result.imported.length && !model.capture) model.capture = result.imported[0].capture;
  await refreshWindow();
  intakeView(result);
  return "Selected pages imported.";
};
forms["new-check"] = async element => {
  const result = await request(underCapture("/alignment-checks"), "POST", { title: value(element, "title"),
    threshold: number(element, "threshold"), landmarks: parsePoints(element, "landmarks") });
  await refreshWindow();
  model.opened = result;
  showRecord(result);
  return "Alignment landmark check saved.";
};
forms["new-condition"] = async element => {
  const result = await request(underWindow("/conditions"), "POST", { title: value(element, "title"), start: value(element, "start"),
    end: value(element, "end"), display_photo: value(element, "display_photo"), include_archived: checked(element, "include_archived"),
    visits: selectMany(element, "visits") });
  await refreshWindow();
  model.opened = result;
  showRecord(result);
  return "Condition sheet saved.";
};
