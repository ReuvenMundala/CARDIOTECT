import * as THREE from "three";

const state = {
  bootstrap: null,
  currentSection: "home",
  selectedPath: "",
  studyId: null,
  summary: null,
  overlayMode: "both",
  mprMode: "ai",
  sliceIndex: 0,
  mprIndices: { axial: 0, coronal: 0, sagittal: 0 },
  pollTimer: null,
  toastTimer: null,
  sliceTimer: null,
  mprTimer: null,
  sliceRevision: 0,
  imageUrls: { slice: null, axial: null, coronal: null, sagittal: null },
  maxPanel: null,
  volumeKey: "",
  lastMeshSignature: "",
  syncSliders: false,
  mprRevisions: { axial: 0, coronal: 0, sagittal: 0 },
  mprViewport: {
    axial: { zoom: 1, panX: 0, panY: 0, active: false, pointerId: null, lastX: 0, lastY: 0 },
    coronal: { zoom: 1, panX: 0, panY: 0, active: false, pointerId: null, lastX: 0, lastY: 0 },
    sagittal: { zoom: 1, panX: 0, panY: 0, active: false, pointerId: null, lastX: 0, lastY: 0 },
  },
  chamberFocus: "all",
  anatomyControls: {
    heart: { visible: true, clip: { axial: 100, coronal: 100, sagittal: 100 } },
    coronary: { visible: true, clip: { axial: 100, coronal: 100, sagittal: 100 } },
    chambers: { visible: true, clip: { axial: 100, coronal: 100, sagittal: 100 } },
  },
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const sliderGroups = {
  axial: ["#mpr-axial-slider", "#mpr-axial-slider-full"],
  coronal: ["#mpr-coronal-slider", "#mpr-coronal-slider-full"],
  sagittal: ["#mpr-sagittal-slider", "#mpr-sagittal-slider-full"],
};

const anatomyGroups = {
  heart: ["heart"],
  coronary: ["coronary_arteries"],
  chambers: [
    "heart_myocardium",
    "heart_atrium_left",
    "heart_ventricle_left",
    "heart_atrium_right",
    "heart_ventricle_right",
    "aorta",
    "pulmonary_artery",
  ],
};

const slicerPlaneColors = {
  axial: 0xff5a5a,
  coronal: 0x59d86f,
  sagittal: 0xf4d14b,
};

const threeState = {
  renderer: null,
  scene: null,
  camera: null,
  orthoCamera: null,
  activeCamera: null,
  controls: null,
  pivotGroup: null,
  rootGroup: null,
  meshGroup: null,
  planeGroup: null,
  guideGroup: null,
  actorEntries: [],
  planes: {},
  materials: {},
  isOrtho: false,
  lastFrame: 0,
  boundsKey: "",
  cameraDistance: 700,
  baseCameraDirection: new THREE.Vector3(0.94, -0.88, 0.92).normalize(),
  dragState: { active: false, pointerId: null, button: 0, x: 0, y: 0 },
  objectRotation: { x: 0, y: 0, z: 0 },
};

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => toast.classList.remove("show"), 2400);
}

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let message = `${response.status}`;
    try {
      const payload = await response.json();
      message = payload.error || message;
    } catch {
      message = response.statusText || message;
    }
    throw new Error(message);
  }
  return response.json();
}

async function postJSON(url, body) {
  return fetchJSON(url, { method: "POST", body: JSON.stringify(body || {}) });
}

function openModal(id) {
  const modal = $(`#${id}`);
  if (!modal) return;
  modal.classList.add("active");
  modal.setAttribute("aria-hidden", "false");
}

function closeModal(id) {
  const modal = $(`#${id}`);
  if (!modal) return;
  modal.classList.remove("active");
  modal.setAttribute("aria-hidden", "true");
}

function closeAllModals() {
  $$(".modal.active").forEach((modal) => closeModal(modal.id));
}

function navigateTo(sectionId) {
  const previousSection = state.currentSection;
  state.currentSection = sectionId;
  $$(".nav-btn").forEach((btn) => btn.classList.toggle("active", btn.dataset.section === sectionId));
  $$(".content-section").forEach((section) => section.classList.toggle("active", section.id === sectionId));
  if (sectionId === "analysis" && state.summary?.fastReady) {
    loadSlice(true);
  }
  if (sectionId === "heart3d" && state.summary?.fastReady) {
    if (previousSection !== "heart3d") {
      resetAllMprViewports();
    }
    loadMprImages(true);
    loadMesh().catch((error) => console.error(error));
  }
  if (sectionId === "report") {
    refreshReport();
  }
  requestAnimationFrame(() => {
    applyMprViewportTransform("axial");
    applyMprViewportTransform("coronal");
    applyMprViewportTransform("sagittal");
    resizeThreeRenderer();
    renderThreeFrame();
  });
}

function initNavigation() {
  $$(".nav-btn").forEach((btn) => btn.addEventListener("click", () => navigateTo(btn.dataset.section)));
  $$("[data-jump]").forEach((btn) => btn.addEventListener("click", () => navigateTo(btn.dataset.jump)));
  $("#home-run-analysis-btn").addEventListener("click", () => navigateTo("analysis"));
}

function initUtilityButtons() {
  $("#support-btn").addEventListener("click", () => showToast("Contact support@cardiotect.ai for assistance."));
  $("#docs-btn").addEventListener("click", () => showToast("CARDIOTECT documentation will open from the local bundle when available."));
  $("#settings-btn").addEventListener("click", () => showToast("Performance-sensitive defaults are active for this workstation."));
}

function initModalHandlers() {
  $$("[data-close-modal]").forEach((button) => {
    button.addEventListener("click", () => closeModal(button.dataset.closeModal));
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeAllModals();
    }
  });
}

function registrationPayload() {
  return {
    patientName: $("#patient-name").value.trim(),
    patientMrn: $("#patient-mrn").value.trim(),
    patientAge: Number($("#patient-age").value || 0),
    patientSex: $("#patient-sex").value,
    studyPhysician: $("#study-physician").value.trim(),
    studyReason: $("#study-reason").value.trim(),
    scanDate: new Date().toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric" }),
    riskFactors: {
      hypertension: $("#risk-hypertension").checked,
      hyperlipidemia: $("#risk-hyperlipidemia").checked,
      diabetes: $("#risk-diabetes").checked,
      smoking: $("#risk-smoking").checked,
      family_hx: $("#risk-family_hx").checked,
    },
  };
}

function blankRegistrationPayload() {
  return {
    patientName: "",
    patientMrn: "",
    patientAge: 0,
    patientSex: "",
    studyPhysician: "",
    studyReason: "",
    scanDate: new Date().toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric" }),
    riskFactors: {},
  };
}

function hasSavedRegistration() {
  const data = registrationPayload();
  return Boolean(
    data.patientName ||
      data.patientMrn ||
      data.patientAge > 0 ||
      data.patientSex ||
      data.studyPhysician ||
      data.studyReason ||
      Object.values(data.riskFactors).some(Boolean),
  );
}

function selectedRunAnatomyTasks() {
  if (!$("#run-enable-3d").checked) {
    return [];
  }
  const tasks = [];
  if ($("#run-task-heart").checked) tasks.push("heart");
  if ($("#run-task-coronary").checked) tasks.push("coronary_arteries");
  if ($("#run-task-chambers").checked) tasks.push("heartchambers_highres");
  return tasks;
}

function syncRunTaskControls() {
  const enabled = $("#run-enable-3d").checked;
  ["#run-task-heart", "#run-task-coronary", "#run-task-chambers"].forEach((selector) => {
    $(selector).disabled = !enabled;
  });
  if (enabled && !selectedRunAnatomyTasks().length) {
    $("#run-task-heart").checked = true;
  }
}

function clearRegistrationForm() {
  $("#patient-name").value = "";
  $("#patient-mrn").value = "";
  $("#patient-age").value = "0";
  $("#patient-sex").value = "";
  $("#study-physician").value = "";
  $("#study-reason").value = "";
  $("#risk-hypertension").checked = false;
  $("#risk-hyperlipidemia").checked = false;
  $("#risk-diabetes").checked = false;
  $("#risk-smoking").checked = false;
  $("#risk-family_hx").checked = false;
  renderRegistrationSummary();
  showToast("Registration cleared.");
}

function registrationSummaryMarkup() {
  const draft = registrationPayload();
  const riskFactors = Object.entries(draft.riskFactors)
    .filter(([, enabled]) => enabled)
    .map(([name]) => name.replace("_", " ").replace(/\b\w/g, (char) => char.toUpperCase()));

  const currentPatient = state.summary?.patient || null;
  const currentHasRegistration = Boolean(
    currentPatient &&
      (currentPatient.name || currentPatient.mrn !== "UNKNOWN" || currentPatient.age > 0 || currentPatient.sex !== "Unknown" || currentPatient.physician || currentPatient.reason || Object.values(currentPatient.riskFactors || {}).some(Boolean)),
  );
  const usingDraftOnly = hasSavedRegistration() && !currentHasRegistration;

  if (!hasSavedRegistration() && !currentHasRegistration) {
    return `
      <div class="summary-line">
        <span>Status</span>
        <strong>No Clinical Registration Saved</strong>
      </div>
      <div class="summary-line">
        <span>Workflow</span>
        <strong>Analysis Can Run Without Registration</strong>
      </div>
    `;
  }

  const active = currentHasRegistration ? currentPatient : draft;
  const activeRiskFactors = currentHasRegistration
    ? Object.entries(active.riskFactors || {})
        .filter(([, enabled]) => enabled)
        .map(([name]) => name.replace("_", " ").replace(/\b\w/g, (char) => char.toUpperCase()))
    : riskFactors;

  return `
    <div class="summary-line">
      <span>Status</span>
      <strong>${usingDraftOnly ? "Saved Draft Not Attached To Current Study" : currentHasRegistration ? "Attached To Current Study" : "Saved Draft"}</strong>
    </div>
    <div class="summary-line">
      <span>Patient</span>
      <strong>${active.name || active.patientName || "Not Provided"}</strong>
    </div>
    <div class="summary-line">
      <span>MRN / Age / Sex</span>
      <strong>${active.mrn || active.patientMrn || "Not Provided"}${(active.age || active.patientAge) ? ` / ${active.age || active.patientAge}` : ""}${(active.sex || active.patientSex) ? ` / ${active.sex || active.patientSex}` : ""}</strong>
    </div>
    <div class="summary-line">
      <span>Physician</span>
      <strong>${active.physician || active.studyPhysician || "Not Provided"}</strong>
    </div>
    <div class="summary-line">
      <span>Risk Factors</span>
      <strong>${activeRiskFactors.length ? activeRiskFactors.join(", ") : "None Reported"}</strong>
    </div>
  `;
}

function renderRegistrationSummary() {
  $("#registration-summary").innerHTML = registrationSummaryMarkup();
}

function updateEngineInfo(engine) {
  if (!engine) return;
  $("#engine-state").textContent = engine.status || "Unknown";
  $("#engine-message").textContent = engine.message || "";
  $("#overview-engine-pill").textContent = engine.status || "Unknown";
  $("#overview-device-pill").textContent = engine.device || "Unknown";
}

function renderMetrics() {
  const metrics = (state.bootstrap?.metrics || []).slice(0, 4);
  $("#metric-grid").innerHTML = metrics
    .map(
      (metric) => `
        <article class="metric-card">
          <h3>${metric.label}</h3>
          <div class="metric-value">
            <span class="metric-number">${Number(metric.value).toFixed(metric.decimals)}</span>
            <span class="metric-suffix">${metric.suffix || ""}</span>
          </div>
          <p class="info-note">${metric.description}</p>
        </article>
      `,
    )
    .join("");
}

function evidenceThumbMarkup(item) {
  return `
    <button class="evidence-thumb" data-evidence-title="${item.title}" data-evidence-image="${item.image}">
      <img src="${item.image}" alt="${item.title}" />
      <div>
        <h4>${item.title}</h4>
        <p>${item.caption}</p>
      </div>
    </button>
  `;
}

function renderEvidence() {
  const evidence = (state.bootstrap?.evidence || []).slice(0, 3);
  $("#home-evidence-grid").innerHTML = evidence.map(evidenceThumbMarkup).join("");
  $("#report-evidence-grid").innerHTML = evidence.map(evidenceThumbMarkup).join("");
  $$("[data-evidence-title]").forEach((button) => {
    button.addEventListener("click", () => {
      $("#modal-title").textContent = button.dataset.evidenceTitle;
      $("#modal-image").src = button.dataset.evidenceImage;
      openModal("image-modal");
    });
  });
}

function colorArrayToCss(color) {
  if (!Array.isArray(color)) return "#ffffff";
  const channels = color.map((channel) => {
    const scaled = channel <= 1 ? Math.round(channel * 255) : Math.round(channel);
    return Math.max(0, Math.min(255, scaled));
  });
  return `rgb(${channels.join(", ")})`;
}

function legendItemState(item) {
  if (item.group === "calcification") {
    return {
      available: Boolean(state.summary?.fastReady),
      hidden: state.mprMode === "gt",
    };
  }
  const available = new Set(state.summary?.availableAnatomy || []);
  const control = state.anatomyControls[item.group];
  const hiddenByFocus =
    item.group === "chambers" &&
    state.chamberFocus !== "all" &&
    state.chamberFocus !== item.name;
  return {
    available: available.has(item.name),
    hidden: Boolean(control && !control.visible) || hiddenByFocus,
  };
}

function legendItemMarkup(item) {
  const { available, hidden } = legendItemState(item);
  const classes = ["legend-item"];
  if (!available) classes.push("is-unavailable");
  if (available && hidden) classes.push("is-hidden");
  return `
    <div class="${classes.join(" ")}" title="${item.label}">
      <span class="legend-swatch" style="background:${colorArrayToCss(item.color)}"></span>
      <span>${item.label}</span>
    </div>
  `;
}

function renderHeartLegend() {
  const legend = state.summary?.modelLegend || state.bootstrap?.modelLegend;
  const container = $("#heart-legend");
  if (!container || !legend) return;
  container.innerHTML = `
    <div class="legend-row">
      <span class="legend-title">Calcium</span>
      <div class="legend-items">${(legend.calcium || []).map(legendItemMarkup).join("")}</div>
    </div>
    <div class="legend-row">
      <span class="legend-title">3D Anatomy</span>
      <div class="legend-items">${(legend.anatomy || []).map(legendItemMarkup).join("")}</div>
    </div>
  `;
}

function heartStatusLabel(summary = state.summary) {
  if (!summary?.fastReady) {
    return "Awaiting Analysis";
  }
  if (!summary.backgroundRequested && !summary.backgroundReady) {
    return "3D Generation Skipped";
  }
  if (summary.phase === "background" && summary.backgroundRequested) {
    return "Building Selected 3D Anatomy";
  }
  if (summary.backgroundReady && (summary.hasCoronary || summary.hasChambers)) {
    return "Advanced Anatomy Ready";
  }
  if (summary.backgroundReady) {
    return "3D Heart Ready";
  }
  if (summary.backgroundError) {
    return "Unavailable";
  }
  return "Awaiting Analysis";
}

function buildPlaceholderReport() {
  const logo = state.bootstrap?.logo || "/assets/CARDIOTECT%20LOGO.png";
  return `
    <div class="report-document">
      <div class="report-banner">
        <img src="${logo}" alt="Cardiotect logo" />
        <div>
          <p class="report-kicker">Automated Coronary Calcium Scoring</p>
          <h2>CARDIOTECT Clinical Report</h2>
        </div>
      </div>
      <div class="paper-meta-grid">
        <div><span>Patient</span><strong>Awaiting Analysis</strong></div>
        <div><span>MRN</span><strong>Not Provided</strong></div>
        <div><span>Age / Sex</span><strong>Not Provided</strong></div>
        <div><span>Status</span><strong>Ready For Live Study</strong></div>
      </div>
      <div class="paper-section">
        <strong>Workflow</strong>
        <p>Select a folder in Analysis, choose whether to add registration, review the axial output, then inspect the 3D workstation before exporting the report.</p>
      </div>
      <div class="paper-section">
        <strong>Embedded Validation Highlights</strong>
        <div class="paper-grid">
          ${(state.bootstrap?.metrics || []).slice(0, 3).map((metric) => `<div class="kv-card"><span>${metric.label}</span><strong>${Number(metric.value).toFixed(metric.decimals)}${metric.suffix || ""}</strong></div>`).join("")}
        </div>
      </div>
      <div class="paper-section">
        <strong>Note</strong>
        <p>Registration is optional. The live report will populate as soon as fast CAC inference completes.</p>
      </div>
    </div>
  `;
}

function renderReportSidebar() {
  const patient = state.summary?.patient || null;
  const scores = state.summary?.scores || null;
  const patientLabel = patient?.name || state.summary?.patientId || "No Study Loaded";
  const riskLabel = state.summary?.riskLabel || "Not Available";
  const totalScore = scores ? Number(scores.Total || 0).toFixed(1) : "--";
  const heartState = heartStatusLabel(state.summary);
  $("#report-side-summary").innerHTML = `
    <div class="summary-line">
      <span>Study</span>
      <strong>${patientLabel}</strong>
    </div>
    <div class="summary-line">
      <span>Total Score</span>
      <strong>${totalScore}</strong>
    </div>
    <div class="summary-line">
      <span>Risk Tier</span>
      <strong>${riskLabel}</strong>
    </div>
    <div class="summary-line">
      <span>3D Status</span>
      <strong>${heartState}</strong>
    </div>
  `;
}

function bootstrapSummaryLabel() {
  const patient = state.summary?.patient || null;
  return patient?.name || state.summary?.patientId || "None";
}

function renderPatientSummary() {
  const patientLabel = bootstrapSummaryLabel();
  $("#overview-study-pill").textContent = patientLabel;
  $("#heart-study-pill").textContent = patientLabel;
  renderRegistrationSummary();
  renderReportSidebar();

  if (!state.summary?.scores) {
    $("#patient-score").textContent = "--";
    $("#patient-risk").textContent = "--";
    $("#patient-territory").textContent = "--";
    $("#patient-gt").textContent = "--";
    $("#patient-note").textContent = "Select a folder and run analysis to populate the workstation.";
    $("#patient-vessels").innerHTML = "";
    return;
  }

  const scores = state.summary.scores;
  const vessels = [
    ["LM-LAD", Number(scores.LM_LAD || 0)],
    ["LCX", Number(scores.LCX || 0)],
    ["RCA", Number(scores.RCA || 0)],
  ];
  const topTerritory = [...vessels].sort((left, right) => right[1] - left[1])[0][0];
  const maxScore = Math.max(...vessels.map((item) => item[1]), 1);

  $("#patient-score").textContent = Number(scores.Total || 0).toFixed(1);
  $("#patient-risk").textContent = state.summary.riskLabel || "--";
  $("#patient-territory").textContent = topTerritory;
  $("#patient-gt").textContent = state.summary.hasGT ? "Loaded" : "Unavailable";
  $("#patient-note").textContent = state.summary.message || "";
  $("#patient-vessels").innerHTML = vessels
    .map(
      ([name, value]) => `
        <div class="mini-bar">
          <div class="line-head"><span>${name}</span><strong>${value.toFixed(1)}</strong></div>
          <div class="bar-track"><div class="bar-fill" style="width:${(value / maxScore) * 100}%"></div></div>
        </div>
      `,
    )
    .join("");
}

function renderHeartStatus() {
  if (!state.summary) {
    $("#heart-build-status").textContent = "Awaiting Analysis";
    $("#heart-progress-label").textContent = "0%";
    $("#heart-progress-bar").style.width = "0%";
    return;
  }
  $("#heart-build-status").textContent = heartStatusLabel(state.summary);
}

function renderAnatomyControlState() {
  const summary = state.summary || {};
  const availability = {
    heart: Boolean(summary.hasHeart),
    coronary: Boolean(summary.hasCoronary),
    chambers: Boolean(summary.hasChambers),
  };
  Object.entries(availability).forEach(([groupName, enabled]) => {
    const card = $(`[data-anatomy-card="${groupName}"]`);
    if (card) {
      card.classList.toggle("is-disabled", !enabled);
    }
    $$(`[id^="toggle-${groupName}-"], [id^="clip-${groupName}-"]`).forEach((control) => {
      control.disabled = !enabled;
    });
  });
  const chamberFocus = $("#chamber-focus");
  chamberFocus.disabled = !availability.chambers;
}

function updateStatus(summary) {
  const previousStudyId = state.studyId;
  const previousFastReady = state.summary?.fastReady;
  const previousBackgroundReady = state.summary?.backgroundReady;
  const previousShape = state.summary?.volumeShape?.join("x") || "";

  state.studyId = summary.studyId;
  state.summary = summary;

  $("#study-status").textContent = summary.message;
  $("#study-progress-label").textContent = `${summary.progress}%`;
  $("#study-progress-bar").style.width = `${summary.progress}%`;
  $("#heart-progress-label").textContent = `${summary.progress}%`;
  $("#heart-progress-bar").style.width = `${summary.progress}%`;
  $("#case-title").textContent = summary.patient?.name || summary.patientId || "Live Study";

  renderPatientSummary();
  renderHeartStatus();
  renderAnatomyControlState();
  renderHeartLegend();

  const shapeKey = summary.volumeShape?.join("x") || "";
  const studyChanged = previousStudyId !== summary.studyId;
  const shapeChanged = previousShape !== shapeKey;
  const fastJustReady = summary.fastReady && (!previousFastReady || studyChanged || shapeChanged);
  const backgroundJustReady = summary.backgroundReady && !previousBackgroundReady;

  if (!summary.fastReady) {
    return;
  }

  configureSliders(summary.volumeShape, studyChanged || shapeChanged);
  if (fastJustReady) {
    loadSlice(true);
    loadMprImages(true);
    refreshReport();
  }
  if (backgroundJustReady) {
    loadMprImages(true);
  }

  const anatomySignature = (summary.availableAnatomy || []).join("|");
  const meshSignature = `${summary.studyId}:${state.mprMode}:${summary.backgroundReady}:${anatomySignature}`;
  if (state.currentSection === "heart3d" && (fastJustReady || backgroundJustReady || state.lastMeshSignature !== meshSignature)) {
    loadMesh().catch((error) => console.error(error));
  } else if (fastJustReady || backgroundJustReady) {
    state.lastMeshSignature = "";
  }
}

function configureSliders(shape, resetValues = false) {
  if (!shape) return;
  const [depth, height, width] = shape;
  const nextKey = shape.join("x");
  if (state.volumeKey !== nextKey || resetValues) {
    state.volumeKey = nextKey;
    state.sliceIndex = Math.floor(depth / 2);
    state.mprIndices = {
      axial: Math.floor(depth / 2),
      coronal: Math.floor(height / 2),
      sagittal: Math.floor(width / 2),
    };
    resetAllMprViewports();
  }
  updateMprFrameRatios();

  $("#slice-slider").max = Math.max(0, depth - 1);
  $("#slice-slider").value = state.sliceIndex;
  updateSliceReadout();

  Object.entries(sliderGroups).forEach(([orientation, selectors]) => {
    const max = orientation === "axial" ? depth - 1 : orientation === "coronal" ? height - 1 : width - 1;
    selectors.forEach((selector) => {
      const input = $(selector);
      input.max = Math.max(0, max);
      input.value = state.mprIndices[orientation];
    });
  });
}

function updateSliceReadout() {
  const total = state.summary?.volumeShape?.[0] || 0;
  $("#slice-readout").textContent = `${Number(state.sliceIndex) + 1} / ${total}`;
}

function setSliderGroupValue(orientation, value, source = null) {
  state.syncSliders = true;
  sliderGroups[orientation].forEach((selector) => {
    const input = $(selector);
    if (input !== source) {
      input.value = value;
    }
  });
  state.syncSliders = false;
}

function clampMprViewport(orientation) {
  const frame = $(`[data-mpr-frame="${orientation}"]`);
  const viewport = state.mprViewport[orientation];
  if (!frame || !viewport) return;
  if (viewport.zoom <= 1.001) {
    viewport.panX = 0;
    viewport.panY = 0;
    return;
  }
  const maxPanX = (frame.clientWidth * (viewport.zoom - 1)) / 2;
  const maxPanY = (frame.clientHeight * (viewport.zoom - 1)) / 2;
  viewport.panX = Math.max(-maxPanX, Math.min(maxPanX, viewport.panX));
  viewport.panY = Math.max(-maxPanY, Math.min(maxPanY, viewport.panY));
}

function applyMprViewportTransform(orientation) {
  const img = $(`#mpr-${orientation}`);
  const frame = $(`[data-mpr-frame="${orientation}"]`);
  const viewport = state.mprViewport[orientation];
  if (!img || !frame || !viewport) return;
  clampMprViewport(orientation);
  img.style.transform =
    viewport.zoom > 1.001
      ? `translate(${viewport.panX}px, ${viewport.panY}px) scale(${viewport.zoom})`
      : "";
  frame.classList.toggle("is-dragging", viewport.active);
  frame.style.cursor = viewport.zoom > 1.001 ? (viewport.active ? "grabbing" : "grab") : "default";
}

function resetMprViewport(orientation) {
  const viewport = state.mprViewport[orientation];
  if (!viewport) return;
  viewport.zoom = 1;
  viewport.panX = 0;
  viewport.panY = 0;
  viewport.active = false;
  viewport.pointerId = null;
  applyMprViewportTransform(orientation);
}

function resetAllMprViewports() {
  ["axial", "coronal", "sagittal"].forEach((orientation) => resetMprViewport(orientation));
}

function activeAnatomyNames() {
  const available = new Set(state.summary?.availableAnatomy || []);
  const names = [];
  Object.entries(state.anatomyControls).forEach(([groupName, control]) => {
    if (!control.visible) return;
    if (groupName === "chambers" && state.chamberFocus !== "all" && available.has(state.chamberFocus)) {
      names.push(state.chamberFocus);
      return;
    }
    names.push(...(anatomyGroups[groupName] || []).filter((name) => available.has(name)));
  });
  return [...new Set(names)];
}

function sliceUrl(index = state.sliceIndex) {
  return `/api/studies/${state.studyId}/slice?index=${index}&overlay=${state.overlayMode}&window=1500&level=300`;
}

function mprUrl(orientation, index) {
  const anatomy = encodeURIComponent(activeAnatomyNames().join(","));
  return `/api/studies/${state.studyId}/mpr?orientation=${orientation}&index=${index}&mode=${state.mprMode}&window=1500&level=300&anatomy=${anatomy}`;
}

function cacheBust(url, revision) {
  const joiner = url.includes("?") ? "&" : "?";
  return `${url}${joiner}v=${revision}`;
}

function replaceImageUrl(key, img, blob) {
  if (!img || !blob) return null;
  if (state.imageUrls[key] && String(state.imageUrls[key]).startsWith("blob:")) {
    URL.revokeObjectURL(state.imageUrls[key]);
  }
  if (typeof blob === "string") {
    state.imageUrls[key] = null;
    img.src = blob;
    return blob;
  }
  const objectUrl = URL.createObjectURL(blob);
  state.imageUrls[key] = objectUrl;
  img.src = objectUrl;
  return objectUrl;
}

function createDrawableFromBlob(blob) {
  if (typeof createImageBitmap === "function") {
    return createImageBitmap(blob);
  }
  return new Promise((resolve, reject) => {
    const objectUrl = URL.createObjectURL(blob);
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(objectUrl);
      resolve(image);
    };
    image.onerror = (error) => {
      URL.revokeObjectURL(objectUrl);
      reject(error);
    };
    image.src = objectUrl;
  });
}

function updateMprFrameRatios() {
  const shape = state.summary?.volumeShape;
  const spacing = state.summary?.spacing;
  if (!shape || !spacing) return;
  const [depth, height, width] = shape;
  const [sz, sy, sx] = spacing;
  const ratios = {
    axial: (width * sx) / Math.max(height * sy, 1e-6),
    coronal: (width * sx) / Math.max(depth * sz, 1e-6),
    sagittal: (height * sy) / Math.max(depth * sz, 1e-6),
  };
  Object.entries(ratios).forEach(([orientation, ratio]) => {
    const frame = $(`[data-mpr-frame="${orientation}"]`);
    if (frame) {
      frame.style.setProperty("--mpr-ratio", String(Math.max(0.5, ratio)));
    }
  });
}

async function requestSliceImage(url) {
  const img = $("#ct-viewer-image");
  if (!img) return;
  const revision = ++state.sliceRevision;
  const sourceUrl = cacheBust(url, revision);
  try {
    const response = await fetch(sourceUrl, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Slice request failed: ${response.status}`);
    }
    const blob = await response.blob();
    if (revision !== state.sliceRevision) return;
    replaceImageUrl("slice", img, blob);
  } catch (error) {
    console.error(error);
  }
}

function updatePlanePosition(orientation) {
  if (!state.summary?.bounds || !state.summary?.spacing) return;
  const bounds = state.summary.bounds;
  const [sz, sy, sx] = state.summary.spacing;
  if (orientation === "axial" && threeState.planes.axial) {
    threeState.planes.axial.position.set(bounds.width / 2, bounds.height / 2, state.mprIndices.axial * sz);
  }
  if (orientation === "coronal" && threeState.planes.coronal) {
    threeState.planes.coronal.position.set(bounds.width / 2, state.mprIndices.coronal * sy, bounds.depth / 2);
  }
  if (orientation === "sagittal" && threeState.planes.sagittal) {
    threeState.planes.sagittal.position.set(state.mprIndices.sagittal * sx, bounds.height / 2, bounds.depth / 2);
  }
}

async function requestMprImage(orientation, url) {
  const img = $(`#mpr-${orientation}`);
  if (!img) return;
  const revision = ++state.mprRevisions[orientation];
  const sourceUrl = cacheBust(url, revision);
  img.dataset.revision = String(revision);
  try {
    const response = await fetch(sourceUrl, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`MPR request failed: ${response.status}`);
    }
    const blob = await response.blob();
    if (revision !== state.mprRevisions[orientation]) return;
    const bitmap = await createDrawableFromBlob(blob);
    if (revision !== state.mprRevisions[orientation]) {
      bitmap.close?.();
      return;
    }
    replaceImageUrl(orientation, img, blob);
    resetMprViewport(orientation);
    const plane = threeState.planes[orientation];
    if (!plane?.userData?.canvas || !plane?.material?.map) {
      bitmap.close?.();
      return;
    }
    const canvas = plane.userData.canvas;
    const ctx = plane.userData.ctx;
    if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
    }
    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = 1;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    bitmap.close?.();
    plane.material.map.needsUpdate = true;
    plane.material.needsUpdate = true;
    updatePlanePosition(orientation);
    updateIntersectionGuides();
    renderThreeFrame();
  } catch (error) {
    console.error(error);
  }
}

function loadSlice(force = false) {
  if (!state.studyId || !state.summary?.fastReady) return;
  if (!force && state.currentSection !== "analysis" && state.currentSection !== "heart3d") return;
  requestSliceImage(sliceUrl());
  prefetchSliceNeighbors();
}

function prefetchSliceNeighbors() {
  if (!state.summary?.volumeShape) return;
  const maxIndex = state.summary.volumeShape[0] - 1;
  [-2, -1, 1, 2].forEach((delta) => {
    const next = state.sliceIndex + delta;
    if (next < 0 || next > maxIndex) return;
    const img = new Image();
    img.src = sliceUrl(next);
  });
}

function reapplyPlaneTextures() {
  requestMprImage("axial", mprUrl("axial", state.mprIndices.axial));
  requestMprImage("coronal", mprUrl("coronal", state.mprIndices.coronal));
  requestMprImage("sagittal", mprUrl("sagittal", state.mprIndices.sagittal));
}

function loadMprImages(force = false) {
  if (!state.studyId || !state.summary?.fastReady) return;
  if (!force && state.currentSection !== "heart3d") return;
  updateThreeBounds(state.summary.bounds);
  const axial = mprUrl("axial", state.mprIndices.axial);
  const coronal = mprUrl("coronal", state.mprIndices.coronal);
  const sagittal = mprUrl("sagittal", state.mprIndices.sagittal);
  requestMprImage("axial", axial);
  requestMprImage("coronal", coronal);
  requestMprImage("sagittal", sagittal);
}

function scheduleSliceLoad() {
  clearTimeout(state.sliceTimer);
  state.sliceTimer = setTimeout(() => loadSlice(true), 24);
}

function scheduleMprLoad() {
  clearTimeout(state.mprTimer);
  state.mprTimer = setTimeout(() => loadMprImages(true), 28);
}

async function browseFolder() {
  try {
    const response = await postJSON("/api/dialog/dicom-folder", {});
    if (!response.selectedPath) return;
    state.selectedPath = response.selectedPath;
    $("#selected-path").textContent = response.selectedPath;
    showToast("Patient folder selected.");
  } catch (error) {
    showToast(error.message);
  }
}

function promptRunAnalysis() {
  if (!state.selectedPath) {
    showToast("Select a patient folder first.");
    return;
  }
  const savedButton = $("#run-with-saved-registration-btn");
  const hasDraft = hasSavedRegistration();
  savedButton.disabled = !hasDraft;
  savedButton.textContent = hasDraft ? "Use Saved Registration" : "No Saved Registration";
  syncRunTaskControls();
  openModal("run-modal");
}

async function executeStudy(includeRegistration) {
  if (!state.selectedPath) {
    showToast("Select a patient folder first.");
    return;
  }
  const requestedAnatomyTasks = selectedRunAnatomyTasks();
  const payload = {
    selectedPath: state.selectedPath,
    generate3dAnatomy: requestedAnatomyTasks.length > 0,
    requestedAnatomyTasks,
    ...(includeRegistration ? registrationPayload() : blankRegistrationPayload()),
  };
  try {
    const summary = await postJSON("/api/studies", payload);
    updateStatus(summary);
    navigateTo("analysis");
    startPolling();
  } catch (error) {
    showToast(error.message);
  }
}

function startPolling() {
  clearInterval(state.pollTimer);
  state.pollTimer = setInterval(async () => {
    if (!state.studyId) return;
    try {
      const summary = await fetchJSON(`/api/studies/${state.studyId}/status`);
      updateStatus(summary);
      if (summary.status === "ready" || summary.status === "error") {
        clearInterval(state.pollTimer);
      }
    } catch {
      clearInterval(state.pollTimer);
    }
  }, 900);
}

async function refreshReport() {
  if (!state.studyId || !state.summary?.fastReady) {
    $("#report-paper").innerHTML = buildPlaceholderReport();
    renderReportSidebar();
    return;
  }
  try {
    const payload = await fetchJSON(`/api/studies/${state.studyId}/report`);
    $("#report-paper").innerHTML = payload.html;
  } catch (error) {
    $("#report-paper").innerHTML = `<div class="report-document"><div class="paper-section"><strong>Report Unavailable</strong><p>${error.message}</p></div></div>`;
  }
}

async function exportReport() {
  if (!state.studyId || !state.summary?.fastReady) {
    showToast("Run analysis first.");
    return;
  }
  const response = await fetch(`/api/studies/${state.studyId}/report/export`, { method: "POST" });
  if (!response.ok) {
    showToast("Report export failed.");
    return;
  }
  const blob = await response.blob();
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `Cardiotect_Report_${state.studyId}.pdf`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  showToast("PDF export started.");
}

function bindOverlayButtons() {
  $$(".overlay-btn").forEach((button) => {
    button.addEventListener("click", () => {
      state.overlayMode = button.dataset.overlay;
      $$(".overlay-btn").forEach((item) => item.classList.toggle("active", item === button));
      loadSlice(true);
    });
  });
}

function bindModeButtons() {
  $$(".mpr-mode-btn").forEach((button) => {
    button.addEventListener("click", () => {
      state.mprMode = button.dataset.mode;
      $$(".mpr-mode-btn").forEach((item) => item.classList.toggle("active", item === button));
      renderHeartLegend();
      loadMprImages(true);
      loadMesh().catch((error) => console.error(error));
    });
  });
}

function bindMprSlider(orientation) {
  sliderGroups[orientation].forEach((selector) => {
    $(selector).addEventListener("input", (event) => {
      if (state.syncSliders) return;
      const value = Number(event.target.value);
      state.mprIndices[orientation] = value;
      setSliderGroupValue(orientation, value, event.target);
      scheduleMprLoad();
    });
  });
}

function bindMprViewport(orientation) {
  const frame = $(`[data-mpr-frame="${orientation}"]`);
  if (!frame) return;
  const viewport = state.mprViewport[orientation];

  frame.addEventListener(
    "wheel",
    (event) => {
      if (!event.ctrlKey && !event.metaKey) {
        return;
      }
      event.preventDefault();
      const nextZoom = viewport.zoom * (event.deltaY > 0 ? 0.9 : 1.12);
      viewport.zoom = Math.max(1, Math.min(5, nextZoom));
      if (viewport.zoom <= 1.001) {
        viewport.zoom = 1;
        viewport.panX = 0;
        viewport.panY = 0;
      }
      applyMprViewportTransform(orientation);
    },
    { passive: false },
  );

  frame.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    if (viewport.zoom <= 1.001) return;
    viewport.active = true;
    viewport.pointerId = event.pointerId;
    viewport.lastX = event.clientX;
    viewport.lastY = event.clientY;
    frame.setPointerCapture(event.pointerId);
    applyMprViewportTransform(orientation);
  });

  frame.addEventListener("pointermove", (event) => {
    if (!viewport.active || viewport.pointerId !== event.pointerId) return;
    viewport.panX += event.clientX - viewport.lastX;
    viewport.panY += event.clientY - viewport.lastY;
    viewport.lastX = event.clientX;
    viewport.lastY = event.clientY;
    applyMprViewportTransform(orientation);
  });

  const stopPan = (event) => {
    if (viewport.pointerId !== event.pointerId) return;
    viewport.active = false;
    viewport.pointerId = null;
    frame.releasePointerCapture(event.pointerId);
    applyMprViewportTransform(orientation);
  };
  frame.addEventListener("pointerup", stopPan);
  frame.addEventListener("pointercancel", stopPan);
  frame.addEventListener("dblclick", () => resetMprViewport(orientation));
}

function bindAnatomyControls() {
  ["heart", "coronary", "chambers"].forEach((groupName) => {
    $(`#toggle-${groupName}-model`).addEventListener("change", (event) => {
      state.anatomyControls[groupName].visible = event.target.checked;
      applyMeshPresentation();
      renderHeartLegend();
      loadMprImages(true);
    });
    ["axial", "coronal", "sagittal"].forEach((axis) => {
      $(`#clip-${groupName}-${axis}`).addEventListener("input", (event) => {
        state.anatomyControls[groupName].clip[axis] = Number(event.target.value);
        applyMeshPresentation();
      });
    });
  });
  $("#chamber-focus").addEventListener("change", (event) => {
    state.chamberFocus = event.target.value;
    applyMeshPresentation();
    renderHeartLegend();
    loadMprImages(true);
  });
}

function initControls() {
  $("#browse-folder-btn").addEventListener("click", browseFolder);
  $("#run-study-btn").addEventListener("click", promptRunAnalysis);
  $("#open-registration-btn").addEventListener("click", () => openModal("registration-modal"));
  $("#edit-registration-btn").addEventListener("click", () => openModal("registration-modal"));
  $("#run-enable-3d").addEventListener("change", () => {
    syncRunTaskControls();
  });
  ["#run-task-heart", "#run-task-coronary", "#run-task-chambers"].forEach((selector) => {
    $(selector).addEventListener("change", () => {
      syncRunTaskControls();
    });
  });
  $("#run-without-registration-btn").addEventListener("click", () => {
    closeModal("run-modal");
    executeStudy(false);
  });
  $("#run-with-saved-registration-btn").addEventListener("click", () => {
    closeModal("run-modal");
    if (!hasSavedRegistration()) {
      showToast("No saved registration is available.");
      return;
    }
    executeStudy(true);
  });
  $("#open-registration-from-run-btn").addEventListener("click", () => {
    closeModal("run-modal");
    openModal("registration-modal");
  });
  $("#save-registration-btn").addEventListener("click", () => {
    closeModal("registration-modal");
    renderRegistrationSummary();
    showToast("Registration saved.");
  });
  $("#save-and-run-btn").addEventListener("click", () => {
    closeModal("registration-modal");
    renderRegistrationSummary();
    executeStudy(true);
  });
  $("#clear-registration-btn").addEventListener("click", clearRegistrationForm);
  $("#slice-slider").addEventListener("input", (event) => {
    state.sliceIndex = Number(event.target.value);
    updateSliceReadout();
    scheduleSliceLoad();
  });
  bindOverlayButtons();
  bindModeButtons();
  bindMprSlider("axial");
  bindMprSlider("coronal");
  bindMprSlider("sagittal");
  bindMprViewport("axial");
  bindMprViewport("coronal");
  bindMprViewport("sagittal");
  $$("[data-mpr-reset]").forEach((button) => {
    button.addEventListener("click", () => resetMprViewport(button.dataset.mprReset));
  });
  bindAnatomyControls();
  $("#heart-opacity").addEventListener("input", (event) => {
    applyMeshPresentation();
  });
  $("#reset-camera-btn").addEventListener("click", resetCamera);
  $("#toggle-ortho-btn").addEventListener("click", toggleOrtho);
  $("#export-report-btn").addEventListener("click", exportReport);
  $$(".panel-toggle").forEach((button) => button.addEventListener("click", () => togglePanel(button.dataset.panelToggle)));
  syncRunTaskControls();
}

function togglePanel(panelName) {
  const grid = $("#workstation-grid");
  if (state.maxPanel === panelName) {
    state.maxPanel = null;
    grid.classList.remove("maximized");
    $$(".work-panel").forEach((panel) => panel.classList.remove("is-maximized"));
  } else {
    state.maxPanel = panelName;
    grid.classList.add("maximized");
    $$(".work-panel").forEach((panel) => panel.classList.toggle("is-maximized", panel.dataset.panel === panelName));
  }
  if (panelName === "axial" || panelName === "coronal" || panelName === "sagittal") {
    resetMprViewport(panelName);
  }
  requestAnimationFrame(() => {
    applyMprViewportTransform("axial");
    applyMprViewportTransform("coronal");
    applyMprViewportTransform("sagittal");
    resizeThreeRenderer();
    renderThreeFrame();
  });
}

function disposeMaterial(material) {
  if (Array.isArray(material)) {
    material.forEach(disposeMaterial);
    return;
  }
  if (material?.map) material.map.dispose();
  if (material) material.dispose();
}

function clearThreeGroup(group) {
  while (group.children.length) {
    const child = group.children[0];
    if (!child) continue;
    group.remove(child);
    if (child.children?.length) {
      while (child.children.length) {
        const nested = child.children[0];
        child.remove(nested);
        if (nested.geometry) nested.geometry.dispose();
        if (nested.material) disposeMaterial(nested.material);
      }
    }
    if (child.geometry) child.geometry.dispose();
    if (child.material) disposeMaterial(child.material);
  }
}

function boundsKey(bounds) {
  if (!bounds) return "";
  return [bounds.width, bounds.height, bounds.depth].map((value) => Number(value).toFixed(3)).join(":");
}

function setPlaneBasis(mesh, right, up, normal) {
  const basis = new THREE.Matrix4().makeBasis(right, up, normal);
  mesh.setRotationFromMatrix(basis);
}

function createPlane(name, width, height, basis) {
  const geometry = new THREE.PlaneGeometry(width, height);
  const canvas = document.createElement("canvas");
  canvas.width = 2;
  canvas.height = 2;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#060202";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  const material = new THREE.MeshBasicMaterial({
    color: 0xffffff,
    map: texture,
    transparent: false,
    opacity: 1,
    side: THREE.DoubleSide,
    depthWrite: true,
    depthTest: true,
    toneMapped: false,
  });
  const mesh = new THREE.Mesh(geometry, material);
  setPlaneBasis(mesh, basis.right, basis.up, basis.normal);
  mesh.renderOrder = 10;
  mesh.userData = { name, texture, canvas, ctx };
  threeState.planeGroup.add(mesh);
  threeState.planes[name] = mesh;
}

function ensurePlanes(bounds) {
  const nextKey = boundsKey(bounds);
  if (!bounds || nextKey === threeState.boundsKey) {
    return;
  }
  clearThreeGroup(threeState.planeGroup);
  threeState.planes = {};
  createPlane("axial", bounds.width, bounds.height, {
    right: new THREE.Vector3(1, 0, 0),
    up: new THREE.Vector3(0, -1, 0),
    normal: new THREE.Vector3(0, 0, -1),
  });
  createPlane("coronal", bounds.width, bounds.depth, {
    right: new THREE.Vector3(1, 0, 0),
    up: new THREE.Vector3(0, 0, 1),
    normal: new THREE.Vector3(0, -1, 0),
  });
  createPlane("sagittal", bounds.height, bounds.depth, {
    right: new THREE.Vector3(0, 1, 0),
    up: new THREE.Vector3(0, 0, 1),
    normal: new THREE.Vector3(1, 0, 0),
  });
  threeState.boundsKey = nextKey;
}

function applyObjectRotation() {
  if (!threeState.pivotGroup) return;
  threeState.pivotGroup.rotation.order = "YXZ";
  threeState.pivotGroup.rotation.x = threeState.objectRotation.x;
  threeState.pivotGroup.rotation.y = threeState.objectRotation.y;
  threeState.pivotGroup.rotation.z = threeState.objectRotation.z;
  threeState.pivotGroup.updateMatrixWorld(true);
}

function syncCameraPose() {
  if (!threeState.camera || !threeState.orthoCamera) return;
  const offset = threeState.baseCameraDirection.clone().multiplyScalar(threeState.cameraDistance);
  threeState.camera.position.copy(offset);
  threeState.camera.lookAt(0, 0, 0);
  threeState.orthoCamera.position.copy(offset);
  threeState.orthoCamera.lookAt(0, 0, 0);
}

function bindThreeInteraction(canvas) {
  canvas.addEventListener("contextmenu", (event) => event.preventDefault());
  canvas.addEventListener("pointerdown", (event) => {
    threeState.dragState.active = true;
    threeState.dragState.pointerId = event.pointerId;
    threeState.dragState.button = event.button;
    threeState.dragState.x = event.clientX;
    threeState.dragState.y = event.clientY;
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!threeState.dragState.active || threeState.dragState.pointerId !== event.pointerId) return;
    const dx = event.clientX - threeState.dragState.x;
    const dy = event.clientY - threeState.dragState.y;
    threeState.dragState.x = event.clientX;
    threeState.dragState.y = event.clientY;
    if (threeState.dragState.button === 2) {
      threeState.objectRotation.z += dx * 0.0085;
    } else {
      threeState.objectRotation.y += dx * 0.0085;
      threeState.objectRotation.x += dy * 0.0085;
    }
    applyObjectRotation();
    applyMeshPresentation();
  });
  const stopDrag = (event) => {
    if (threeState.dragState.pointerId !== event.pointerId) return;
    threeState.dragState.active = false;
    threeState.dragState.pointerId = null;
    canvas.releasePointerCapture(event.pointerId);
  };
  canvas.addEventListener("pointerup", stopDrag);
  canvas.addEventListener("pointercancel", stopDrag);
  canvas.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      if (threeState.isOrtho) {
        const nextZoom = Math.max(0.45, Math.min(3.8, threeState.orthoCamera.zoom * (event.deltaY > 0 ? 0.92 : 1.08)));
        threeState.orthoCamera.zoom = nextZoom;
        threeState.orthoCamera.updateProjectionMatrix();
      } else {
        threeState.cameraDistance = Math.max(140, Math.min(2400, threeState.cameraDistance * (event.deltaY > 0 ? 1.08 : 0.92)));
        syncCameraPose();
      }
      renderThreeFrame();
    },
    { passive: false },
  );
}

function fitCameraToBounds(bounds) {
  if (!bounds || !threeState.camera || !threeState.orthoCamera) return;
  const maxDim = Math.max(bounds.width, bounds.height, bounds.depth);
  const distance = maxDim * 1.85;
  threeState.cameraDistance = distance;
  threeState.objectRotation = { x: 0, y: 0, z: 0 };
  applyObjectRotation();
  syncCameraPose();
  threeState.camera.near = 0.1;
  threeState.camera.far = distance * 12;
  threeState.camera.updateProjectionMatrix();
  threeState.orthoCamera.near = 0.1;
  threeState.orthoCamera.far = distance * 12;
  threeState.orthoCamera.zoom = 1;
  threeState.orthoCamera.updateProjectionMatrix();
}

function updateThreeBounds(bounds) {
  if (!bounds || !threeState.rootGroup) return;
  threeState.rootGroup.position.set(bounds.x, bounds.y, bounds.z);
  threeState.rootGroup.updateMatrixWorld(true);
  const previousKey = threeState.boundsKey;
  ensurePlanes(bounds);
  if (previousKey !== threeState.boundsKey) {
    fitCameraToBounds(bounds);
  }
}

function bindMprImageTextures() {
  ["axial", "coronal", "sagittal"].forEach((orientation) => {
    const img = $(`#mpr-${orientation}`);
    if (!img) return;
    img.addEventListener("load", () => {
      applyMprViewportTransform(orientation);
      renderThreeFrame();
    });
  });
}

function updatePlanePositions() {
  if (!state.summary?.bounds || !state.summary?.spacing) return;
  updatePlanePosition("axial");
  updatePlanePosition("coronal");
  updatePlanePosition("sagittal");
  updateIntersectionGuides();
  renderThreeFrame();
}

function buildClipPlanes(groupName) {
  if (!state.summary?.bounds || !state.anatomyControls[groupName]) return null;
  const bounds = state.summary.bounds;
  const clip = state.anatomyControls[groupName].clip;
  const planes = [];
  if (clip.sagittal < 100) {
    planes.push(new THREE.Plane(new THREE.Vector3(-1, 0, 0), bounds.x + bounds.width * (clip.sagittal / 100)));
  }
  if (clip.coronal < 100) {
    planes.push(new THREE.Plane(new THREE.Vector3(0, -1, 0), bounds.y + bounds.height * (clip.coronal / 100)));
  }
  if (clip.axial < 100) {
    planes.push(new THREE.Plane(new THREE.Vector3(0, 0, -1), bounds.z + bounds.depth * (clip.axial / 100)));
  }
  if (!planes.length) return null;
  const pivotMatrix = threeState.pivotGroup?.matrixWorld;
  return pivotMatrix ? planes.map((plane) => plane.clone().applyMatrix4(pivotMatrix)) : planes;
}

function actorIsVisible(entry) {
  if (!entry.group || !state.anatomyControls[entry.group]) {
    return true;
  }
  if (!state.anatomyControls[entry.group].visible) {
    return false;
  }
  if (entry.group === "chambers" && state.chamberFocus !== "all" && entry.name !== state.chamberFocus) {
    return false;
  }
  return true;
}

function applyMeshPresentation() {
  if (!threeState.actorEntries.length) {
    renderThreeFrame();
    return;
  }
  const heartOpacity = Number($("#heart-opacity").value) / 100;
  threeState.actorEntries.forEach((entry) => {
    const visible = actorIsVisible(entry);
    const clippingPlanes = buildClipPlanes(entry.group);
    entry.objects.forEach((object) => {
      object.visible = visible;
    });
    entry.materials.forEach((material) => {
      if (!material) return;
      if (material.userData?.heartRole === "shell") {
        material.opacity = Math.max(heartOpacity * 0.38, 0.18);
      } else if (material.userData?.heartRole === "surface") {
        material.opacity = heartOpacity;
      } else {
        material.opacity = entry.baseOpacity ?? material.opacity;
      }
      material.clippingPlanes = clippingPlanes;
      material.clipIntersection = false;
      material.needsUpdate = true;
    });
  });
  renderThreeFrame();
}

function renderMeshActors(payload) {
  clearThreeGroup(threeState.meshGroup);
  threeState.materials = {};
  threeState.actorEntries = [];
  const registerMaterial = (name, material) => {
    if (!threeState.materials[name]) {
      threeState.materials[name] = [];
    }
    threeState.materials[name].push(material);
  };
  const registerEntry = (entry) => {
    threeState.actorEntries.push(entry);
  };
  (payload.actors || []).forEach((actor) => {
    const isHeart = actor.name === "heart";
    const actorGroup = actor.group || "calcification";
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(actor.vertices.flat(), 3));
    geometry.setAttribute("normal", new THREE.Float32BufferAttribute(actor.normals.flat(), 3));
    geometry.setIndex(actor.indices.flat());
    geometry.computeBoundingSphere();
    if (isHeart) {
      const heartOpacity = Number($("#heart-opacity").value) / 100;
      const heartColor = new THREE.Color(...(actor.color || [0.88, 0.16, 0.28]));
      const shellMaterial = new THREE.MeshPhongMaterial({
        color: heartColor.clone().multiplyScalar(0.42),
        transparent: true,
        opacity: Math.max(heartOpacity * 0.45, 0.24),
        side: THREE.BackSide,
        shininess: 24,
        depthTest: true,
        depthWrite: false,
      });
      shellMaterial.userData = { name: actor.name, heartRole: "shell" };
      const shell = new THREE.Mesh(geometry.clone(), shellMaterial);
      shell.scale.setScalar(1.012);
      shell.renderOrder = 5;
      threeState.meshGroup.add(shell);
      registerMaterial(actor.name, shellMaterial);

      const material = new THREE.MeshPhongMaterial({
        color: heartColor,
        emissive: heartColor.clone().multiplyScalar(0.16),
        specular: new THREE.Color(0xffffff),
        shininess: 48,
        transparent: true,
        opacity: heartOpacity,
        side: THREE.DoubleSide,
        depthTest: true,
        depthWrite: true,
      });
      material.userData = { name: actor.name, heartRole: "surface" };
      const mesh = new THREE.Mesh(geometry, material);
      mesh.userData = { name: actor.name, group: actorGroup };
      mesh.renderOrder = 6;
      threeState.meshGroup.add(mesh);
      registerMaterial(actor.name, material);
      registerEntry({
        name: actor.name,
        group: actorGroup,
        objects: [shell, mesh],
        materials: [shellMaterial, material],
        baseOpacity: heartOpacity,
      });
      return;
    }
    const material = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(...actor.color),
      transparent: true,
      opacity: actor.opacity,
      roughness: 0.32,
      metalness: 0.18,
      side: THREE.DoubleSide,
      depthWrite: true,
      depthTest: true,
    });
    material.userData = { name: actor.name };
    const mesh = new THREE.Mesh(geometry, material);
    mesh.userData = { name: actor.name, group: actorGroup };
    mesh.renderOrder = 4;
    threeState.meshGroup.add(mesh);
    registerMaterial(actor.name, material);
    registerEntry({
      name: actor.name,
      group: actorGroup,
      objects: [mesh],
      materials: [material],
      baseOpacity: actor.opacity,
    });
  });
  applyMeshPresentation();
}

function createGuideLine(points, color) {
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({
    color,
    transparent: true,
    opacity: 0.9,
    depthTest: false,
  });
  const line = new THREE.Line(geometry, material);
  line.renderOrder = 9;
  return line;
}

function updateIntersectionGuides() {
  if (!threeState.guideGroup || !state.summary?.bounds || !state.summary?.spacing) return;
  clearThreeGroup(threeState.guideGroup);
  const bounds = state.summary.bounds;
  const [sz, sy, sx] = state.summary.spacing;
  const x = state.mprIndices.sagittal * sx;
  const y = state.mprIndices.coronal * sy;
  const z = state.mprIndices.axial * sz;
  threeState.guideGroup.add(
    createGuideLine([new THREE.Vector3(0, y, z), new THREE.Vector3(bounds.width, y, z)], slicerPlaneColors.axial),
  );
  threeState.guideGroup.add(
    createGuideLine([new THREE.Vector3(x, 0, z), new THREE.Vector3(x, bounds.height, z)], slicerPlaneColors.coronal),
  );
  threeState.guideGroup.add(
    createGuideLine([new THREE.Vector3(x, y, 0), new THREE.Vector3(x, y, bounds.depth)], slicerPlaneColors.sagittal),
  );
}

async function loadMesh() {
  if (!state.studyId || !state.summary?.fastReady) return;
  const anatomySignature = (state.summary.availableAnatomy || []).join("|");
  const signature = `${state.studyId}:${state.mprMode}:${state.summary.backgroundReady}:${anatomySignature}`;
  if (state.lastMeshSignature === signature) {
    applyMeshPresentation();
    return;
  }
  const payload = await fetchJSON(`/api/studies/${state.studyId}/mesh?mode=${state.mprMode}`);
  updateThreeBounds(payload.bounds);
  renderMeshActors(payload);
  reapplyPlaneTextures();
  updatePlanePositions();
  state.lastMeshSignature = signature;
  renderThreeFrame();
}

function resizeThreeRenderer() {
  if (!threeState.renderer) return;
  const canvas = $("#three-canvas");
  const width = Math.max(320, Math.floor(canvas.clientWidth || 640));
  const height = Math.max(240, Math.floor(canvas.clientHeight || 420));
  threeState.renderer.setSize(width, height, false);
  threeState.camera.aspect = width / height;
  threeState.camera.updateProjectionMatrix();
  const orthoScale = Math.max(220, threeState.cameraDistance * 0.42);
  threeState.orthoCamera.left = -orthoScale * (width / height);
  threeState.orthoCamera.right = orthoScale * (width / height);
  threeState.orthoCamera.top = orthoScale;
  threeState.orthoCamera.bottom = -orthoScale;
  threeState.orthoCamera.updateProjectionMatrix();
}

function renderThreeFrame() {
  if (!threeState.renderer || !threeState.scene || !threeState.activeCamera) return;
  if (state.currentSection !== "heart3d") return;
  threeState.renderer.render(threeState.scene, threeState.activeCamera);
}

function resetCamera() {
  if (!state.summary?.bounds) return;
  fitCameraToBounds(state.summary.bounds);
  threeState.activeCamera = threeState.isOrtho ? threeState.orthoCamera : threeState.camera;
  applyMeshPresentation();
}

function toggleOrtho() {
  threeState.isOrtho = !threeState.isOrtho;
  threeState.activeCamera = threeState.isOrtho ? threeState.orthoCamera : threeState.camera;
  $("#toggle-ortho-btn").classList.toggle("active", threeState.isOrtho);
  renderThreeFrame();
}

function initThreeScene() {
  const canvas = $("#three-canvas");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
  renderer.setClearColor(0x0b0202, 1);

  const scene = new THREE.Scene();
  scene.add(new THREE.AmbientLight(0xffffff, 0.72));
  const key = new THREE.DirectionalLight(0xf6d6db, 1.05);
  key.position.set(220, 180, 240);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xff7b8c, 0.64);
  fill.position.set(-180, -140, 160);
  scene.add(fill);
  const rim = new THREE.DirectionalLight(0xf5c25c, 0.34);
  rim.position.set(0, 220, -160);
  scene.add(rim);

  renderer.localClippingEnabled = true;

  const pivotGroup = new THREE.Group();
  const rootGroup = new THREE.Group();
  const meshGroup = new THREE.Group();
  const planeGroup = new THREE.Group();
  const guideGroup = new THREE.Group();
  rootGroup.add(meshGroup);
  rootGroup.add(planeGroup);
  rootGroup.add(guideGroup);
  pivotGroup.add(rootGroup);
  scene.add(pivotGroup);

  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 5000);
  const orthoCamera = new THREE.OrthographicCamera(-400, 400, 400, -400, 0.1, 5000);
  bindThreeInteraction(canvas);

  threeState.renderer = renderer;
  threeState.scene = scene;
  threeState.camera = camera;
  threeState.orthoCamera = orthoCamera;
  threeState.activeCamera = camera;
  threeState.controls = null;
  threeState.pivotGroup = pivotGroup;
  threeState.rootGroup = rootGroup;
  threeState.meshGroup = meshGroup;
  threeState.planeGroup = planeGroup;
  threeState.guideGroup = guideGroup;

  bindMprImageTextures();
  resizeThreeRenderer();
  fitCameraToBounds({ width: 220, height: 220, depth: 160, x: -110, y: -110, z: -80 });

  const animate = (timestamp) => {
    requestAnimationFrame(animate);
    if (state.currentSection !== "heart3d") return;
    if (timestamp - threeState.lastFrame < 33) return;
    threeState.lastFrame = timestamp;
    renderThreeFrame();
  };
  requestAnimationFrame(animate);
}

function initBackgroundCanvas() {
  const canvas = $("#bg-canvas");
  const ctx = canvas.getContext("2d");
  const points = [];

  function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    points.length = 0;
    for (let index = 0; index < 32; index += 1) {
      points.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.08,
        vy: (Math.random() - 0.5) * 0.08,
        radius: 1 + Math.random() * 2,
        alpha: 0.08 + Math.random() * 0.08,
      });
    }
  }

  let last = 0;
  function render(timestamp) {
    requestAnimationFrame(render);
    if (state.currentSection !== "home") return;
    if (timestamp - last < 33) return;
    last = timestamp;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    points.forEach((point) => {
      point.x += point.vx;
      point.y += point.vy;
      if (point.x < -20) point.x = canvas.width + 20;
      if (point.x > canvas.width + 20) point.x = -20;
      if (point.y < -20) point.y = canvas.height + 20;
      if (point.y > canvas.height + 20) point.y = -20;
      ctx.beginPath();
      ctx.fillStyle = `rgba(196, 30, 58, ${point.alpha})`;
      ctx.arc(point.x, point.y, point.radius, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  window.addEventListener("resize", resize);
  resize();
  requestAnimationFrame(render);
}

function initPointerSpotlight() {
  document.addEventListener("pointermove", (event) => {
    document.documentElement.style.setProperty("--spot-x", `${event.clientX}px`);
    document.documentElement.style.setProperty("--spot-y", `${event.clientY}px`);
  });
}

async function bootstrap() {
  state.bootstrap = await fetchJSON("/api/bootstrap");
  updateEngineInfo(state.bootstrap.engine);
  renderMetrics();
  renderEvidence();
  renderRegistrationSummary();
  renderReportSidebar();
  renderAnatomyControlState();
  renderHeartLegend();
  $("#report-paper").innerHTML = buildPlaceholderReport();
}

function initApp() {
  initNavigation();
  initUtilityButtons();
  initModalHandlers();
  initControls();
  initBackgroundCanvas();
  initPointerSpotlight();
  initThreeScene();
  window.addEventListener("resize", () => {
    applyMprViewportTransform("axial");
    applyMprViewportTransform("coronal");
    applyMprViewportTransform("sagittal");
    resizeThreeRenderer();
    renderThreeFrame();
  });
  bootstrap().catch((error) => showToast(error.message));
}

document.addEventListener("DOMContentLoaded", initApp);
