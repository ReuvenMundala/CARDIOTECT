const studyMetrics = [
  {
    label: 'Dice (CAC-positive)',
    value: 0.7968,
    decimals: 4,
    suffix: '',
    description: 'Segmentation overlap quality in calcium-positive studies.'
  },
  {
    label: 'ICC(2,1)',
    value: 0.9953,
    decimals: 4,
    suffix: '',
    description: 'Agreement between automated and expert Agatston scores.'
  },
  {
    label: 'R²',
    value: 0.9912,
    decimals: 4,
    suffix: '',
    description: 'Variance explained by CARDIOTECT score predictions.'
  },
  {
    label: 'Risk Accuracy',
    value: 98.9,
    decimals: 1,
    suffix: '%',
    description: 'Correct classification rate across five cardiovascular tiers.'
  },
  {
    label: 'Precision',
    value: 0.9325,
    decimals: 4,
    suffix: '',
    description: 'False-positive suppression in predicted calcium pixels.'
  },
  {
    label: 'Sensitivity',
    value: 0.7303,
    decimals: 4,
    suffix: '',
    description: 'Ability to retrieve true calcified regions.'
  },
  {
    label: 'Specificity',
    value: 1.0,
    decimals: 4,
    suffix: '',
    description: 'Perfect identification of CAC-zero patients in evaluation.'
  }
];

const territoryAgreement = [
  {
    name: 'LM-LAD',
    patients: 273,
    dice: 0.5703,
    icc: 0.8556,
    mae: 67.42,
    tone: 'good agreement on the merged left-anterior territory.'
  },
  {
    name: 'LCX',
    patients: 199,
    dice: 0.3896,
    icc: 0.7424,
    mae: 64.22,
    tone: 'moderate stability in the circumflex territory.'
  },
  {
    name: 'RCA',
    patients: 184,
    dice: 0.5179,
    icc: 0.9545,
    mae: 50.06,
    tone: 'the strongest agreement profile among the three territories.'
  }
];

const tiers = [
  { code: 'I', label: '0', title: 'No calcium detected', min: 0, max: 0 },
  { code: 'II', label: '1–10', title: 'Minimal burden', min: 1, max: 10 },
  { code: 'III', label: '11–100', title: 'Mild burden', min: 11, max: 100 },
  { code: 'IV', label: '101–400', title: 'Moderate burden', min: 101, max: 400 },
  { code: 'V', label: '>400', title: 'High burden', min: 401, max: Infinity }
];

const cases = [
  {
    id: 'P-021',
    title: 'Demo Patient 021',
    subtitle: 'Mild burden profile',
    age: 58,
    sex: 'F',
    score: 37,
    risk: 'III (11–100)',
    tier: 'III',
    confidence: 0.94,
    topTerritory: 'LM-LAD',
    note: 'A subtle anterior calcium pattern with small, high-confidence foci. This case is ideal for showcasing early detection, calm overlays, and crisp report generation.',
    vesselScores: { 'LM-LAD': 24, 'LCX': 5, 'RCA': 8 },
    lesions: [
      { x: 0.53, y: 0.42, rx: 24, ry: 14, slice: 18, spread: 5, territory: 'LM-LAD', driftX: 2, driftY: -1 },
      { x: 0.57, y: 0.45, rx: 17, ry: 10, slice: 21, spread: 3, territory: 'LM-LAD', driftX: -2, driftY: 1 },
      { x: 0.49, y: 0.53, rx: 12, ry: 9, slice: 26, spread: 4, territory: 'RCA', driftX: 1, driftY: 1 }
    ]
  },
  {
    id: 'P-117',
    title: 'Demo Patient 117',
    subtitle: 'Moderate multi-focal burden',
    age: 64,
    sex: 'M',
    score: 218,
    risk: 'IV (101–400)',
    tier: 'IV',
    confidence: 0.92,
    topTerritory: 'RCA',
    note: 'Multi-focal calcifications create a strong visual narrative for side-by-side overlays, territory bars, and tier escalation into the moderate-risk bucket.',
    vesselScores: { 'LM-LAD': 81, 'LCX': 37, 'RCA': 100 },
    lesions: [
      { x: 0.48, y: 0.39, rx: 32, ry: 16, slice: 16, spread: 6, territory: 'LM-LAD', driftX: 2, driftY: -1 },
      { x: 0.57, y: 0.47, rx: 28, ry: 15, slice: 19, spread: 5, territory: 'RCA', driftX: -3, driftY: 2 },
      { x: 0.44, y: 0.52, rx: 20, ry: 12, slice: 24, spread: 4, territory: 'LCX', driftX: 1, driftY: 2 },
      { x: 0.60, y: 0.38, rx: 16, ry: 10, slice: 28, spread: 3, territory: 'RCA', driftX: 3, driftY: 0 }
    ]
  },
  {
    id: 'P-208',
    title: 'Demo Patient 208',
    subtitle: 'Severe high-burden profile',
    age: 71,
    sex: 'M',
    score: 1287,
    risk: 'V (>400)',
    tier: 'V',
    confidence: 0.97,
    topTerritory: 'RCA',
    note: 'Dense, high-burden calcification designed to feel dramatic on-screen. This scene is built for wow-factor presentations, especially when the heart hologram and territory board update together.',
    vesselScores: { 'LM-LAD': 486, 'LCX': 221, 'RCA': 580 },
    lesions: [
      { x: 0.49, y: 0.37, rx: 46, ry: 22, slice: 13, spread: 8, territory: 'LM-LAD', driftX: 4, driftY: -2 },
      { x: 0.58, y: 0.44, rx: 42, ry: 20, slice: 17, spread: 7, territory: 'RCA', driftX: -4, driftY: 3 },
      { x: 0.44, y: 0.51, rx: 34, ry: 17, slice: 22, spread: 6, territory: 'LCX', driftX: 2, driftY: 3 },
      { x: 0.55, y: 0.57, rx: 28, ry: 14, slice: 26, spread: 5, territory: 'RCA', driftX: -1, driftY: 2 },
      { x: 0.62, y: 0.35, rx: 25, ry: 13, slice: 31, spread: 4, territory: 'LM-LAD', driftX: 1, driftY: -2 }
    ]
  },
  {
    id: 'P-311',
    title: 'Demo Patient 311',
    subtitle: 'CAC-zero healthy scan',
    age: 46,
    sex: 'F',
    score: 0,
    risk: 'I (0)',
    tier: 'I',
    confidence: 0.99,
    topTerritory: 'None',
    note: 'A clean zero-calcium case designed to emphasize specificity, low visual clutter, and the interface state for healthy scans.',
    vesselScores: { 'LM-LAD': 0, 'LCX': 0, 'RCA': 0 },
    lesions: []
  },
  {
    id: 'P-404',
    title: 'Demo Patient 404',
    subtitle: 'Minimal trace burden',
    age: 53,
    sex: 'M',
    score: 9,
    risk: 'II (1–10)',
    tier: 'II',
    confidence: 0.91,
    topTerritory: 'LM-LAD',
    note: 'A tiny-burden case that is useful for demonstrating how the product can keep the UI elegant while showing clinically small but visually meaningful findings.',
    vesselScores: { 'LM-LAD': 6, 'LCX': 0, 'RCA': 3 },
    lesions: [
      { x: 0.54, y: 0.44, rx: 14, ry: 8, slice: 20, spread: 3, territory: 'LM-LAD', driftX: 2, driftY: -1 },
      { x: 0.50, y: 0.56, rx: 8, ry: 6, slice: 24, spread: 2, territory: 'RCA', driftX: 1, driftY: 1 }
    ]
  }
];

const matrixData = [
  { label: 'I (0)', values: [261, 0, 0, 0, 0] },
  { label: 'II (1–10)', values: [0, 31, 0, 0, 0] },
  { label: 'III (11–100)', values: [0, 0, 96, 2, 0] },
  { label: 'IV (101–400)', values: [0, 0, 2, 69, 2] },
  { label: 'V (>400)', values: [0, 0, 0, 0, 88] }
];

let currentSection = 'overview';
let currentCaseIndex = 0;
let currentSlice = 18;
let heatStrength = 0.72;
let overlayEnabled = true;
let playingSlices = false;
let territoryMode = 'agreement';
let toastTimer;

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

function formatNumber(value, decimals = 0) {
  return Number(value).toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  });
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function gaussian(x, mean, spread) {
  return Math.exp(-Math.pow(x - mean, 2) / (2 * spread * spread));
}

function mulberry32(seed) {
  let t = seed >>> 0;
  return function () {
    t += 0x6D2B79F5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function showToast(message) {
  const toast = $('#toast');
  toast.textContent = message;
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 2200);
}

function getCurrentCase() {
  return cases[currentCaseIndex];
}

function navigateTo(sectionId) {
  currentSection = sectionId;
  $$('.nav-btn').forEach((btn) => btn.classList.toggle('active', btn.dataset.section === sectionId));
  $$('.content-section').forEach((section) => section.classList.toggle('active', section.id === sectionId));
}

function initNavigation() {
  $$('.nav-btn').forEach((btn) => {
    btn.addEventListener('click', () => navigateTo(btn.dataset.section));
  });

  $$('[data-jump]').forEach((btn) => {
    btn.addEventListener('click', () => navigateTo(btn.dataset.jump));
  });

  $('#quick-search').addEventListener('keydown', (event) => {
    if (event.key !== 'Enter') return;
    const q = event.target.value.trim().toLowerCase();
    if (!q) return;
    const sectionKeywords = {
      overview: ['overview', 'hero', 'dashboard'],
      studio: ['ct', 'scan', 'studio', 'slice'],
      territories: ['territory', 'lm-lad', 'lcx', 'rca', 'vessel'],
      risk: ['risk', 'tier', 'classification', 'kappa'],
      evidence: ['figure', 'evidence', 'plot', 'bland', 'scatter'],
      report: ['report', 'export', 'pdf', 'summary']
    };

    const foundSection = Object.entries(sectionKeywords).find(([, words]) => words.some((word) => q.includes(word)));
    const foundCase = cases.find((item) => `${item.id} ${item.title} ${item.subtitle}`.toLowerCase().includes(q));

    if (foundCase) {
      currentCaseIndex = cases.indexOf(foundCase);
      renderCaseList();
      updatePatientViews();
      navigateTo('studio');
      showToast(`Loaded ${foundCase.title}.`);
      return;
    }

    if (foundSection) {
      navigateTo(foundSection[0]);
      showToast(`Jumped to ${foundSection[0]}.`);
      return;
    }

    showToast('No direct match found. Try “risk”, “studio”, or a demo patient ID.');
  });
}

function renderMetrics() {
  const metricGrid = $('#metric-grid');
  metricGrid.innerHTML = '';

  studyMetrics.forEach((metric, index) => {
    const card = document.createElement('article');
    card.className = 'metric-card glass tilt-card';
    card.innerHTML = `
      <h3>${metric.label}</h3>
      <div class="metric-value">
        <span class="metric-number" data-target="${metric.value}" data-decimals="${metric.decimals}">0</span>
        <span class="metric-suffix">${metric.suffix || ''}</span>
      </div>
      <p>${metric.description}</p>
    `;
    metricGrid.appendChild(card);

    requestAnimationFrame(() => animateMetric(card.querySelector('.metric-number'), metric.value, metric.decimals, index * 120));
  });
}

function animateMetric(node, target, decimals, delay = 0) {
  const duration = 1100;
  const start = performance.now() + delay;
  function step(now) {
    const progress = clamp((now - start) / duration, 0, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    node.textContent = formatNumber(target * eased, decimals);
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function initPipelineCycle() {
  const steps = $$('.pipeline-step');
  let active = 0;
  setInterval(() => {
    steps[active].classList.remove('active');
    active = (active + 1) % steps.length;
    steps[active].classList.add('active');
  }, 2200);
}

function renderCaseList() {
  const caseList = $('#case-list');
  caseList.innerHTML = '';
  cases.forEach((item, index) => {
    const button = document.createElement('button');
    button.className = `case-item${index === currentCaseIndex ? ' active' : ''}`;
    button.innerHTML = `
      <h4>${item.title}</h4>
      <small>${item.id} · Age ${item.age} · ${item.sex}</small>
      <p>${item.subtitle}</p>
    `;
    button.addEventListener('click', () => {
      currentCaseIndex = index;
      renderCaseList();
      updatePatientViews();
      navigateTo('studio');
    });
    caseList.appendChild(button);
  });
}

function updatePatientViews() {
  const patient = getCurrentCase();
  $('#case-title').textContent = `${patient.title} — ${patient.subtitle}`;
  $('#patient-score').textContent = formatNumber(patient.score, 0);
  $('#patient-risk').textContent = patient.risk;
  $('#patient-territory').textContent = patient.topTerritory;
  $('#patient-confidence').textContent = formatNumber(patient.confidence, 2);
  $('#patient-note').textContent = patient.note;

  const vesselWrap = $('#patient-vessels');
  vesselWrap.innerHTML = '';
  const maxScore = Math.max(...Object.values(patient.vesselScores), 1);
  Object.entries(patient.vesselScores).forEach(([name, score]) => {
    const row = document.createElement('div');
    row.className = 'mini-bar';
    row.innerHTML = `
      <div class="line-head"><span>${name}</span><strong>${formatNumber(score, 0)}</strong></div>
      <div class="bar-track"><div class="bar-fill" style="width:${(score / maxScore) * 100}%"></div></div>
    `;
    vesselWrap.appendChild(row);
  });

  currentSlice = clamp(currentSlice, 0, 47);
  $('#slice-slider').value = currentSlice;
  $('#slice-readout').textContent = `${currentSlice} / 48`;
  updateRiskPanel();
  renderTerritoryBoard();
  renderReport();
  drawCTCanvas();
}

function classifyTier(score) {
  return tiers.find((tier) => score >= tier.min && score <= tier.max) || tiers[tiers.length - 1];
}

function riskGradientForTier(tierCode) {
  const activeIndex = Math.max(0, tiers.findIndex((tier) => tier.code === tierCode));
  const colors = [
    'rgba(105,240,255,0.95)',
    'rgba(105,240,255,0.68)',
    'rgba(247,205,125,0.92)',
    'rgba(255,93,133,0.92)',
    'rgba(159,131,255,0.95)'
  ];
  const muted = 'rgba(255,255,255,0.08)';
  const segments = colors.map((color, index) => index <= activeIndex ? color : muted);
  const segmentWidth = 72;
  const stops = segments.map((color, index) => `${color} ${index * segmentWidth}deg ${(index + 1) * segmentWidth}deg`);
  return `conic-gradient(from -90deg, ${stops.join(',')})`;
}

function updateRiskPanel() {
  const patient = getCurrentCase();
  const tier = classifyTier(patient.score);
  $('#risk-tier-label').textContent = tier.code;
  $('#risk-donut').style.background = riskGradientForTier(tier.code);

  const tierList = $('#tier-list');
  tierList.innerHTML = '';
  tiers.forEach((item) => {
    const card = document.createElement('div');
    card.className = `tier-item${item.code === tier.code ? ' active' : ''}`;
    card.innerHTML = `<strong>${item.code} · ${item.label}</strong><small>${item.title}</small>`;
    tierList.appendChild(card);
  });
}

function renderMatrix() {
  const grid = $('#matrix-grid');
  grid.innerHTML = '';
  const labels = ['Pred / GT', ...matrixData.map((row) => row.label)];

  labels.forEach((label, index) => {
    const div = document.createElement('div');
    div.className = `matrix-label${index === 0 ? '' : ''}`;
    div.textContent = label;
    grid.appendChild(div);
  });

  const maxValue = 261;
  matrixData.forEach((row) => {
    const rowLabel = document.createElement('div');
    rowLabel.className = 'matrix-label';
    rowLabel.textContent = row.label;
    grid.appendChild(rowLabel);

    const rowTotal = row.values.reduce((sum, value) => sum + value, 0);
    row.values.forEach((value, colIndex) => {
      const pct = rowTotal ? Math.round((value / rowTotal) * 100) : 0;
      const isDiagonal = matrixData.indexOf(row) === colIndex;
      const alpha = clamp(value / maxValue, 0, 1);
      const bg = isDiagonal
        ? `linear-gradient(180deg, rgba(105, 240, 255, ${0.18 + alpha * 0.72}), rgba(18, 52, 93, ${0.32 + alpha * 0.42}))`
        : value > 0
          ? `linear-gradient(180deg, rgba(255, 93, 133, ${0.20 + alpha * 0.55}), rgba(82, 12, 24, 0.74))`
          : 'rgba(255,255,255,0.04)';
      const cell = document.createElement('div');
      cell.className = 'matrix-cell';
      cell.style.background = bg;
      cell.innerHTML = `<strong>${value}</strong><small>${pct}%</small>`;
      grid.appendChild(cell);
    });
  });
}

function renderTerritoryBoard() {
  const wrap = $('#territory-bars');
  wrap.innerHTML = '';
  const patient = getCurrentCase();
  const title = $('#territory-board-title');

  if (territoryMode === 'agreement') {
    title.textContent = 'Study-wide agreement profile';
    const maxMae = Math.max(...territoryAgreement.map((item) => item.mae));
    territoryAgreement.forEach((item) => {
      const row = document.createElement('div');
      row.className = 'territory-row';
      row.innerHTML = `
        <div class="territory-row-top">
          <div>
            <h4>${item.name}</h4>
            <p>${item.tone}</p>
          </div>
          <span class="chip">n = ${item.patients}</span>
        </div>
        <div class="metric-line">
          <div class="line-head"><span>Mean Dice</span><strong>${formatNumber(item.dice, 4)}</strong></div>
          <div class="line-track"><div class="line-fill fill-red" style="width:${item.dice * 100}%"></div></div>
        </div>
        <div class="metric-line">
          <div class="line-head"><span>ICC(2,1)</span><strong>${formatNumber(item.icc, 4)}</strong></div>
          <div class="line-track"><div class="line-fill fill-cyan" style="width:${item.icc * 100}%"></div></div>
        </div>
        <div class="metric-line">
          <div class="line-head"><span>MAE</span><strong>${formatNumber(item.mae, 2)}</strong></div>
          <div class="line-track"><div class="line-fill fill-gold" style="width:${(item.mae / maxMae) * 100}%"></div></div>
        </div>
      `;
      wrap.appendChild(row);
    });
  } else {
    title.textContent = `${patient.title} territory burden`;
    const values = Object.entries(patient.vesselScores);
    const maxBurden = Math.max(...values.map(([, score]) => score), 1);
    values.forEach(([name, score]) => {
      const studyRow = territoryAgreement.find((item) => item.name === name);
      const row = document.createElement('div');
      row.className = 'territory-row';
      row.innerHTML = `
        <div class="territory-row-top">
          <div>
            <h4>${name}</h4>
            <p>Patient-specific contribution with study agreement context.</p>
          </div>
          <span class="chip">ICC ${formatNumber(studyRow.icc, 4)}</span>
        </div>
        <div class="metric-line">
          <div class="line-head"><span>Patient burden</span><strong>${formatNumber(score, 0)}</strong></div>
          <div class="line-track"><div class="line-fill fill-gold" style="width:${(score / maxBurden) * 100}%"></div></div>
        </div>
        <div class="metric-line">
          <div class="line-head"><span>Study Dice</span><strong>${formatNumber(studyRow.dice, 4)}</strong></div>
          <div class="line-track"><div class="line-fill fill-red" style="width:${studyRow.dice * 100}%"></div></div>
        </div>
        <div class="metric-line">
          <div class="line-head"><span>Study ICC</span><strong>${formatNumber(studyRow.icc, 4)}</strong></div>
          <div class="line-track"><div class="line-fill fill-cyan" style="width:${studyRow.icc * 100}%"></div></div>
        </div>
      `;
      wrap.appendChild(row);
    });
  }
}

function initTerritoryToggle() {
  $$('.territory-toggle').forEach((btn) => {
    btn.addEventListener('click', () => {
      territoryMode = btn.dataset.mode;
      $$('.territory-toggle').forEach((b) => b.classList.toggle('active', b === btn));
      renderTerritoryBoard();
    });
  });
}

function renderReport() {
  const patient = getCurrentCase();
  const report = $('#report-paper');
  const tier = classifyTier(patient.score);
  const topTerritory = Object.entries(patient.vesselScores).sort((a, b) => b[1] - a[1])[0] || ['None', 0];
  const vesselRows = Object.entries(patient.vesselScores)
    .map(([name, score]) => `<div class="kv-card"><span>${name}</span><strong>${formatNumber(score, 0)}</strong></div>`)
    .join('');

  report.innerHTML = `
    <h3>${patient.title} · Automated CAC Summary</h3>
    <p>Draft report interface for presentation and product visualization. This layout mirrors a future export-ready CARDIOTECT summary page.</p>
    <div class="paper-meta">
      <div><span>Patient ID</span><strong>${patient.id}</strong></div>
      <div><span>Age / Sex</span><strong>${patient.age} / ${patient.sex}</strong></div>
      <div><span>Agatston</span><strong>${formatNumber(patient.score, 0)}</strong></div>
      <div><span>Risk Tier</span><strong>${tier.code} · ${patient.risk}</strong></div>
    </div>

    <div class="paper-section">
      <strong>Automated impression</strong>
      <p>${patient.note}</p>
    </div>

    <div class="paper-section">
      <strong>Territory burden</strong>
      <div class="paper-grid">${vesselRows}</div>
    </div>

    <div class="paper-section">
      <strong>Confidence and output logic</strong>
      <p>Segmentation results are filtered using a 0.75 confidence threshold, an HU &gt; 130 rule, and connected-component lesion filtering before Agatston scoring and tier classification are produced.</p>
    </div>

    <div class="paper-section">
      <strong>Presentation note</strong>
      <p>Top territory: <strong>${topTerritory[0]}</strong>. This simulated interface is designed to help communicate the study’s outputs visually, while keeping the narrative aligned with the paper’s segmentation, scoring, and risk stratification pipeline.</p>
    </div>
  `;
}

function drawRoundedBlob(ctx, x, y, rx, ry, rotation = 0) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(rotation);
  ctx.beginPath();
  ctx.moveTo(-rx * 0.8, 0);
  ctx.bezierCurveTo(-rx, -ry * 0.8, -rx * 0.2, -ry, 0, -ry * 0.7);
  ctx.bezierCurveTo(rx * 0.3, -ry, rx, -ry * 0.4, rx * 0.9, 0);
  ctx.bezierCurveTo(rx, ry * 0.55, rx * 0.25, ry, 0, ry * 0.82);
  ctx.bezierCurveTo(-rx * 0.35, ry, -rx, ry * 0.5, -rx * 0.8, 0);
  ctx.closePath();
  ctx.restore();
}

function drawCTCanvas() {
  const canvas = $('#ct-canvas');
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  const patient = getCurrentCase();
  const seed = (currentCaseIndex + 1) * 1000 + currentSlice * 31;
  const rand = mulberry32(seed);

  ctx.clearRect(0, 0, w, h);

  const bg = ctx.createLinearGradient(0, 0, w, h);
  bg.addColorStop(0, '#0f1118');
  bg.addColorStop(1, '#05070b');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);

  const cx = w * 0.5;
  const cy = h * 0.51;
  const scalePulse = 1 + Math.sin(currentSlice * 0.22) * 0.01;

  const vignette = ctx.createRadialGradient(cx, cy, 40, cx, cy, w * 0.48);
  vignette.addColorStop(0, 'rgba(255,255,255,0.04)');
  vignette.addColorStop(1, 'rgba(0,0,0,0.72)');

  ctx.save();
  ctx.translate(cx, cy);
  ctx.scale(scalePulse, scalePulse);

  const bodyGrad = ctx.createRadialGradient(0, 0, 40, 0, 0, 260);
  bodyGrad.addColorStop(0, 'rgba(180, 180, 188, 0.72)');
  bodyGrad.addColorStop(0.48, 'rgba(110, 110, 120, 0.95)');
  bodyGrad.addColorStop(1, 'rgba(46, 46, 54, 0.98)');
  ctx.fillStyle = bodyGrad;
  ctx.beginPath();
  ctx.ellipse(0, 0, 285, 255, 0, 0, Math.PI * 2);
  ctx.fill();

  ctx.globalCompositeOperation = 'multiply';
  ctx.fillStyle = 'rgba(25, 25, 29, 0.82)';
  ctx.beginPath();
  ctx.ellipse(-118, -18, 95, 140, -0.18, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.ellipse(118, -18, 95, 140, 0.18, 0, Math.PI * 2);
  ctx.fill();

  ctx.globalCompositeOperation = 'screen';
  const heartGrad = ctx.createRadialGradient(15, 12, 20, 12, 18, 150);
  heartGrad.addColorStop(0, 'rgba(255,255,255,0.30)');
  heartGrad.addColorStop(0.4, 'rgba(135, 135, 145, 0.30)');
  heartGrad.addColorStop(1, 'rgba(70, 70, 80, 0.12)');
  ctx.fillStyle = heartGrad;
  ctx.beginPath();
  ctx.ellipse(22, 35, 130, 108, 0.18, 0, Math.PI * 2);
  ctx.fill();

  ctx.globalCompositeOperation = 'source-over';
  ctx.fillStyle = 'rgba(215, 215, 228, 0.20)';
  ctx.beginPath();
  ctx.arc(0, 180, 42, 0, Math.PI * 2);
  ctx.fill();

  for (let i = 0; i < 1300; i += 1) {
    const x = (rand() - 0.5) * 560;
    const y = (rand() - 0.5) * 500;
    const alpha = 0.03 + rand() * 0.05;
    const size = 0.7 + rand() * 1.5;
    ctx.fillStyle = `rgba(255,255,255,${alpha})`;
    ctx.fillRect(x, y, size, size);
  }

  ctx.restore();

  const territoryColors = {
    'LM-LAD': 'rgba(255, 93, 133, 0.95)',
    'LCX': 'rgba(105, 240, 255, 0.95)',
    'RCA': 'rgba(247, 205, 125, 0.95)'
  };

  patient.lesions.forEach((lesion) => {
    const presence = gaussian(currentSlice, lesion.slice, lesion.spread);
    if (presence < 0.04) return;
    const px = lesion.x * w;
    const py = lesion.y * h;
    const rx = lesion.rx * (0.65 + presence * 0.75);
    const ry = lesion.ry * (0.65 + presence * 0.75);
    const rotation = Math.sin(currentSlice * 0.14 + lesion.rx) * 0.18;

    ctx.save();
    ctx.filter = `blur(${1 + presence * 2.5}px)`;
    const lesionGlow = ctx.createRadialGradient(px, py, 4, px, py, rx * 1.6);
    lesionGlow.addColorStop(0, `rgba(255,255,255,${0.45 + presence * 0.45})`);
    lesionGlow.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = lesionGlow;
    ctx.beginPath();
    ctx.ellipse(px, py, rx * 1.6, ry * 1.8, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    ctx.save();
    ctx.fillStyle = 'rgba(245, 246, 250, 0.95)';
    drawRoundedBlob(ctx, px, py, rx, ry, rotation);
    ctx.fill();
    ctx.restore();

    const heatAlpha = 0.08 + heatStrength * 0.28 * presence;
    const heat = ctx.createRadialGradient(px, py, 2, px, py, rx * 3.1);
    heat.addColorStop(0, `rgba(255, 84, 124, ${heatAlpha})`);
    heat.addColorStop(0.45, `rgba(255, 178, 84, ${heatAlpha * 0.7})`);
    heat.addColorStop(1, 'rgba(255, 178, 84, 0)');
    ctx.fillStyle = heat;
    ctx.beginPath();
    ctx.arc(px, py, rx * 3.1, 0, Math.PI * 2);
    ctx.fill();

    ctx.save();
    ctx.strokeStyle = 'rgba(130, 255, 166, 0.88)';
    ctx.setLineDash([8, 7]);
    ctx.lineWidth = 2.4;
    drawRoundedBlob(ctx, px, py, rx * 1.03, ry * 1.03, rotation);
    ctx.stroke();
    ctx.restore();

    if (overlayEnabled) {
      ctx.save();
      ctx.strokeStyle = territoryColors[lesion.territory];
      ctx.lineWidth = 3;
      ctx.shadowColor = territoryColors[lesion.territory];
      ctx.shadowBlur = 18;
      drawRoundedBlob(ctx, px + lesion.driftX, py + lesion.driftY, rx * 0.95, ry * 0.95, rotation + 0.05);
      ctx.stroke();
      ctx.restore();
    }
  });

  const scanY = ((performance.now() * 0.12) % (h + 120)) - 60;
  const scan = ctx.createLinearGradient(0, scanY - 40, 0, scanY + 40);
  scan.addColorStop(0, 'rgba(255,255,255,0)');
  scan.addColorStop(0.5, 'rgba(105,240,255,0.18)');
  scan.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = scan;
  ctx.fillRect(0, scanY - 40, w, 80);

  ctx.fillStyle = vignette;
  ctx.fillRect(0, 0, w, h);

  ctx.fillStyle = 'rgba(255,255,255,0.8)';
  ctx.font = '600 18px Inter, system-ui, sans-serif';
  ctx.fillText(`Case ${patient.id} · ${patient.subtitle}`, 28, 36);
  ctx.fillStyle = 'rgba(255,255,255,0.55)';
  ctx.font = '500 15px Inter, system-ui, sans-serif';
  ctx.fillText(overlayEnabled ? 'Expert outline + AI overlay active' : 'Expert outline only', 28, 62);
}

function initStudioControls() {
  $('#slice-slider').addEventListener('input', (event) => {
    currentSlice = Number(event.target.value);
    $('#slice-readout').textContent = `${currentSlice} / 48`;
    drawCTCanvas();
  });

  $('#heat-slider').addEventListener('input', (event) => {
    heatStrength = Number(event.target.value) / 100;
    drawCTCanvas();
  });

  $('#toggle-overlay').addEventListener('click', () => {
    overlayEnabled = !overlayEnabled;
    $('#toggle-overlay').classList.toggle('active', overlayEnabled);
    drawCTCanvas();
    showToast(overlayEnabled ? 'AI overlay enabled.' : 'AI overlay hidden.');
  });

  $('#play-slices').addEventListener('click', () => {
    playingSlices = !playingSlices;
    $('#play-slices').textContent = playingSlices ? 'Pause slices' : 'Play slices';
    showToast(playingSlices ? 'Slice playback started.' : 'Slice playback paused.');
  });

  $('#cycle-case').addEventListener('click', () => {
    currentCaseIndex = (currentCaseIndex + 1) % cases.length;
    renderCaseList();
    updatePatientViews();
    navigateTo('studio');
  });
}

function initEvidenceModal() {
  const modal = $('#image-modal');
  const modalTitle = $('#modal-title');
  const modalImage = $('#modal-image');

  $$('.evidence-card').forEach((card) => {
    card.addEventListener('click', () => {
      modal.classList.add('active');
      modal.setAttribute('aria-hidden', 'false');
      modalTitle.textContent = card.dataset.title;
      modalImage.src = card.dataset.image;
    });
  });

  const closeModal = () => {
    modal.classList.remove('active');
    modal.setAttribute('aria-hidden', 'true');
  };

  $('#modal-close').addEventListener('click', closeModal);
  $('#modal-x').addEventListener('click', closeModal);
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') closeModal();
  });
}

function initReportActions() {
  $('#simulate-export').addEventListener('click', () => {
    showToast('Export simulated — in a real build this would generate a polished PDF report.');
  });
}

function initDemoPulse() {
  $('#simulate-btn').addEventListener('click', () => {
    document.body.classList.add('demo-pulse');
    showToast('Demo pulse activated. Metrics and visuals are emphasizing presentation mode.');
    setTimeout(() => document.body.classList.remove('demo-pulse'), 1800);
  });
}

function initTilt() {
  $$('.tilt-card').forEach((card) => {
    card.addEventListener('mousemove', (event) => {
      const rect = card.getBoundingClientRect();
      const px = (event.clientX - rect.left) / rect.width;
      const py = (event.clientY - rect.top) / rect.height;
      const rx = (0.5 - py) * 6;
      const ry = (px - 0.5) * 8;
      card.style.transform = `perspective(1200px) rotateX(${rx}deg) rotateY(${ry}deg) translateY(-2px)`;
    });
    card.addEventListener('mouseleave', () => {
      card.style.transform = 'perspective(1200px) rotateX(0deg) rotateY(0deg) translateY(0px)';
    });
  });
}

function initSpotlight() {
  document.addEventListener('pointermove', (event) => {
    document.documentElement.style.setProperty('--spot-x', `${event.clientX}px`);
    document.documentElement.style.setProperty('--spot-y', `${event.clientY}px`);
  });
}

function initBackgroundCanvas() {
  const canvas = $('#bg-canvas');
  const ctx = canvas.getContext('2d');
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  let width = 0;
  let height = 0;
  let particles = [];

  const resize = () => {
    width = canvas.clientWidth = window.innerWidth;
    height = canvas.clientHeight = window.innerHeight;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    particles = Array.from({ length: 70 }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.16,
      vy: (Math.random() - 0.5) * 0.16,
      r: 1 + Math.random() * 2.4,
      a: 0.05 + Math.random() * 0.20,
      c: Math.random() > 0.55 ? '255,93,133' : Math.random() > 0.5 ? '105,240,255' : '255,255,255'
    }));
  };

  const render = () => {
    ctx.clearRect(0, 0, width, height);
    particles.forEach((p) => {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < -10) p.x = width + 10;
      if (p.x > width + 10) p.x = -10;
      if (p.y < -10) p.y = height + 10;
      if (p.y > height + 10) p.y = -10;
      ctx.beginPath();
      ctx.fillStyle = `rgba(${p.c}, ${p.a})`;
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();
    });

    for (let i = 0; i < particles.length; i += 1) {
      const a = particles[i];
      for (let j = i + 1; j < particles.length; j += 1) {
        const b = particles[j];
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 110) {
          ctx.strokeStyle = `rgba(255,255,255,${(1 - dist / 110) * 0.04})`;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }

    requestAnimationFrame(render);
  };

  window.addEventListener('resize', resize);
  resize();
  render();
}

function initHeartCanvas() {
  const canvas = $('#heart-canvas');
  const ctx = canvas.getContext('2d');
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  let width = 0;
  let height = 0;
  let points = [];
  const state = {
    rotX: 0.16,
    rotY: 0.38,
    targetX: 0.16,
    targetY: 0.38
  };

  function rebuildPoints() {
    points = [];
    for (let z = -1; z <= 1; z += 0.08) {
      const layer = Math.sqrt(Math.max(0.02, 1 - z * z));
      for (let t = 0; t < Math.PI * 2; t += 0.09) {
        const hx = 16 * Math.pow(Math.sin(t), 3) / 18;
        const hy = -(13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t)) / 18;
        const density = z > -0.2 && z < 0.5 ? 3 : 2;
        for (let i = 0; i < density; i += 1) {
          const r = 0.32 + Math.random() * 0.68;
          points.push({
            x: hx * layer * r * 0.98,
            y: hy * layer * r,
            z: z * (1.18 - r * 0.35),
            s: 0.7 + Math.random() * 1.7,
            tint: Math.random()
          });
        }
      }
    }
  }

  function resize() {
    width = canvas.clientWidth || 680;
    height = canvas.clientHeight || 500;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    rebuildPoints();
  }

  function render(now) {
    ctx.clearRect(0, 0, width, height);
    const cx = width * 0.5;
    const cy = height * 0.52;
    const scale = Math.min(width, height) * 0.23;
    const pulse = 1 + Math.sin(now * 0.003) * 0.045;

    state.rotX += (state.targetX - state.rotX) * 0.03;
    state.rotY += (state.targetY - state.rotY) * 0.03;

    const aura = ctx.createRadialGradient(cx, cy, 10, cx, cy, width * 0.33);
    aura.addColorStop(0, 'rgba(255, 98, 140, 0.20)');
    aura.addColorStop(0.4, 'rgba(105, 240, 255, 0.07)');
    aura.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = aura;
    ctx.beginPath();
    ctx.arc(cx, cy, width * 0.33, 0, Math.PI * 2);
    ctx.fill();

    const transformed = points.map((point) => {
      let x = point.x * pulse;
      let y = point.y * pulse;
      let z = point.z * pulse;

      const cosY = Math.cos(state.rotY);
      const sinY = Math.sin(state.rotY);
      const x1 = x * cosY + z * sinY;
      const z1 = z * cosY - x * sinY;

      const cosX = Math.cos(state.rotX);
      const sinX = Math.sin(state.rotX);
      const y2 = y * cosX - z1 * sinX;
      const z2 = z1 * cosX + y * sinX;

      const perspective = 3.8 / (4.6 - z2);
      return {
        x: cx + x1 * scale * perspective,
        y: cy + y2 * scale * perspective,
        z: z2,
        size: point.s * perspective,
        tint: point.tint
      };
    }).sort((a, b) => a.z - b.z);

    transformed.forEach((p) => {
      const alpha = 0.18 + ((p.z + 1.3) / 2.6) * 0.82;
      const hue = p.tint > 0.68 ? '105,240,255' : p.tint > 0.35 ? '255,141,168' : '255,83,130';
      ctx.fillStyle = `rgba(${hue}, ${alpha})`;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size * 0.92, 0, Math.PI * 2);
      ctx.fill();
    });

    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.ellipse(cx, cy, width * 0.2, height * 0.3, now * 0.00012, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.ellipse(cx, cy, width * 0.16, height * 0.22, -now * 0.0001, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();

    requestAnimationFrame(render);
  }

  canvas.addEventListener('pointermove', (event) => {
    const rect = canvas.getBoundingClientRect();
    const px = (event.clientX - rect.left) / rect.width;
    const py = (event.clientY - rect.top) / rect.height;
    state.targetY = 0.18 + (px - 0.5) * 0.9;
    state.targetX = 0.05 + (0.5 - py) * 0.5;
  });
  canvas.addEventListener('pointerleave', () => {
    state.targetX = 0.16;
    state.targetY = 0.38;
  });

  window.addEventListener('resize', resize);
  resize();
  requestAnimationFrame(render);
}

function animationLoop() {
  if (playingSlices) {
    currentSlice = (currentSlice + 1) % 48;
    $('#slice-slider').value = currentSlice;
    $('#slice-readout').textContent = `${currentSlice} / 48`;
  }
  drawCTCanvas();
  requestAnimationFrame(animationLoop);
}

function initButtons() {
  $('#toggle-overlay').classList.add('active');
}

function initApp() {
  initNavigation();
  renderMetrics();
  initPipelineCycle();
  renderCaseList();
  renderMatrix();
  initTerritoryToggle();
  updatePatientViews();
  initStudioControls();
  initEvidenceModal();
  initReportActions();
  initDemoPulse();
  initTilt();
  initSpotlight();
  initButtons();
  initBackgroundCanvas();
  initHeartCanvas();
  animationLoop();
}

document.addEventListener('DOMContentLoaded', initApp);
