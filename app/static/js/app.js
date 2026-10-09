/* ==========================================================================
   Churn Early-Warning Dashboard — front end (Application Lead: Praveen)

   Data flow
   ---------
   GET  /api/config   → τ*, business costs, risk tiers, model metadata
   POST /api/predict  → customers ranked by calibrated churn probability
                        (the backend also appends every prediction to the log)
   POST /api/explain  → top-5 SHAP drivers for the customer opened in the drawer
   GET  /api/logs     → recent prediction-log records + totals
   DELETE /api/logs   → "Clear log" button (asks for confirmation first)

   Campaign economics are computed in the browser so the slider is instant.
   Formulas are identical to src/cost_optimizer.expected_cost_unlabeled():
       campaign cost = #(p ≥ τ) × C_offer
       missed loss   = Σ_{p < τ} p × C_lost
       do nothing    = Σ p × C_lost
       net savings   = do nothing − (campaign cost + missed loss)
   Probabilities are sorted once per upload and prefix-summed, so every τ is
   evaluated with one binary search — O(log n) — even for 50,000 customers.

   Security: every value that came from an uploaded file is escaped before it
   is inserted into the page, and CSV exports neutralise spreadsheet formulas.
   ========================================================================== */
'use strict';

const PAGE_SIZE = 50;
const THEME_KEY = 'churn-theme';
const GRID = Array.from({ length: 99 }, (_, i) => (i + 1) / 100);   // τ = 0.01 … 0.99 (as notebook 06)

const state = {
  modelReady: false,
  config: null,
  customers: [],
  sortedP: new Float64Array(0),
  prefix: new Float64Array(1),
  batch: null,
  tau: 0.5,
  tauStar: 0.5,
  cOffer: 1500,
  cLost: 15000,
  currency: '₹',
  search: '',
  tier: 'all',
  onlyTargeted: false,
  sort: { key: 'risk', dir: 'desc' },
  visible: PAGE_SIZE,
  view: [],
  curve: null,
  distBins: [],
  costGeom: null,
  drawerIndex: -1,
  drawerCustomer: null,
  explainCache: new Map(),
  busy: false,
  logs: { records: [], stats: null, search: '', tier: 'all' },
};

const el = (id) => document.getElementById(id);
let drawer = null;
let renderQueued = false;
let lastFocusedRow = null;

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------
function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (ch) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]
  ));
}
function icon(name, cls = '') {
  return `<svg class="icon ${cls}" aria-hidden="true"><use href="#i-${name}"/></svg>`;
}
const inr = new Intl.NumberFormat('en-IN', { maximumFractionDigits: 0 });
function money(value) {
  const r = Math.round(value);
  return `${r < 0 ? '−' : ''}${state.currency}${inr.format(Math.abs(r))}`;
}
/** Compact Indian units: K (thousand), L (lakh = 1e5), Cr (crore = 1e7). */
function moneyCompact(value) {
  const a = Math.abs(value);
  const sign = value < 0 ? '−' : '';
  const fmt = (x) => String(Number(x.toFixed(x >= 10 ? 0 : 1)));
  if (a >= 1e7) return `${sign}${state.currency}${fmt(a / 1e7)}Cr`;
  if (a >= 1e5) return `${sign}${state.currency}${fmt(a / 1e5)}L`;
  if (a >= 1e3) return `${sign}${state.currency}${fmt(a / 1e3)}K`;
  return `${sign}${state.currency}${Math.round(a)}`;
}
function pct(p, digits = 1) { return `${(p * 100).toFixed(digits)}%`; }
function int(n) { return Number(n).toLocaleString('en-IN'); }
function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }
const relFmt = typeof Intl.RelativeTimeFormat === 'function' ? new Intl.RelativeTimeFormat('en', { numeric: 'auto' }) : null;
function relTime(iso) {
  const seconds = (new Date(iso).getTime() - Date.now()) / 1000;
  if (!relFmt || !Number.isFinite(seconds)) return new Date(iso).toLocaleString();
  const units = [['day', 86400], ['hour', 3600], ['minute', 60], ['second', 1]];
  for (const [unit, size] of units) {
    if (Math.abs(seconds) >= size || unit === 'second') return relFmt.format(Math.round(seconds / size), unit);
  }
  return '';
}

const TIERS = ['High', 'Medium', 'Low'];
const TIER_ICON = { High: 'high', Medium: 'medium', Low: 'low' };
function tierChip(tier) {
  return `<span class="tier-chip tier-${escapeHtml(tier)}">${icon(TIER_ICON[tier] || 'info')}${escapeHtml(tier)}</span>`;
}

// ---------------------------------------------------------------------------
// Networking, alerts, toasts, busy state
// ---------------------------------------------------------------------------
async function api(path, options = {}) {
  const response = await fetch(path, options);
  let body = null;
  try { body = await response.json(); } catch (_) { /* e.g. a proxy error page */ }
  if (!response.ok) {
    let message = response.statusText || `HTTP ${response.status}`;
    const detail = body && body.detail;
    if (typeof detail === 'string') message = detail;
    else if (detail && detail.message) message = detail.message;
    else if (Array.isArray(detail) && detail.length) message = detail.map((d) => d.msg).join('; ');
    const error = new Error(message);
    error.status = response.status;
    error.body = body;
    throw error;
  }
  return body;
}

function showAlert(message, kind = 'error') {
  const node = document.createElement('div');
  node.className = `app-alert is-${kind}`;
  node.setAttribute('role', kind === 'error' ? 'alert' : 'status');
  node.innerHTML = `${icon(kind === 'error' ? 'high' : 'alert')}<div class="alert-body">${escapeHtml(message)}</div>
    <button type="button" class="alert-close" aria-label="Dismiss">${icon('x')}</button>`;
  node.querySelector('.alert-close').addEventListener('click', () => node.remove());
  el('alert-area').appendChild(node);
}
function clearAlerts() { el('alert-area').innerHTML = ''; }

function toast(message) {
  const node = document.createElement('div');
  node.className = 'toast app-toast';
  node.setAttribute('role', 'status');
  node.innerHTML = `<div class="toast-body">${icon('low')}<span>${escapeHtml(message)}</span>
    <button type="button" class="toast-close" data-bs-dismiss="toast" aria-label="Close">${icon('x')}</button></div>`;
  el('toast-area').appendChild(node);
  node.addEventListener('hidden.bs.toast', () => node.remove());
  bootstrap.Toast.getOrCreateInstance(node, { delay: 4200 }).show();
}

function setBusy(busy, message) {
  state.busy = busy;
  el('top-progress').classList.toggle('is-active', busy);
  el('dropzone').classList.toggle('is-busy', busy);
  el('btn-sample').disabled = busy || !state.modelReady;
  el('btn-replace').disabled = busy;
  if (message) el('dz-status').textContent = message;
}

// ---------------------------------------------------------------------------
// Campaign economics
// ---------------------------------------------------------------------------
function prepareProbabilities() {
  const p = Float64Array.from(state.customers.map((c) => c.churn_probability)).sort();
  const prefix = new Float64Array(p.length + 1);
  for (let i = 0; i < p.length; i += 1) prefix[i + 1] = prefix[i] + p[i];
  state.sortedP = p;
  state.prefix = prefix;
  state.curve = null;
}

function economics(tau) {
  const p = state.sortedP;
  const n = p.length;
  let lo = 0; let hi = n;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (p[mid] >= tau) hi = mid; else lo = mid + 1; }  // first p ≥ τ
  const targeted = n - lo;
  const sumBelow = state.prefix[lo];
  const sumAll = state.prefix[n];
  const campaign = targeted * state.cOffer;
  const missed = sumBelow * state.cLost;
  const doNothing = sumAll * state.cLost;
  const total = campaign + missed;
  return { tau, n, targeted, reached: sumAll - sumBelow, expectedChurners: sumAll, campaign, missed, total, doNothing, savings: doNothing - total };
}

function getCurve() {
  if (!state.curve) {
    const points = GRID.map(economics);
    const best = points.reduce((a, b) => (b.total < a.total - 1e-9 ? b : a));
    state.curve = { points, best, doNothing: points[0].doNothing };
  }
  return state.curve;
}
function breakEven() { return state.cLost > 0 ? clamp(state.cOffer / state.cLost, 0, 1) : 1; }

function setTau(value, { fromSlider = false } = {}) {
  state.tau = Math.round(clamp(value, 0, 1) * 100) / 100;
  if (!fromSlider) el('tau-slider').value = state.tau.toFixed(2);
  scheduleRender();
}

// ---------------------------------------------------------------------------
// Rendering — dashboard
// ---------------------------------------------------------------------------
function scheduleRender() {
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => {
    renderQueued = false;
    renderDashboard();
  });
}

function renderDashboard() {
  const loaded = state.customers.length > 0;
  el('empty-state').hidden = loaded;
  el('loaded-state').hidden = !loaded;
  if (!loaded) return;
  renderSource();
  renderPlanner();
  renderKpis();
  renderDistChart();
  renderCostChart();
  renderTable();
  if (state.drawerCustomer) {
    el('gauge').innerHTML = gaugeSvg(state.drawerCustomer);
    el('drawer-decision').innerHTML = decisionHtml(state.drawerCustomer);
  }
}

function tierCounts() {
  const counts = { High: 0, Medium: 0, Low: 0 };
  state.customers.forEach((c) => { counts[c.risk_tier] = (counts[c.risk_tier] || 0) + 1; });
  return counts;
}

function renderSource() {
  const b = state.batch;
  const n = state.customers.length;
  const counts = tierCounts();
  el('source-name').textContent = b.filename;
  const skipped = b.skipped ? ` · ${int(b.skipped)} row(s) skipped` : '';
  el('source-meta').textContent = `${int(n)} customers · logged as batch ${b.batchId.slice(0, 8)} · ${new Date(b.at).toLocaleTimeString()}${skipped}`;
  el('tier-bar').innerHTML = TIERS.filter((t) => counts[t] > 0)
    .map((t) => `<span class="seg-${t}" style="flex-grow:${counts[t]}"></span>`).join('');
  el('tier-bar').setAttribute('aria-label', TIERS.map((t) => `${t} ${counts[t]}`).join(', '));
  el('tier-legend').innerHTML = TIERS.map((t) => `<span>${icon(TIER_ICON[t], `tier-glyph ${t.toLowerCase()}`)}${t} <b>${counts[t]}</b> <span class="text-body-secondary">(${pct(counts[t] / n, 0)})</span></span>`).join('');
  el('export-count').textContent = int(economics(state.tau).targeted);
}

function renderPlanner() {
  const slider = el('tau-slider');
  el('tau-value').textContent = state.tau.toFixed(2);
  slider.style.setProperty('--pct', `${state.tau * 100}%`);
  slider.setAttribute('aria-valuetext', `τ ${state.tau.toFixed(2)}: ${economics(state.tau).targeted} customers targeted`);
  const presets = { recommended: state.tauStar, breakeven: breakEven(), batch: getCurve().best.tau };
  el('preset-rec').textContent = presets.recommended.toFixed(2);
  el('preset-be').textContent = presets.breakeven.toFixed(2);
  el('preset-batch').textContent = presets.batch.toFixed(2);
  document.querySelectorAll('.preset').forEach((btn) => {
    btn.setAttribute('aria-pressed', String(Math.abs(presets[btn.dataset.preset] - state.tau) < 0.005));
  });
  el('range-ticks').innerHTML = [['mk-rec', presets.recommended], ['mk-be', presets.breakeven], ['mk-batch', presets.batch]]
    .map(([cls, v]) => `<span class="range-tick" style="--v:${v}"><span class="mk ${cls}"></span></span>`).join('');
}

function renderKpis() {
  const e = economics(state.tau);
  const n = e.n;
  const savings = el('kpi-savings');
  savings.textContent = money(e.savings);
  savings.classList.toggle('is-negative', e.savings < 0);
  const share = e.doNothing > 0 ? e.savings / e.doNothing : 0;
  el('savings-meter-fill').style.width = `${clamp(share, 0, 1) * 100}%`;
  el('kpi-savings-sub').textContent = e.doNothing > 0
    ? `${pct(share)} cheaper than doing nothing (${money(e.doNothing)} expected loss)`
    : 'No churn expected in this batch';
  el('kpi-targeted').textContent = int(e.targeted);
  el('kpi-targeted-sub').textContent = `${pct(e.targeted / n, 0)} of customers · reaches ≈ ${e.reached.toFixed(1)} of ${e.expectedChurners.toFixed(1)} expected churners`;
  el('kpi-campaign').textContent = money(e.campaign);
  el('kpi-campaign-sub').textContent = `${int(e.targeted)} offers × ${money(state.cOffer)}`;
  el('kpi-missed').textContent = money(e.missed);
  el('kpi-missed-sub').textContent = `≈ ${(e.expectedChurners - e.reached).toFixed(1)} churners below τ × ${money(state.cLost)}`;
  el('kpi-total').textContent = money(e.total);
  el('kpi-total-sub').textContent = `vs ${money(e.doNothing)} if nobody is contacted`;
}

// ---- SVG chart helpers ----------------------------------------------------
function niceStep(maxValue, targetTicks = 3, integer = false) {
  const raw = Math.max(maxValue, 1e-9) / targetTicks;
  const pow = 10 ** Math.floor(Math.log10(raw));
  const f = raw / pow;
  const nice = f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10;
  const step = nice * pow;
  return integer ? Math.max(1, Math.round(step)) : step;
}
function roundedTopBar(x, y, w, h, r, cls) {
  if (h <= 0 || w <= 0) return '';
  const rr = Math.max(0, Math.min(r, w / 2, h));
  return `<path class="${cls}" d="M${x},${y + h}V${y + rr}Q${x},${y} ${x + rr},${y}H${x + w - rr}Q${x + w},${y} ${x + w},${y + rr}V${y + h}Z"/>`;
}
function chartWidth(host) { return Math.max(280, Math.floor(host.getBoundingClientRect().width)); }

function showTip(html, clientX, clientY) {
  const tip = el('chart-tooltip');
  tip.innerHTML = html;
  tip.hidden = false;
  const { width, height } = tip.getBoundingClientRect();
  const x = clientX + 14 + width > window.innerWidth ? clientX - width - 14 : clientX + 14;
  const y = clamp(clientY - height - 12, 8, window.innerHeight - height - 8);
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}
function hideTip() { el('chart-tooltip').hidden = true; }
function tipRow(label, value) { return `<div class="tt-row"><span>${label}</span><b>${value}</b></div>`; }

// ---- Distribution histogram ----------------------------------------------
function renderDistChart() {
  const host = el('dist-chart');
  const W = chartWidth(host);
  const H = 172;
  const m = { l: 36, r: 10, t: 24, b: 24 };
  const iw = W - m.l - m.r;
  const ih = H - m.t - m.b;
  const NB = 20;
  const bins = Array.from({ length: NB }, (_, i) => ({ i, lo: i / NB, hi: (i + 1) / NB, on: 0, off: 0 }));
  state.customers.forEach((c) => {
    const i = clamp(Math.floor(c.churn_probability * NB), 0, NB - 1);
    if (c.churn_probability >= state.tau) bins[i].on += 1; else bins[i].off += 1;
  });
  state.distBins = bins;
  const maxCount = Math.max(1, ...bins.map((b) => b.on + b.off));
  const step = niceStep(maxCount, 3, true);
  const top = Math.ceil(maxCount / step) * step;
  const x = (v) => m.l + v * iw;
  const y = (v) => m.t + ih - (v / top) * ih;
  const bw = iw / NB;
  const gap = Math.max(2, Math.min(4, bw * 0.2));

  let s = `<svg class="chart-svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`;
  for (let t = step; t <= top; t += step) {
    s += `<line class="grid-line" x1="${m.l}" x2="${W - m.r}" y1="${y(t)}" y2="${y(t)}"/>`;
    s += `<text x="${m.l - 7}" y="${y(t) + 3.5}" text-anchor="end">${t}</text>`;
  }
  [0, 0.25, 0.5, 0.75, 1].forEach((v) => { s += `<text x="${x(v)}" y="${H - 6}" text-anchor="middle">${Math.round(v * 100)}%</text>`; });
  bins.forEach((b) => {
    const bx = x(b.lo) + gap / 2;
    const w = bw - gap;
    const hOff = (b.off / top) * ih;
    const hOn = (b.on / top) * ih;
    let g = `<g class="bar-group" data-i="${b.i}">`;
    if (hOff > 0) g += roundedTopBar(bx, y(0) - hOff, w, hOff, hOn > 0 ? 0 : 4, 'bar-off');
    if (hOn > 0) {
      const base = y(0) - hOff - (hOff > 0 ? 2 : 0);          // 2px surface gap between stacked segments
      g += roundedTopBar(bx, base - hOn + (hOff > 0 ? 2 : 0), w, hOn - (hOff > 0 ? 2 : 0), 4, 'bar-on');
    }
    g += `<rect class="hit" x="${x(b.lo)}" y="${m.t}" width="${bw}" height="${ih}"/></g>`;
    s += g;
  });
  s += `<line class="baseline" x1="${m.l}" x2="${W - m.r}" y1="${y(0)}" y2="${y(0)}"/>`;
  const tx = x(state.tau);
  const tagW = 50;
  const tagX = clamp(tx - tagW / 2, m.l - 4, W - m.r - tagW);
  s += `<line class="tau-line" x1="${tx}" x2="${tx}" y1="${m.t - 4}" y2="${y(0)}"/>`;
  s += `<g class="tau-tag"><rect x="${tagX}" y="2" width="${tagW}" height="17" rx="6"/><text x="${tagX + tagW / 2}" y="14" text-anchor="middle">τ ${state.tau.toFixed(2)}</text></g>`;
  s += '</svg>';
  host.innerHTML = s;
  host.setAttribute('aria-label', `Histogram of ${state.customers.length} customers by churn probability; ${economics(state.tau).targeted} at or above τ ${state.tau.toFixed(2)} are targeted.`);
}

function onDistHover(event) {
  const group = event.target.closest('.bar-group');
  el('dist-chart').querySelectorAll('.bar-group.is-hover').forEach((g) => g.classList.remove('is-hover'));
  if (!group) { hideTip(); return; }
  group.classList.add('is-hover');
  const b = state.distBins[Number(group.dataset.i)];
  const total = b.on + b.off;
  showTip(`<div class="tt-title">${Math.round(b.lo * 100)}–${Math.round(b.hi * 100)}% churn risk</div>
    ${tipRow('Customers', int(total))}${tipRow('Targeted at τ', int(b.on))}${tipRow('Not targeted', int(b.off))}`,
  event.clientX, event.clientY);
}

// ---- Cost curve ------------------------------------------------------------
function renderCostChart() {
  const host = el('cost-chart');
  const curve = getCurve();
  const W = chartWidth(host);
  const H = 196;
  const m = { l: 58, r: 12, t: 14, b: 24 };
  const iw = W - m.l - m.r;
  const ih = H - m.t - m.b;
  const maxY = Math.max(curve.doNothing, ...curve.points.map((p) => p.total)) * 1.06 || 1;
  const step = niceStep(maxY, 3);
  const top = Math.ceil(maxY / step) * step;
  const x = (t) => m.l + t * iw;
  const y = (v) => m.t + ih - (v / top) * ih;
  state.costGeom = { m, iw, ih, x, y };

  const line = curve.points.map((p, i) => `${i ? 'L' : 'M'}${x(p.tau).toFixed(1)},${y(p.total).toFixed(1)}`).join('');
  const first = curve.points[0];
  const last = curve.points[curve.points.length - 1];
  let s = `<svg class="chart-svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`;
  for (let v = step; v <= top + 1e-9; v += step) {
    s += `<line class="grid-line" x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}"/>`;
    s += `<text x="${m.l - 8}" y="${y(v) + 3.5}" text-anchor="end">${moneyCompact(v)}</text>`;
  }
  [0, 0.25, 0.5, 0.75, 1].forEach((v) => { s += `<text x="${x(v)}" y="${H - 6}" text-anchor="middle">${v === 0 || v === 1 ? v : v.toFixed(2)}</text>`; });
  s += `<path class="cost-area" d="${line}L${x(last.tau)},${y(0)}L${x(first.tau)},${y(0)}Z"/>`;
  s += `<line class="ref-line" x1="${m.l}" x2="${W - m.r}" y1="${y(curve.doNothing)}" y2="${y(curve.doNothing)}"/>`;
  s += `<text class="ref-label" x="${W - m.r}" y="${y(curve.doNothing) - 5}" text-anchor="end">Do nothing ${moneyCompact(curve.doNothing)}</text>`;
  s += `<path class="cost-line" d="${line}"/>`;
  s += `<line class="baseline" x1="${m.l}" x2="${W - m.r}" y1="${y(0)}" y2="${y(0)}"/>`;
  const current = economics(state.tau);
  s += `<line class="tau-line" x1="${x(state.tau)}" x2="${x(state.tau)}" y1="${m.t}" y2="${y(0)}" stroke-opacity=".35"/>`;
  s += `<circle class="dot-best" cx="${x(curve.best.tau)}" cy="${y(curve.best.total)}" r="5"/>`;
  s += `<circle class="dot-current" cx="${x(state.tau)}" cy="${y(current.total)}" r="5.5"/>`;
  s += `<g id="cost-hover" visibility="hidden"><line class="crosshair" x1="0" x2="0" y1="${m.t}" y2="${y(0)}"/><circle class="hover-dot" r="4.5" cx="0" cy="0"/></g>`;
  s += `<rect class="hit" x="${m.l}" y="${m.t}" width="${iw}" height="${ih}"/>`;
  s += '</svg>';
  host.innerHTML = s;
  host.setAttribute('aria-label', `Expected total cost by threshold. Cheapest at τ ${curve.best.tau.toFixed(2)} (${money(curve.best.total)}); current τ ${state.tau.toFixed(2)} costs ${money(current.total)}; doing nothing costs ${money(curve.doNothing)}.`);
}

function costPointFromEvent(event) {
  const svg = el('cost-chart').querySelector('svg');
  if (!svg || !state.costGeom) return null;
  const { m, iw } = state.costGeom;
  const px = event.clientX - svg.getBoundingClientRect().left;
  const t = clamp(Math.round(((px - m.l) / iw) * 100), 1, 99);
  return getCurve().points[t - 1];
}
function onCostHover(event) {
  const p = costPointFromEvent(event);
  const hover = el('cost-hover');
  if (!p || !hover) return;
  const { x, y } = state.costGeom;
  hover.setAttribute('visibility', 'visible');
  hover.querySelector('line').setAttribute('x1', x(p.tau));
  hover.querySelector('line').setAttribute('x2', x(p.tau));
  hover.querySelector('circle').setAttribute('cx', x(p.tau));
  hover.querySelector('circle').setAttribute('cy', y(p.total));
  const isBest = p.tau === getCurve().best.tau;
  showTip(`<div class="tt-title">τ = ${p.tau.toFixed(2)}${isBest ? ' · cheapest' : ''}</div>
    ${tipRow('Customers targeted', int(p.targeted))}${tipRow('Offers', money(p.campaign))}
    ${tipRow('Missed churn', money(p.missed))}${tipRow('Total', money(p.total))}${tipRow('Savings', money(p.savings))}
    <div class="tt-hint">Click to use this threshold</div>`, event.clientX, event.clientY);
}
function onCostLeave() {
  const hover = el('cost-hover');
  if (hover) hover.setAttribute('visibility', 'hidden');
  hideTip();
}

// ---- Customer table --------------------------------------------------------
const SORTERS = {
  risk: (c) => c.churn_probability,
  tenure: (c) => c.key_services.tenure_months,
  monthly: (c) => c.key_services.monthly_charges,
};
const SORT_LABEL = { risk: 'churn risk', tenure: 'tenure', monthly: 'monthly charges' };

function currentView() {
  const q = state.search.trim().toLowerCase();
  const rows = state.customers.filter((c) => (state.tier === 'all' || c.risk_tier === state.tier)
    && (!state.onlyTargeted || c.churn_probability >= state.tau)
    && (!q || c.customer_id.toLowerCase().includes(q)));
  const value = SORTERS[state.sort.key];
  const sign = state.sort.dir === 'desc' ? -1 : 1;
  return rows.sort((a, b) => sign * (value(a) - value(b)) || a.rank - b.rank);
}

function rowHtml(c, index) {
  const k = c.key_services;
  const target = c.churn_probability >= state.tau;
  const internet = k.internet_service === 'No' ? 'No internet' : k.internet_service;
  const addons = k.add_ons.length ? `${k.add_ons.length} add-on${k.add_ons.length > 1 ? 's' : ''}` : 'no add-ons';
  const active = state.drawerCustomer && state.drawerCustomer.rank === c.rank;
  return `<tr data-idx="${index}" tabindex="0" class="${target ? 'is-target' : ''}${active ? ' is-active' : ''}"
      aria-label="${escapeHtml(c.customer_id)}, ${pct(c.churn_probability)} churn risk, ${escapeHtml(c.risk_tier)} tier${target ? ', offer' : ''}. Press Enter for details.">
    <td class="num rank">${c.rank}</td>
    <td><span class="cust-id mono">${escapeHtml(c.customer_id)}</span></td>
    <td><div class="risk"><span class="risk-track" aria-hidden="true"><span style="width:${(c.churn_probability * 100).toFixed(1)}%"></span><i style="left:${state.tau * 100}%"></i></span><span class="risk-val num">${pct(c.churn_probability)}</span></div></td>
    <td>${tierChip(c.risk_tier)}</td>
    <td class="plan"><span class="plan-main">${escapeHtml(k.contract)}</span><span class="plan-sub">${escapeHtml(internet)} · ${addons}</span></td>
    <td class="num col-tenure">${int(k.tenure_months)} mo</td>
    <td class="num col-monthly">${Number(k.monthly_charges).toFixed(2)}</td>
    <td>${target ? `<span class="action-chip">${icon('send')}Send offer</span>` : '<span class="action-none">Monitor</span>'}</td>
    <td class="open-cell">${icon('chev-right')}</td>
  </tr>`;
}

function renderTable() {
  state.view = currentView();
  const shown = state.view.slice(0, state.visible);
  const tbody = el('risk-tbody');
  tbody.innerHTML = shown.length
    ? shown.map(rowHtml).join('')
    : `<tr class="empty-row"><td colspan="9">No customers match these filters. <button type="button" class="btn btn-ghost btn-sm ms-2" data-action="clear-filters">Clear filters</button></td></tr>`;

  const counts = tierCounts();
  el('count-all').textContent = int(state.customers.length);
  TIERS.forEach((t) => { el(`count-${t}`).textContent = int(counts[t]); });
  const remaining = state.view.length - shown.length;
  el('btn-more').hidden = remaining <= 0;
  el('btn-more').textContent = `Show ${int(Math.min(PAGE_SIZE, remaining))} more`;
  const e = economics(state.tau);
  el('table-caption').textContent = `${int(state.view.length)} of ${int(state.customers.length)} customers · ${int(e.targeted)} get an offer at τ = ${state.tau.toFixed(2)} · sorted by ${SORT_LABEL[state.sort.key]}`;
  document.querySelectorAll('.risk-table th[data-sort]').forEach((th) => {
    th.setAttribute('aria-sort', th.dataset.sort === state.sort.key ? (state.sort.dir === 'desc' ? 'descending' : 'ascending') : 'none');
  });
}

// ---------------------------------------------------------------------------
// Customer drawer
// ---------------------------------------------------------------------------
function gaugeSvg(c) {
  const p = clamp(c.churn_probability, 0, 1);
  const cx = 110; const cy = 112; const r = 84; const sw = 14;
  const at = (v, radius = r) => { const a = Math.PI * (1 - v); return [cx + radius * Math.cos(a), cy - radius * Math.sin(a)]; };
  const [sx, sy] = at(0); const [ex, ey] = at(1); const [vx, vy] = at(Math.max(p, 0.002));
  const [t1x, t1y] = at(state.tau, r - 12); const [t2x, t2y] = at(state.tau, r + 12); const [lx, ly] = at(state.tau, r + 22);
  return `<svg viewBox="0 0 220 140" role="img" aria-label="Churn probability ${pct(p)}, threshold ${state.tau.toFixed(2)}">
    <path class="g-track" d="M${sx} ${sy}A${r} ${r} 0 0 1 ${ex} ${ey}" fill="none" stroke-width="${sw}" stroke-linecap="round"/>
    <path class="g-value-${escapeHtml(c.risk_tier)}" d="M${sx} ${sy}A${r} ${r} 0 0 1 ${vx.toFixed(2)} ${vy.toFixed(2)}" fill="none" stroke-width="${sw}" stroke-linecap="round"/>
    <line class="g-tau" x1="${t1x.toFixed(2)}" y1="${t1y.toFixed(2)}" x2="${t2x.toFixed(2)}" y2="${t2y.toFixed(2)}"/>
    <text class="g-tau-label" x="${lx.toFixed(2)}" y="${(ly + 3).toFixed(2)}" text-anchor="middle">τ</text>
    <text class="g-pct" x="110" y="104" text-anchor="middle">${pct(p)}</text>
    <text class="g-label" x="110" y="128" text-anchor="middle">churn probability</text>
  </svg>`;
}

function decisionHtml(c) {
  const target = c.churn_probability >= state.tau;
  const ev = c.churn_probability * state.cLost - state.cOffer;
  let note = '';
  if (target && ev < 0) note = 'In expectation this offer costs more than it saves — τ is below the break-even point.';
  if (!target && ev > 0) note = `An offer would still be worth ${money(ev)} in expectation — lower τ to include this customer.`;
  return `<div class="decision ${target ? 'is-target' : ''}">
    <span class="decision-icon">${icon(target ? 'send' : 'clock')}</span>
    <div>
      <div class="decision-title">${target ? 'Send a retention offer' : 'No offer — keep monitoring'}</div>
      <div class="decision-sub">At τ = ${state.tau.toFixed(2)} · expected value of an offer = ${pct(c.churn_probability)} × ${money(state.cLost)} − ${money(state.cOffer)} = <b>${money(ev)}</b></div>
      ${note ? `<div class="decision-note">${escapeHtml(note)}</div>` : ''}
    </div>
  </div>`;
}

function driversHtml(drivers) {
  if (!drivers || !drivers.length) return '<p class="fine-print">No drivers returned.</p>';
  const maxAbs = Math.max(...drivers.map((d) => Math.abs(d.impact_pp)), 0.01);
  return drivers.map((d) => {
    const up = d.impact_pp > 0;
    const width = ((Math.abs(d.impact_pp) / maxAbs) * 50).toFixed(1);   // largest driver fills half the track
    return `<div class="driver-row">
      <div class="driver-label"><span class="driver-feature">${escapeHtml(d.feature)}</span><span class="driver-value" title="${escapeHtml(d.value)}">${escapeHtml(d.value)}</span></div>
      <div class="driver-track" aria-hidden="true"><span class="driver-bar ${up ? 'up' : 'down'}" style="${up ? 'left' : 'right'}:50%;width:${width}%"></span></div>
      <div class="driver-impact ${up ? 'up' : 'down'}">${up ? '+' : '−'}${Math.abs(d.impact_pp).toFixed(1)} pp</div>
    </div>`;
  }).join('');
}

/** Map risk-raising drivers to retention talking points (only actionable features). */
const PLAYBOOK = {
  Contract: (v) => (v === 'Month-to-month' ? 'Offer a discounted 1- or 2-year contract.' : null),
  tenure: () => 'Early-life customer — schedule an onboarding check-in call.',
  TechSupport: (v) => (v === 'No' ? 'Bundle free tech support for three months.' : null),
  OnlineSecurity: (v) => (v === 'No' ? 'Offer a free online-security trial.' : null),
  OnlineBackup: (v) => (v === 'No' ? 'Include online backup as a loyalty perk.' : null),
  DeviceProtection: (v) => (v === 'No' ? 'Offer device protection at a discount.' : null),
  PaymentMethod: (v) => (v === 'Electronic check' ? 'Nudge towards automatic payment with a small bill credit.' : null),
  InternetService: (v) => (v === 'Fiber optic' ? 'Check fibre service quality and whether the price is competitive.' : null),
  MonthlyCharges: () => 'Review the plan price — a targeted loyalty discount may help.',
  StreamingTV: (v) => (v === 'Yes' ? 'Review the value of the streaming bundle.' : null),
  StreamingMovies: (v) => (v === 'Yes' ? 'Review the value of the streaming bundle.' : null),
};
function actionsHtml(drivers) {
  const seen = new Set();
  const items = [];
  (drivers || []).filter((d) => d.impact_pp > 0).forEach((d) => {
    const rule = PLAYBOOK[d.feature];
    const text = rule ? rule(d.value) : null;
    if (text && !seen.has(text)) {
      seen.add(text);
      items.push(`<li>${icon('check')}<div>${escapeHtml(text)}<small>because ${escapeHtml(d.feature)} = ${escapeHtml(d.value)} (+${d.impact_pp.toFixed(1)} pp)</small></div></li>`);
    }
  });
  return items.length ? items.join('') : `<li>${icon('info')}<div>No actionable risk drivers in the top five — standard relationship management is enough.</div></li>`;
}

const PROFILE_GROUPS = [
  ['Account', [['tenure', 'Tenure (months)'], ['Contract', 'Contract'], ['PaymentMethod', 'Payment'], ['PaperlessBilling', 'Paperless billing'], ['MonthlyCharges', 'Monthly charges'], ['TotalCharges', 'Total charges']]],
  ['Services', [['PhoneService', 'Phone'], ['MultipleLines', 'Multiple lines'], ['InternetService', 'Internet'], ['OnlineSecurity', 'Online security'], ['OnlineBackup', 'Online backup'], ['DeviceProtection', 'Device protection'], ['TechSupport', 'Tech support'], ['StreamingTV', 'Streaming TV'], ['StreamingMovies', 'Streaming movies']]],
  ['Household', [['gender', 'Gender'], ['SeniorCitizen', 'Senior citizen'], ['Partner', 'Partner'], ['Dependents', 'Dependents']]],
];
function profileHtml(record) {
  return PROFILE_GROUPS.map(([title, fields]) => `<div><h4>${title}</h4><dl>${fields.map(([key, label]) => {
    let value = record[key];
    if ((key === 'MonthlyCharges' || key === 'TotalCharges') && value !== null && value !== undefined) value = Number(value).toFixed(2);
    return `<div><dt>${label}</dt><dd>${escapeHtml(value ?? '—')}</dd></div>`;
  }).join('')}</dl></div>`).join('');
}

function skeletonDrivers() {
  return Array.from({ length: 5 }, () => `<div class="driver-row"><span class="skeleton" style="width:80%"></span><span class="skeleton" style="width:100%"></span><span class="skeleton" style="width:70%"></span></div>`).join('');
}

async function openDrawer(index) {
  const c = state.view[index];
  if (!c) return;
  state.drawerIndex = index;
  state.drawerCustomer = c;
  el('drawer-title').textContent = c.customer_id;
  el('drawer-pos').textContent = `${index + 1} / ${state.view.length}`;
  el('drawer-prev').disabled = index <= 0;
  el('drawer-next').disabled = index >= state.view.length - 1;
  el('gauge').innerHTML = gaugeSvg(c);
  el('drawer-tier').innerHTML = tierChip(c.risk_tier);
  el('drawer-base').innerHTML = '&nbsp;';
  el('drawer-decision').innerHTML = decisionHtml(c);
  el('profile').innerHTML = profileHtml(c.record);
  document.querySelectorAll('#risk-tbody tr.is-active').forEach((tr) => tr.classList.remove('is-active'));
  const row = document.querySelector(`#risk-tbody tr[data-idx="${index}"]`);
  if (row) { row.classList.add('is-active'); row.scrollIntoView({ block: 'nearest' }); }
  drawer.show();

  const key = `${state.batch.batchId}|${c.rank}`;
  let result = state.explainCache.get(key);
  if (!result) {
    el('drivers').innerHTML = skeletonDrivers();
    el('actions').innerHTML = '';
    try {
      result = await api('/api/explain', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ record: c.record, top_k: 5 }),
      });
      state.explainCache.set(key, result);
    } catch (error) {
      if (state.drawerCustomer !== c) return;
      el('drivers').innerHTML = `<div class="app-alert is-error">${icon('high')}<div class="alert-body">Could not compute drivers: ${escapeHtml(error.message)}</div></div>`;
      return;
    }
  }
  if (state.drawerCustomer !== c) return;                // the user moved on while we were waiting
  const diff = c.churn_probability - result.base_value;
  el('drawer-base').innerHTML = `Average customer: <b>${pct(result.base_value)}</b><br>This customer is <b>${diff >= 0 ? '+' : '−'}${Math.abs(diff * 100).toFixed(1)} pp</b> ${diff >= 0 ? 'above' : 'below'} average`;
  el('drivers').innerHTML = driversHtml(result.drivers);
  el('actions').innerHTML = actionsHtml(result.drivers);
}

function stepDrawer(delta) {
  const next = state.drawerIndex + delta;
  if (next < 0 || next >= state.view.length) return;
  if (next >= state.visible) { state.visible = next + 1; renderTable(); }
  openDrawer(next);
}

// ---------------------------------------------------------------------------
// Upload, sample data, export
// ---------------------------------------------------------------------------
function applyResult(result, filename) {
  state.customers = result.customers;
  state.batch = { filename, batchId: result.batch_id, at: Date.now(), skipped: (result.cleaning?.dropped_rows || []).length };
  state.visible = PAGE_SIZE;
  state.explainCache.clear();
  state.drawerCustomer = null;
  prepareProbabilities();
  showView('dashboard');
  renderDashboard();
}

async function uploadFile(file) {
  if (!file || state.busy) return;
  if (!state.modelReady) { showAlert('The model is not loaded yet — see the message above.', 'warning'); return; }
  if (!/\.csv$/i.test(file.name)) { showAlert(`"${file.name}" is not a .csv file.`, 'warning'); return; }
  clearAlerts();
  setBusy(true, `Scoring ${file.name}…`);
  const form = new FormData();
  form.append('file', file);
  form.append('threshold', state.tau.toFixed(2));          // logged as threshold_used
  try {
    const result = await api('/api/predict', { method: 'POST', body: form });
    applyResult(result, file.name);
    toast(`${int(result.n_customers)} customers scored · logged as batch ${result.batch_id.slice(0, 8)}`);
    (result.warnings || []).forEach((w) => showAlert(w, 'warning'));
    loadLogs({ quiet: true });
  } catch (error) {
    const detail = error.body && error.body.detail;
    const message = detail && detail.missing_columns ? `missing column(s) ${detail.missing_columns.join(', ')}` : error.message;
    showAlert(`Upload failed: ${message}`, 'error');
  } finally {
    setBusy(false, 'CSV up to 10 MB · customerID optional · a Churn column is ignored');
    el('file-input').value = '';
    el('replace-input').value = '';
  }
}

async function loadSample() {
  if (state.busy) return;
  try {
    const response = await fetch('/api/sample-csv');
    if (!response.ok) throw new Error(`sample file unavailable (HTTP ${response.status})`);
    const blob = await response.blob();
    await uploadFile(new File([blob], 'sample_demo.csv', { type: 'text/csv' }));
  } catch (error) {
    showAlert(`Could not load the sample: ${error.message}`, 'error');
  }
}

function csvCell(value) {
  let text = String(value ?? '');
  if (/^[=+\-@\t\r]/.test(text)) text = `'${text}`;            // stop spreadsheet formula injection
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}
function exportTargets() {
  const rows = state.customers.filter((c) => c.churn_probability >= state.tau);
  if (!rows.length) { showAlert('No customers are targeted at this threshold — lower τ to export a list.', 'warning'); return; }
  const header = ['rank', 'customerID', 'churn_probability', 'risk_tier', 'threshold', 'expected_value_of_offer', 'contract', 'internet_service', 'tenure_months', 'monthly_charges', 'payment_method'];
  const lines = rows.map((c) => [c.rank, c.customer_id, c.churn_probability.toFixed(4), c.risk_tier, state.tau.toFixed(2),
    Math.round(c.churn_probability * state.cLost - state.cOffer), c.key_services.contract, c.key_services.internet_service,
    c.key_services.tenure_months, c.key_services.monthly_charges, c.key_services.payment_method].map(csvCell).join(','));
  const blob = new Blob([[header.join(','), ...lines].join('\n')], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `retention_targets_tau_${state.tau.toFixed(2)}.csv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  toast(`Exported ${int(rows.length)} customers to ${a.download}`);
}

// ---------------------------------------------------------------------------
// Logs view
// ---------------------------------------------------------------------------
function statTile(iconName, label, value, sub, subClass = '') {
  return `<div class="stat"><div class="stat-top"><span class="kpi-icon">${icon(iconName)}</span>${label}</div>
    <div class="stat-value">${value}</div><div class="stat-sub ${subClass}">${sub}</div></div>`;
}

async function loadLogs({ quiet = false } = {}) {
  const refresh = el('refresh-icon');
  if (!quiet) refresh.style.animation = 'spin 0.8s linear infinite';
  try {
    const { records, stats } = await api('/api/logs?limit=500');
    state.logs.records = records;
    state.logs.stats = stats;
    const count = el('nav-log-count');
    count.hidden = !stats.total_predictions;
    count.textContent = stats.total_predictions > 999 ? `${Math.floor(stats.total_predictions / 1000)}k` : String(stats.total_predictions);
    renderLogs();
  } catch (error) {
    if (!quiet) showAlert(`Could not load the prediction log: ${error.message}`, 'error');
  } finally {
    refresh.style.animation = '';
  }
}

function askClearLogs() {
  const total = (state.logs.stats && state.logs.stats.total_predictions) || 0;
  el('clear-log-text').innerHTML = `This permanently deletes <b>${int(total)}</b> logged prediction${total === 1 ? '' : 's'} from <code>predictions.jsonl</code> and the SQLite mirror. It cannot be undone.`;
  bootstrap.Modal.getOrCreateInstance(el('clear-log-modal')).show();
}

async function clearLogs() {
  const button = el('btn-confirm-clear');
  button.disabled = true;
  try {
    const { removed } = await api('/api/logs', { method: 'DELETE' });
    bootstrap.Modal.getOrCreateInstance(el('clear-log-modal')).hide();
    toast(`Prediction log cleared · ${int(removed)} record${removed === 1 ? '' : 's'} removed`);
    await loadLogs({ quiet: true });
  } catch (error) {
    bootstrap.Modal.getOrCreateInstance(el('clear-log-modal')).hide();
    showAlert(`Could not clear the prediction log: ${error.message}`, 'error');
  } finally {
    button.disabled = false;
  }
}

function renderLogs() {
  const { records, stats } = state.logs;
  if (!stats) return;
  const clearButton = el('btn-clear-logs');
  clearButton.hidden = stats.clear_enabled === false;          // deployment may forbid clearing
  clearButton.disabled = !stats.total_predictions;
  const tiers = stats.by_tier || {};
  const total = stats.total_predictions || 0;
  const latest = records[0];
  el('log-stats').innerHTML = [
    statTile('list', 'Predictions logged', int(total), `${(stats.backends || []).join(' + ') || '—'} backend`),
    statTile('file', 'Batches scored', int(stats.batches || 0), 'one batch per upload or API call'),
    statTile('high', 'High-risk share', total ? pct((tiers.High || 0) / total) : '—', `${int(tiers.High || 0)} High · ${int(tiers.Medium || 0)} Medium · ${int(tiers.Low || 0)} Low`),
    statTile('clock', 'Last prediction', latest ? relTime(latest.timestamp) : '—', latest ? escapeHtml(latest.model_version || '') : 'nothing logged yet'),
  ].join('');
  const q = state.logs.search.trim().toLowerCase();
  const rows = records.filter((r) => (state.logs.tier === 'all' || r.risk_tier === state.logs.tier)
    && (!q || String(r.customer_id).toLowerCase().includes(q)));
  el('log-caption').textContent = `${int(rows.length)} of the latest ${int(records.length)} records`;
  el('log-tbody').innerHTML = rows.length ? rows.slice(0, 300).map((r) => `<tr>
      <td class="text-nowrap" title="${escapeHtml(new Date(r.timestamp).toLocaleString())}">${escapeHtml(relTime(r.timestamp))}</td>
      <td><span class="mono">${escapeHtml(r.customer_id)}</span></td>
      <td class="text-nowrap"><span class="mini-bar" aria-hidden="true"><span style="width:${(r.churn_probability * 100).toFixed(1)}%"></span></span><span class="num">${pct(r.churn_probability)}</span></td>
      <td>${tierChip(r.risk_tier)}</td>
      <td class="num">${Number(r.threshold_used).toFixed(2)}</td>
      <td>${r.flagged_for_offer ? `<span class="action-chip">${icon('send')}Offer</span>` : '<span class="action-none">—</span>'}</td>
      <td><span class="source-tag">${escapeHtml((r.source || '').replace('_', ' '))}</span></td>
      <td><span class="mono text-body-secondary">${escapeHtml((r.batch_id || '').slice(0, 8))}</span></td>
      <td><span class="mono text-body-secondary">${escapeHtml(r.model_version || '')}</span></td>
    </tr>`).join('') : `<tr class="empty-row"><td colspan="9">${records.length ? 'No log records match these filters.' : 'No predictions logged yet — score a CSV on the dashboard.'}</td></tr>`;
}

// ---------------------------------------------------------------------------
// Model card
// ---------------------------------------------------------------------------
function renderModel(cfg) {
  const t = cfg.test_metrics || {};
  const has = t.pr_auc !== undefined;
  const delta = has ? t.pr_auc - t.logreg_pr_auc : 0;
  el('metric-tiles').innerHTML = has ? [
    statTile('trend-up', 'PR-AUC (test)', Number(t.pr_auc).toFixed(3), `Logistic regression ${Number(t.logreg_pr_auc).toFixed(3)} · Δ ${delta >= 0 ? '+' : '−'}${Math.abs(delta).toFixed(3)}`),
    statTile('gauge', 'ROC-AUC (test)', Number(t.roc_auc).toFixed(3), 'random = 0.500'),
    statTile('target', 'Recall at τ*', pct(t.recall), 'of real churners get an offer', 'is-good'),
    statTile('send', 'Precision at τ*', pct(t.precision), 'of offers reach a churner'),
    statTile('sigma', 'F1 at τ*', Number(t.f1).toFixed(3), 'cost-optimal, not F1-optimal'),
    statTile('users', 'Test customers', int(t.n_test), 'held out, evaluated once'),
  ].join('') : statTile('info', 'Test metrics', '—', 'Run notebook 07 to add them');

  const s = cfg.strategy || {};
  const a = cfg.architecture || {};
  const th = cfg.threshold || {};
  const c = cfg.calibration || {};
  const rows = [
    ['Model version', `<span class="mono">${escapeHtml(cfg.model_version)}</span>`],
    ['Architecture', `MLP ${a.input_dim} → ${(a.hidden_dims || []).join(' → ')} → 1 · BatchNorm · ReLU · Dropout ${a.dropout}`],
    ['Imbalance strategy', escapeHtml(s.label || '—')],
    ['Why this one', escapeHtml(s.selection_rule || '—')],
    ['Calibration', c.method ? `Platt: p = σ(${Number(c.a).toFixed(3)}·z ${Number(c.b) < 0 ? '−' : '+'} ${Math.abs(Number(c.b)).toFixed(3)})` : '—'],
    ['Recommended τ*', th.optimal !== undefined ? `${Number(th.optimal).toFixed(2)} — minimum validation cost` : '—'],
    ['Theoretical τ', th.theoretical !== undefined ? `${Number(th.theoretical).toFixed(2)} = C_offer / C_lost` : '—'],
    ['F1-optimal τ (val)', th.f1_optimal_val !== undefined ? Number(th.f1_optimal_val).toFixed(2) : '—'],
    ['Business costs', `Offer ${money(cfg.costs.c_offer)} · lost churner ${money(cfg.costs.c_lost)}`],
    ['Risk tiers', `High &gt; ${pct(cfg.risk_tiers.high, 0)} · Medium ${pct(cfg.risk_tiers.medium, 0)}–${pct(cfg.risk_tiers.high, 0)} · Low &lt; ${pct(cfg.risk_tiers.medium, 0)}`],
  ];
  el('model-details').innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
}

// ---------------------------------------------------------------------------
// Views, theme, config
// ---------------------------------------------------------------------------
const VIEWS = ['dashboard', 'logs', 'model'];
function showView(name, { focus = false } = {}) {
  if (!VIEWS.includes(name)) name = 'dashboard';
  VIEWS.forEach((v) => {
    const btn = el(`nav-${v}`);
    const active = v === name;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-selected', String(active));
    btn.tabIndex = active ? 0 : -1;
    el(`view-${v}`).hidden = !active;
  });
  if (focus) el(`nav-${name}`).focus();
  history.replaceState(null, '', name === 'dashboard' ? location.pathname : `#${name}`);
  if (name === 'logs') loadLogs();
  if (name === 'dashboard') scheduleRender();
}

const THEME_META = {
  auto: { icon: 'auto', label: 'Theme: automatic (follows your system)' },
  light: { icon: 'sun', label: 'Theme: light' },
  dark: { icon: 'moon', label: 'Theme: dark' },
};
const darkQuery = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
function applyTheme(pref) {
  const dark = pref === 'dark' || (pref === 'auto' && darkQuery && darkQuery.matches);
  document.documentElement.setAttribute('data-bs-theme', dark ? 'dark' : 'light');
  document.documentElement.setAttribute('data-theme-pref', pref);
  const meta = THEME_META[pref];
  el('theme-icon').setAttribute('href', `#i-${meta.icon}`);
  el('theme-toggle').setAttribute('aria-label', meta.label);
  el('theme-toggle').title = meta.label;
}
function cycleTheme() {
  const order = ['auto', 'light', 'dark'];
  const current = document.documentElement.getAttribute('data-theme-pref') || 'auto';
  const next = order[(order.indexOf(current) + 1) % order.length];
  try { localStorage.setItem(THEME_KEY, next); } catch (_) { /* storage blocked: theme still applies for this visit */ }
  applyTheme(next);
}

function applyConfig(cfg) {
  state.config = cfg;
  state.modelReady = true;
  state.tauStar = Number(cfg.threshold.optimal);
  state.tau = state.tauStar;
  state.cOffer = Number(cfg.costs.c_offer);
  state.cLost = Number(cfg.costs.c_lost);
  state.currency = cfg.costs.currency || '₹';
  el('tau-slider').value = state.tau.toFixed(2);
  el('c-offer').value = state.cOffer;
  el('c-lost').value = state.cLost;
  document.querySelectorAll('.currency').forEach((node) => { node.textContent = state.currency; });
  el('required-columns').textContent = (cfg.required_columns || []).join(', ');
  const chip = el('model-status');
  chip.className = 'status-chip is-ready';
  const [strategy, hash] = String(cfg.model_version).split('-');
  el('model-status-text').textContent = `Model ready · ${strategy.replace('_', ' ')} · ${hash || ''}`;
  chip.title = `Deployed model ${cfg.model_version}`;
  renderModel(cfg);
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------
function readCost(input, key) {
  const value = Number(input.value);
  const valid = input.value !== '' && Number.isFinite(value) && value >= 0 && !(key === 'cLost' && value === 0 && state.cOffer === 0);
  input.closest('.input-affix').classList.toggle('is-invalid', !valid);
  if (!valid) return;
  state[key] = value;
  state.curve = null;
  scheduleRender();
}

function bindEvents() {
  // Navigation (tabs with arrow-key support).
  document.querySelectorAll('.nav-btn').forEach((btn) => btn.addEventListener('click', () => showView(btn.dataset.view)));
  document.querySelector('.app-nav').addEventListener('keydown', (event) => {
    if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
    const current = VIEWS.findIndex((v) => el(`nav-${v}`).classList.contains('active'));
    const next = (current + (event.key === 'ArrowRight' ? 1 : VIEWS.length - 1)) % VIEWS.length;
    showView(VIEWS[next], { focus: true });
  });
  // Deep links: react when only the #hash changes (e.g. a pasted /#logs link on an open page).
  window.addEventListener('hashchange', () => {
    const target = (location.hash || '').replace('#', '') || 'dashboard';
    if (VIEWS.includes(target) && el(`view-${target}`).hidden) showView(target);
  });
  el('theme-toggle').addEventListener('click', cycleTheme);
  if (darkQuery) darkQuery.addEventListener('change', () => {
    if ((document.documentElement.getAttribute('data-theme-pref') || 'auto') === 'auto') applyTheme('auto');
  });

  // Upload: file pickers, sample, page-wide drag & drop.
  el('file-input').addEventListener('change', (event) => uploadFile(event.target.files[0]));
  el('replace-input').addEventListener('change', (event) => uploadFile(event.target.files[0]));
  el('btn-sample').addEventListener('click', loadSample);
  el('btn-replace').addEventListener('click', () => el('replace-input').click());
  el('btn-export').addEventListener('click', exportTargets);
  let dragDepth = 0;
  const hasFiles = (event) => event.dataTransfer && Array.from(event.dataTransfer.types || []).includes('Files');
  window.addEventListener('dragenter', (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    dragDepth += 1;
    el('drop-overlay').hidden = false;
  });
  window.addEventListener('dragover', (event) => { if (hasFiles(event)) event.preventDefault(); });
  window.addEventListener('dragleave', (event) => {
    if (!hasFiles(event)) return;
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) el('drop-overlay').hidden = true;
  });
  window.addEventListener('drop', (event) => {
    event.preventDefault();
    dragDepth = 0;
    el('drop-overlay').hidden = true;
    const file = event.dataTransfer && event.dataTransfer.files[0];
    if (file) uploadFile(file);
  });

  // Planner.
  el('tau-slider').addEventListener('input', (event) => setTau(Number(event.target.value), { fromSlider: true }));
  document.querySelectorAll('.preset').forEach((btn) => btn.addEventListener('click', () => {
    const value = { recommended: state.tauStar, breakeven: breakEven(), batch: getCurve().best.tau }[btn.dataset.preset];
    setTau(value);
  }));
  el('c-offer').addEventListener('input', (event) => readCost(event.target, 'cOffer'));
  el('c-lost').addEventListener('input', (event) => readCost(event.target, 'cLost'));

  // Charts.
  const dist = el('dist-chart');
  dist.addEventListener('mousemove', onDistHover);
  dist.addEventListener('mouseleave', () => { hideTip(); dist.querySelectorAll('.is-hover').forEach((g) => g.classList.remove('is-hover')); });
  const cost = el('cost-chart');
  cost.addEventListener('mousemove', onCostHover);
  cost.addEventListener('mouseleave', onCostLeave);
  cost.addEventListener('click', (event) => { const p = costPointFromEvent(event); if (p) setTau(p.tau); });
  if (window.ResizeObserver) {
    let last = 0;
    new ResizeObserver((entries) => {
      const width = Math.round(entries[0].contentRect.width);
      if (width !== last) { last = width; if (state.customers.length) { renderDistChart(); renderCostChart(); } }
    }).observe(el('insights'));
  }

  // Table: filters, sorting, paging, opening customers.
  el('search').addEventListener('input', (event) => { state.search = event.target.value; state.visible = PAGE_SIZE; scheduleRender(); });
  el('tier-filter').addEventListener('click', (event) => {
    const btn = event.target.closest('button[data-tier]');
    if (!btn) return;
    state.tier = btn.dataset.tier;
    state.visible = PAGE_SIZE;
    el('tier-filter').querySelectorAll('button').forEach((b) => { b.classList.toggle('active', b === btn); b.setAttribute('aria-pressed', String(b === btn)); });
    scheduleRender();
  });
  el('only-targeted').addEventListener('change', (event) => { state.onlyTargeted = event.target.checked; state.visible = PAGE_SIZE; scheduleRender(); });
  document.querySelectorAll('.risk-table th[data-sort] .sort-btn').forEach((btn) => btn.addEventListener('click', () => {
    const key = btn.closest('th').dataset.sort;
    const defaultDir = key === 'tenure' ? 'asc' : 'desc';
    state.sort = state.sort.key === key ? { key, dir: state.sort.dir === 'desc' ? 'asc' : 'desc' } : { key, dir: defaultDir };
    scheduleRender();
  }));
  el('btn-more').addEventListener('click', () => { state.visible += PAGE_SIZE; renderTable(); });
  const tbody = el('risk-tbody');
  const openFromRow = (event) => {
    if (event.target.closest('[data-action="clear-filters"]')) {
      state.search = ''; state.tier = 'all'; state.onlyTargeted = false;
      el('search').value = ''; el('only-targeted').checked = false;
      el('tier-filter').querySelectorAll('button').forEach((b) => { const on = b.dataset.tier === 'all'; b.classList.toggle('active', on); b.setAttribute('aria-pressed', String(on)); });
      scheduleRender();
      return;
    }
    const row = event.target.closest('tr[data-idx]');
    if (!row) return;
    lastFocusedRow = row;
    openDrawer(Number(row.dataset.idx));
  };
  tbody.addEventListener('click', openFromRow);
  tbody.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); openFromRow(event); }
  });

  // Drawer.
  el('drawer-prev').addEventListener('click', () => stepDrawer(-1));
  el('drawer-next').addEventListener('click', () => stepDrawer(1));
  el('customer-drawer').addEventListener('keydown', (event) => {
    if (event.target.matches('input, textarea, select')) return;
    if (event.key === 'ArrowDown' || event.key === 'j') { event.preventDefault(); stepDrawer(1); }
    if (event.key === 'ArrowUp' || event.key === 'k') { event.preventDefault(); stepDrawer(-1); }
  });
  el('customer-drawer').addEventListener('hidden.bs.offcanvas', () => {
    state.drawerCustomer = null;
    state.drawerIndex = -1;
    document.querySelectorAll('#risk-tbody tr.is-active').forEach((tr) => tr.classList.remove('is-active'));
    const row = lastFocusedRow && document.body.contains(lastFocusedRow) ? lastFocusedRow : null;
    if (row) row.focus({ preventScroll: true });
  });

  // Logs.
  el('btn-refresh-logs').addEventListener('click', () => loadLogs());
  el('btn-clear-logs').addEventListener('click', askClearLogs);
  el('btn-confirm-clear').addEventListener('click', clearLogs);
  el('log-search').addEventListener('input', (event) => { state.logs.search = event.target.value; renderLogs(); });
  el('log-tier-filter').addEventListener('click', (event) => {
    const btn = event.target.closest('button[data-tier]');
    if (!btn) return;
    state.logs.tier = btn.dataset.tier;
    el('log-tier-filter').querySelectorAll('button').forEach((b) => { b.classList.toggle('active', b === btn); b.setAttribute('aria-pressed', String(b === btn)); });
    renderLogs();
  });
}

async function init() {
  drawer = bootstrap.Offcanvas.getOrCreateInstance(el('customer-drawer'));
  applyTheme(document.documentElement.getAttribute('data-theme-pref') || 'auto');
  bindEvents();
  const initialView = (location.hash || '').replace('#', '');
  showView(VIEWS.includes(initialView) ? initialView : 'dashboard');
  try {
    applyConfig(await api('/api/config'));
    setBusy(false);
  } catch (error) {
    state.modelReady = false;
    const chip = el('model-status');
    chip.className = 'status-chip is-error';
    el('model-status-text').textContent = 'Model not loaded';
    el('dropzone').classList.add('is-disabled');
    el('file-input').disabled = true;
    el('btn-sample').disabled = true;
    showAlert(error.status === 503 ? `Model not ready: ${error.message}` : `Could not reach the API: ${error.message}`, 'warning');
  }
  loadLogs({ quiet: true });
}

document.addEventListener('DOMContentLoaded', init);
