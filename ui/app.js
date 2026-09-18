// app.js - all UI logic for the Explainability Demo. Plain JS, no
// framework/build step - see CLAUDE.md/README for why (matches this
// project's lightweight style; ui/ has no build tooling to run).

let CATALOG = null;
let POLL_TIMER = null;

// ---------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatNumber(value) {
  if (typeof value !== "number") return escapeHtml(value);
  if (Number.isInteger(value)) return String(value);
  if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
  return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

function formatValue(value) {
  if (value === null || value === undefined) return "&mdash;";
  if (typeof value === "number") return formatNumber(value);
  if (typeof value === "object") return escapeHtml(JSON.stringify(value));
  return escapeHtml(value);
}

function kvTable(obj) {
  if (!obj || Object.keys(obj).length === 0) return '<p class="model-unavailable">No data available.</p>';
  const rows = Object.entries(obj)
    .map(([k, v]) => `<tr><td>${escapeHtml(k)}</td><td>${formatValue(v)}</td></tr>`)
    .join("");
  return `<table class="kv-table">${rows}</table>`;
}

async function apiGet(path) {
  const r = await fetch(`${API_BASE}${path}`);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `${r.status} ${r.statusText}`);
  return r.json();
}

async function apiPost(path, body) {
  const r = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `${r.status} ${r.statusText}`);
  return r.json();
}

function formatTimestamp(iso) {
  if (!iso) return "&mdash;";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return escapeHtml(iso);
  }
}

// ---------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------

function setupTabs() {
  document.querySelectorAll(".tab-button").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-button").forEach((b) => b.classList.remove("is-active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("is-active"));
      btn.classList.add("is-active");
      document.getElementById(`tab-${btn.dataset.tab}`).classList.add("is-active");
    });
  });
}

// ---------------------------------------------------------------------
// Models tab
// ---------------------------------------------------------------------

function renderSignature(describe) {
  if (describe.modality === "text") {
    const cats = (describe.categories || []).join(", ");
    return `
      <div class="section-label">Input</div>
      <p style="font-size:0.86rem; margin:4px 0;">${escapeHtml(describe.input_field?.name)} (${escapeHtml(describe.input_field?.field_type)}) &mdash; ${escapeHtml(describe.input_field?.description)}</p>
      <div class="section-label">Categories</div>
      <p style="font-size:0.86rem; margin:4px 0;">${escapeHtml(cats)}</p>`;
  }

  const rows = (describe.features || []).map((f) => {
    if (f.field_type === "numeric") {
      const s = f.stats || {};
      return `<tr><td>${escapeHtml(f.name)}</td><td>numeric &mdash; mean ${formatNumber(s.mean)}, std ${formatNumber(s.std)}${s.min !== undefined ? `, range [${formatNumber(s.min)}, ${formatNumber(s.max)}]` : ""}</td></tr>`;
    }
    return `<tr><td>${escapeHtml(f.name)}</td><td>categorical &mdash; ${escapeHtml((f.categories || []).join(", "))}</td></tr>`;
  }).join("");

  return `<table class="kv-table">${rows}</table>`;
}

function buildTryItForm(dataset, modelKey, describe) {
  const example = describe.how_to_call?.example_request_body || {};
  const formId = `form-${dataset}-${modelKey}`;

  let fields = "";
  if (describe.modality === "text") {
    fields = `
      <div class="form-field">
        <label for="${formId}-text">text</label>
        <textarea id="${formId}-text" data-field="text">${escapeHtml(example.text || "")}</textarea>
      </div>`;
  } else {
    fields = (describe.features || []).map((f) => {
      const value = example[f.name];
      if (f.field_type === "categorical") {
        const options = (f.categories || [])
          .map((c) => `<option value="${escapeHtml(c)}" ${c === value ? "selected" : ""}>${escapeHtml(c)}</option>`)
          .join("");
        return `<div class="form-field"><label>${escapeHtml(f.name)}</label><select data-field="${escapeHtml(f.name)}">${options}</select></div>`;
      }
      return `<div class="form-field"><label>${escapeHtml(f.name)}</label><input type="number" step="any" data-field="${escapeHtml(f.name)}" value="${escapeHtml(value)}"></div>`;
    }).join("");
  }

  return `
    <form class="try-it-form" id="${formId}" onsubmit="return false;">
      ${fields}
      <button type="submit">Run Prediction</button>
    </form>
    <div class="predict-result is-hidden" id="${formId}-result"></div>`;
}

function collectFormValues(formEl) {
  const payload = {};
  formEl.querySelectorAll("[data-field]").forEach((el) => {
    const field = el.dataset.field;
    payload[field] = el.tagName === "SELECT" || el.tagName === "TEXTAREA" ? el.value : Number(el.value);
  });
  return payload;
}

function renderPredictResult(resultEl, data, jobType) {
  resultEl.classList.remove("is-hidden", "is-error");
  if (data.predicted_price !== undefined) {
    resultEl.innerHTML = `<strong>Predicted price:</strong> ${data.predicted_price.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
    return;
  }
  const probs = Object.entries(data.predicted_probabilities || {})
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `${escapeHtml(k)}: ${(v * 100).toFixed(1)}%`)
    .join(" &nbsp;|&nbsp; ");
  resultEl.innerHTML = `<strong>Predicted class:</strong> ${escapeHtml(data.predicted_class)} (confidence ${(data.confidence * 100).toFixed(1)}%)<br><span style="color:var(--text-muted)">${probs}</span>`;
}

function renderModelCard(dataset, modelKey, describe) {
  if (describe.error) {
    return `<div class="model-card"><h3>${escapeHtml(modelKey)}</h3><p class="model-unavailable">${escapeHtml(describe.error)}</p></div>`;
  }

  const jobBadge = `<span class="badge badge--${describe.job_type}">${escapeHtml(describe.job_type)}</span>`;
  const modBadge = `<span class="badge badge--${describe.modality}">${escapeHtml(describe.modality)}</span>`;

  return `
    <div class="model-card">
      <h3>${escapeHtml(describe.model_name)}</h3>
      <div class="badges">${jobBadge}${modBadge}</div>

      <div class="section-label">Test metrics</div>
      ${kvTable(describe.metrics)}

      <details class="signature">
        <summary>Model signature &amp; details</summary>
        ${renderSignature(describe)}
      </details>

      <div class="section-label">Try it</div>
      ${buildTryItForm(dataset, modelKey, describe)}
    </div>`;
}

function attachTryItHandlers(container) {
  container.querySelectorAll("form.try-it-form").forEach((formEl) => {
    formEl.addEventListener("submit", async (e) => {
      e.preventDefault();
      const [_, dataset, modelKey] = formEl.id.split("-");
      const resultEl = document.getElementById(`${formEl.id}-result`);
      const payload = collectFormValues(formEl);
      try {
        const data = await apiPost(`/api/models/${dataset}/${modelKey}/predict`, payload);
        renderPredictResult(resultEl, data);
      } catch (err) {
        resultEl.classList.remove("is-hidden");
        resultEl.classList.add("is-error");
        resultEl.textContent = `Error: ${err.message}`;
      }
    });
  });
}

function renderModelsTab() {
  const container = document.getElementById("models-container");
  const sections = Object.entries(CATALOG).map(([datasetKey, dataset]) => {
    const modelEntries = Object.entries(dataset.models);
    const cards = modelEntries
      .map(([modelKey, describe]) => renderModelCard(datasetKey, modelKey, describe))
      .join("");
    return `
      <section class="dataset-section">
        <h2>${escapeHtml(dataset.display_name)}</h2>
        <details class="dataset-explore">
          <summary>Explore all models built on this dataset (${modelEntries.length})</summary>
          <div class="model-grid">${cards}</div>
        </details>
      </section>`;
  });
  container.innerHTML = sections.join("");
  attachTryItHandlers(container);
}

// ---------------------------------------------------------------------
// Run Test tab
// ---------------------------------------------------------------------

// Mirrors web/jobs.py's NUM_POINTS_BY_DATASET / DEFAULT_NUM_POINTS - just
// the suggested starting value shown in the field; the user can change it,
// and the server re-validates/clamps regardless (see api_start_test).
const NUM_POINTS_DEFAULTS = { bbc_news: 20 };
const DEFAULT_NUM_POINTS = 100;

function updateNumPointsDefault() {
  const [dataset] = document.getElementById("run-test-select").value.split("::");
  document.getElementById("run-test-num-points").value = NUM_POINTS_DEFAULTS[dataset] ?? DEFAULT_NUM_POINTS;
}

function populateRunTestSelect() {
  const select = document.getElementById("run-test-select");
  const options = [];
  Object.entries(CATALOG).forEach(([datasetKey, dataset]) => {
    Object.entries(dataset.models).forEach(([modelKey, describe]) => {
      if (describe.error) return;
      options.push(`<option value="${datasetKey}::${modelKey}">${escapeHtml(dataset.display_name)} &mdash; ${escapeHtml(describe.model_name)}</option>`);
    });
  });
  select.innerHTML = options.join("");
  document.getElementById("run-test-button").disabled = options.length === 0;
  select.addEventListener("change", updateNumPointsDefault);
  if (options.length > 0) updateNumPointsDefault();
}

function setRunTestState(state) {
  document.getElementById("run-test-progress").classList.toggle("is-hidden", state !== "running");
  document.getElementById("run-test-done").classList.toggle("is-hidden", state !== "completed");
  document.getElementById("run-test-error").classList.toggle("is-hidden", state !== "failed");
  document.getElementById("run-test-button").disabled = state === "running";
}

async function pollStatus(sessionId) {
  clearInterval(POLL_TIMER);
  POLL_TIMER = setInterval(async () => {
    try {
      const status = await apiGet(`/api/tests/${sessionId}/status`);
      document.getElementById("run-test-progress-text").textContent =
        status.status === "pending" ? "Queued…" : "Running… (this can take a few minutes)";

      if (status.status === "completed") {
        clearInterval(POLL_TIMER);
        setRunTestState("completed");
        document.getElementById("run-test-view-dashboard").onclick = () => openDashboard(sessionId);
        loadHistory();
      } else if (status.status === "failed") {
        clearInterval(POLL_TIMER);
        setRunTestState("failed");
        document.getElementById("run-test-error").textContent = `Test failed: ${status.error || "unknown error"}`;
        loadHistory();
      }
    } catch (err) {
      clearInterval(POLL_TIMER);
      setRunTestState("failed");
      document.getElementById("run-test-error").textContent = `Error checking status: ${err.message}`;
    }
  }, 3000);
}

function setupRunTestButton() {
  document.getElementById("run-test-button").addEventListener("click", async () => {
    const [dataset, modelKey] = document.getElementById("run-test-select").value.split("::");
    const numPoints = Number(document.getElementById("run-test-num-points").value) || undefined;
    setRunTestState("running");
    document.getElementById("run-test-progress-text").textContent = "Starting…";
    try {
      const { session_id } = await apiPost("/api/tests", { dataset, model_key: modelKey, num_points: numPoints });
      pollStatus(session_id);
    } catch (err) {
      setRunTestState("failed");
      document.getElementById("run-test-error").textContent = `Could not start test: ${err.message}`;
    }
  });
}

function statusBadge(status) {
  return `<span class="badge badge--${status}">${escapeHtml(status)}</span>`;
}

async function loadHistory() {
  const container = document.getElementById("history-container");
  try {
    const sessions = await apiGet("/api/tests");
    if (sessions.length === 0) {
      container.innerHTML = '<p class="model-unavailable">No test runs yet.</p>';
      return;
    }
    const rows = sessions.map((s) => `
      <tr>
        <td>${escapeHtml(s.dataset)} &mdash; ${escapeHtml(s.model_key)}</td>
        <td>${statusBadge(s.status)}</td>
        <td>${formatTimestamp(s.created_at)}</td>
        <td>${s.status === "completed" ? `<button class="secondary" data-session="${s.session_id}">View Dashboard</button>` : ""}</td>
      </tr>`).join("");
    container.innerHTML = `
      <table class="history-table">
        <thead><tr><th>Model</th><th>Status</th><th>Created</th><th></th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
    container.querySelectorAll("button[data-session]").forEach((btn) => {
      btn.addEventListener("click", () => openDashboard(btn.dataset.session));
    });
  } catch (err) {
    container.innerHTML = `<p class="model-unavailable">Could not load history: ${escapeHtml(err.message)}</p>`;
  }
}

// ---------------------------------------------------------------------
// Dashboard (modal)
// ---------------------------------------------------------------------

function renderTabularReport(report) {
  const numeric = (report.top_numeric_features || [])
    .map((f) => `<tr><td>${escapeHtml(f.name)}</td><td>${(f.sensitivity_score * 100).toFixed(1)}%</td></tr>`).join("");
  const categorical = (report.top_categorical_features || [])
    .map((f) => `<tr><td>${escapeHtml(f.name)}</td><td>${(f.sensitivity_score * 100).toFixed(1)}%</td></tr>`).join("");

  const impactFeatures = (report.feature_impact_details || []).map((f) => f.feature_name);
  const chartImgs = impactFeatures.map((name) =>
    `<img src="${API_BASE}/api/tests/${CURRENT_DASHBOARD_SESSION}/artifacts/impact_chart_${encodeURIComponent(name)}" alt="Impact of ${escapeHtml(name)}">`
  ).join("");

  return `
    <div class="section-label">Sensitivity ranking</div>
    <table class="kv-table">
      <tr><td colspan="2" style="color:var(--text-muted); font-size:0.8rem;">Numeric</td></tr>
      ${numeric || '<tr><td colspan="2">None</td></tr>'}
      <tr><td colspan="2" style="color:var(--text-muted); font-size:0.8rem;">Categorical</td></tr>
      ${categorical || '<tr><td colspan="2">None</td></tr>'}
    </table>

    <div class="section-label">Overall sensitivity chart</div>
    <div class="chart-grid"><img src="${API_BASE}/api/tests/${CURRENT_DASHBOARD_SESSION}/artifacts/sensitivity_chart" alt="Sensitivity chart"></div>

    ${chartImgs ? `<div class="section-label">Feature impact charts</div><div class="chart-grid">${chartImgs}</div>` : ""}`;
}

function renderTextReport(report) {
  return `
    <div class="section-label">Categories</div>
    <p style="font-size:0.88rem;">${escapeHtml((report.categories || []).join(", "))}</p>

    <div class="section-label">Documents generated per category</div>
    ${kvTable(report.documents_generated_per_category)}

    <div class="section-label">Model's predicted category distribution</div>
    ${kvTable(report.predicted_category_distribution)}

    <p class="model-unavailable">Per-document highlighting is in the PDF report below.</p>`;
}

let CURRENT_DASHBOARD_SESSION = null;

async function openDashboard(sessionId) {
  CURRENT_DASHBOARD_SESSION = sessionId;
  const modalBody = document.getElementById("modal-body");
  modalBody.innerHTML = '<p class="loading-placeholder">Loading dashboard&hellip;</p>';
  document.getElementById("modal-backdrop").classList.remove("is-hidden");

  try {
    const { status, result, report } = await apiGet(`/api/tests/${sessionId}/dashboard`);

    const headerRows = `
      <tr><td>Model</td><td>${escapeHtml(result.model_name)}</td></tr>
      <tr><td>Dataset / model key</td><td>${escapeHtml(status.dataset)} / ${escapeHtml(status.model_key)}</td></tr>
      <tr><td>Task type</td><td>${escapeHtml(result.job_type)}${result.reference_class ? ` (explaining likelihood of: ${escapeHtml(result.reference_class)})` : ""}</td></tr>
      <tr><td>Data points analyzed</td><td>${escapeHtml(result.num_points_analyzed)}</td></tr>
      <tr><td>Run at</td><td>${formatTimestamp(status.completed_at)}</td></tr>`;

    const artifacts = result.artifacts || {};
    const pdfKey = Object.keys(artifacts).find((k) => k === "report_pdf");
    const otherLinks = Object.keys(artifacts)
      .filter((k) => k !== "report_pdf")
      .map((k) => `<a href="${API_BASE}/api/tests/${sessionId}/artifacts/${encodeURIComponent(k)}" target="_blank">${escapeHtml(k)}</a>`)
      .join("");

    const bodyContent = status.dataset === "bbc_news" ? renderTextReport(report) : renderTabularReport(report);

    modalBody.innerHTML = `
      <h2>Test dashboard</h2>
      <table class="dashboard-header-table">${headerRows}</table>

      ${pdfKey ? `<a class="download-pdf-button" href="${API_BASE}/api/tests/${sessionId}/artifacts/report_pdf" target="_blank">Download PDF Report</a>` : ""}

      ${bodyContent}

      <div class="section-label">All artifacts</div>
      <div class="artifact-list">${otherLinks || "None"}</div>`;
  } catch (err) {
    modalBody.innerHTML = `<p class="model-unavailable">Could not load dashboard: ${escapeHtml(err.message)}</p>`;
  }
}

function setupModal() {
  document.getElementById("modal-close").addEventListener("click", () => {
    document.getElementById("modal-backdrop").classList.add("is-hidden");
  });
  document.getElementById("modal-backdrop").addEventListener("click", (e) => {
    if (e.target.id === "modal-backdrop") e.target.classList.add("is-hidden");
  });
}

// ---------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------

async function boot() {
  setupTabs();
  setupModal();
  setupRunTestButton();

  try {
    CATALOG = await apiGet("/api/models");
    renderModelsTab();
    populateRunTestSelect();
  } catch (err) {
    document.getElementById("models-container").innerHTML = `<p class="model-unavailable">Could not load models: ${escapeHtml(err.message)}</p>`;
  }

  loadHistory();
}

boot();
