const GROUP_BY_OPTIONS = ["reporting_date", "asset_class", "rating", "customer_id"];
const AGG_OPTIONS = {
  row_count: "Row Count",
  sum_exposure: "Sum Exposure",
  avg_exposure: "Average Exposure",
  avg_pd: "Average PD",
  weighted_avg_pd: "Weighted Average PD",
};
const RATING_ORDER = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"];
const API_BASE =
  window.location.hostname === "localhost" && window.location.port === "3000"
    ? "http://localhost:8000"
    : "";

const state = {
  rawData: [],
  metadata: {
    reporting_dates: [],
    ratings: [],
    asset_classes: [],
  },
  selectedDates: [],
  selectedAssetClasses: [],
  selectedRatings: [],
  customerSearch: "",
  pdMin: 0,
  pdMax: 1,
  exposureMin: 0,
  exposureMax: 0,
  groupBy: ["reporting_date"],
  aggregations: ["row_count", "sum_exposure", "weighted_avg_pd"],
  filteredData: [],
  availableModels: [],
  analysisResult: null,
};

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function fmtPct(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function fmtNum(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

function fmtEur(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `€${Number(value).toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits })}`;
}

function toPre(text) {
  return `<pre>${escapeHtml(text || "")}</pre>`;
}

function checkedValues(containerId) {
  return Array.from(document.querySelectorAll(`#${containerId} input[type="checkbox"]:checked`)).map((el) => el.value);
}

function apiUrl(path) {
  return `${API_BASE}${path}`;
}

function setError(message = "") {
  const errorEl = document.getElementById("llmError");
  if (!message) {
    errorEl.style.display = "none";
    errorEl.textContent = "";
    return;
  }
  errorEl.style.display = "block";
  errorEl.textContent = message;
}

function setStatus(message = "") {
  document.getElementById("llmStatus").textContent = message;
}

function renderChecklist(containerId, options, selectedValues, onToggle) {
  const container = document.getElementById(containerId);
  container.innerHTML = options
    .map((value) => {
      const checked = selectedValues.includes(value) ? "checked" : "";
      return `<label><input type="checkbox" value="${escapeHtml(value)}" ${checked} /> ${escapeHtml(value)}</label>`;
    })
    .join("");

  container.querySelectorAll("input[type='checkbox']").forEach((el) => {
    el.addEventListener("change", (event) => onToggle(event.target.value, event.target.checked, event.target));
  });
}

function applyFilters() {
  const pdMinInput = Number(document.getElementById("pdMin").value);
  const pdMaxInput = Number(document.getElementById("pdMax").value);
  const expMinInput = Number(document.getElementById("expMin").value);
  const expMaxInput = Number(document.getElementById("expMax").value);
  state.customerSearch = document.getElementById("customerSearch").value.trim().toLowerCase();
  state.pdMin = Number.isFinite(pdMinInput) ? pdMinInput : state.pdMin;
  state.pdMax = Number.isFinite(pdMaxInput) ? pdMaxInput : state.pdMax;
  state.exposureMin = Number.isFinite(expMinInput) ? expMinInput : state.exposureMin;
  state.exposureMax = Number.isFinite(expMaxInput) ? expMaxInput : state.exposureMax;
  state.groupBy = checkedValues("groupByFilters");
  state.aggregations = checkedValues("aggregationFilters");

  state.filteredData = state.rawData.filter((row) => {
    const matchesDate = state.selectedDates.includes(row.reporting_date);
    const matchesAsset = state.selectedAssetClasses.includes(row.asset_class);
    const matchesRating = state.selectedRatings.includes(row.rating);
    const matchesPd = Number(row.pd) >= state.pdMin && Number(row.pd) <= state.pdMax;
    const matchesExposure =
      Number(row.exposure) >= state.exposureMin && Number(row.exposure) <= state.exposureMax;
    const matchesCustomer =
      !state.customerSearch || String(row.customer_id).toLowerCase().includes(state.customerSearch);
    return (
      matchesDate &&
      matchesAsset &&
      matchesRating &&
      matchesPd &&
      matchesExposure &&
      matchesCustomer
    );
  });
}

function renderTable(containerId, rows, orderedColumns = null) {
  const container = document.getElementById(containerId);
  if (!rows || rows.length === 0) {
    container.innerHTML = `<p class="muted">No rows to display.</p>`;
    return;
  }
  const columns = orderedColumns || Object.keys(rows[0]);
  const head = columns.map((col) => `<th>${escapeHtml(col)}</th>`).join("");
  const body = rows
    .map((row) => {
      return `<tr>${columns.map((col) => `<td>${escapeHtml(row[col] ?? "-")}</td>`).join("")}</tr>`;
    })
    .join("");
  container.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

function renderDataMetrics() {
  const rows = state.filteredData.length;
  const customers = new Set(state.filteredData.map((row) => row.customer_id)).size;
  const exposure = state.filteredData.reduce((acc, row) => acc + Number(row.exposure), 0);
  const html = `
    <div class="metric"><div class="k">Rows</div><div class="v">${fmtNum(rows, 0)}</div></div>
    <div class="metric"><div class="k">Customers</div><div class="v">${fmtNum(customers, 0)}</div></div>
    <div class="metric"><div class="k">Total Exposure</div><div class="v">${fmtEur(exposure, 2)}</div></div>
  `;
  document.getElementById("dataMetrics").innerHTML = html;
}

function buildAssetClassComparison() {
  if (state.selectedDates.length !== 2) return null;
  const [d0, d1] = state.selectedDates;

  const rows = state.metadata.asset_classes.map((assetClass) => {
    const d0Rows = state.filteredData.filter(
      (row) => row.asset_class === assetClass && row.reporting_date === d0
    );
    const d1Rows = state.filteredData.filter(
      (row) => row.asset_class === assetClass && row.reporting_date === d1
    );

    const d0Exposure = d0Rows.reduce((acc, row) => acc + Number(row.exposure), 0);
    const d1Exposure = d1Rows.reduce((acc, row) => acc + Number(row.exposure), 0);
    const d0AvgPd =
      d0Rows.length === 0
        ? null
        : d0Rows.reduce((acc, row) => acc + Number(row.pd), 0) / d0Rows.length;
    const d1AvgPd =
      d1Rows.length === 0
        ? null
        : d1Rows.reduce((acc, row) => acc + Number(row.pd), 0) / d1Rows.length;

    return {
      asset_class: assetClass,
      [`exposure_${d0}`]: fmtEur(d0Exposure, 2),
      [`avg_pd_${d0}`]: fmtPct(d0AvgPd, 4),
      [`exposure_${d1}`]: fmtEur(d1Exposure, 2),
      [`avg_pd_${d1}`]: fmtPct(d1AvgPd, 4),
    };
  });
  return rows;
}

function aggregateRows() {
  if (!state.aggregations.length) return [];
  const rows = state.filteredData;
  if (!rows.length) return [];

  const finalizeMetrics = (group) => {
    const output = {};
    if (state.aggregations.includes("row_count")) output.row_count = group.count;
    if (state.aggregations.includes("sum_exposure")) output.sum_exposure = fmtEur(group.sumExposure);
    if (state.aggregations.includes("avg_exposure"))
      output.avg_exposure = fmtEur(group.count ? group.sumExposure / group.count : 0);
    if (state.aggregations.includes("avg_pd")) output.avg_pd = fmtPct(group.count ? group.sumPd / group.count : 0, 4);
    if (state.aggregations.includes("weighted_avg_pd"))
      output.weighted_avg_pd = fmtPct(
        group.sumExposure ? group.sumPdExposure / group.sumExposure : 0,
        4
      );
    return output;
  };

  if (!state.groupBy.length) {
    const all = rows.reduce(
      (acc, row) => {
        const pd = Number(row.pd);
        const exposure = Number(row.exposure);
        acc.count += 1;
        acc.sumExposure += exposure;
        acc.sumPd += pd;
        acc.sumPdExposure += pd * exposure;
        return acc;
      },
      { count: 0, sumExposure: 0, sumPd: 0, sumPdExposure: 0 }
    );
    return [finalizeMetrics(all)];
  }

  const groups = new Map();
  for (const row of rows) {
    const keyParts = state.groupBy.map((field) => row[field]);
    const key = JSON.stringify(keyParts);
    if (!groups.has(key)) {
      const seed = {
        count: 0,
        sumExposure: 0,
        sumPd: 0,
        sumPdExposure: 0,
        keys: {},
      };
      state.groupBy.forEach((field, index) => {
        seed.keys[field] = keyParts[index];
      });
      groups.set(key, seed);
    }
    const group = groups.get(key);
    const pd = Number(row.pd);
    const exposure = Number(row.exposure);
    group.count += 1;
    group.sumExposure += exposure;
    group.sumPd += pd;
    group.sumPdExposure += pd * exposure;
  }

  return Array.from(groups.values()).map((group) => ({
    ...group.keys,
    ...finalizeMetrics(group),
  }));
}

function renderDataExplorer() {
  applyFilters();
  renderDataMetrics();
  renderTable("rawDataTable", state.filteredData, [
    "customer_id",
    "rating",
    "pd",
    "exposure",
    "asset_class",
    "reporting_date",
  ]);

  const comparisonCaption = document.getElementById("comparisonCaption");
  const comparisonRows = buildAssetClassComparison();
  if (!comparisonRows) {
    comparisonCaption.textContent = "Select exactly two reporting dates to compare asset classes.";
    renderTable("comparisonTable", []);
  } else {
    const [d0, d1] = state.selectedDates;
    comparisonCaption.textContent =
      `Rows: asset classes. Columns: exposure and average PD for ${d0} and ${d1}.`;
    renderTable("comparisonTable", comparisonRows);
  }

  const aggRows = aggregateRows();
  renderTable("customAggTable", aggRows);
}

function bindFilterInputs() {
  document.getElementById("customerSearch").addEventListener("input", renderDataExplorer);
  ["pdMin", "pdMax", "expMin", "expMax"].forEach((id) => {
    document.getElementById(id).addEventListener("input", renderDataExplorer);
  });
}

function initFilterControls() {
  const dates = [...state.metadata.reporting_dates];
  const ratings = [...state.metadata.ratings].sort(
    (a, b) => RATING_ORDER.indexOf(a) - RATING_ORDER.indexOf(b)
  );
  const assetClasses = [...state.metadata.asset_classes];

  state.selectedDates = dates.slice(0, 2);
  state.selectedRatings = ratings;
  state.selectedAssetClasses = assetClasses;

  const pds = state.rawData.map((row) => Number(row.pd));
  const exposures = state.rawData.map((row) => Number(row.exposure));
  state.pdMin = Math.min(...pds);
  state.pdMax = Math.max(...pds);
  state.exposureMin = Math.min(...exposures);
  state.exposureMax = Math.max(...exposures);

  document.getElementById("pdMin").value = String(state.pdMin);
  document.getElementById("pdMax").value = String(state.pdMax);
  document.getElementById("expMin").value = String(state.exposureMin);
  document.getElementById("expMax").value = String(state.exposureMax);

  renderChecklist("dateFilters", dates, state.selectedDates, (value, checked, node) => {
    if (checked && state.selectedDates.length >= 2 && !state.selectedDates.includes(value)) {
      node.checked = false;
      setStatus("Only two comparison dates are allowed.");
      return;
    }
    state.selectedDates = checked
      ? [...state.selectedDates, value]
      : state.selectedDates.filter((item) => item !== value);
    renderDataExplorer();
  });
  renderChecklist("assetFilters", assetClasses, state.selectedAssetClasses, (value, checked) => {
    state.selectedAssetClasses = checked
      ? [...state.selectedAssetClasses, value]
      : state.selectedAssetClasses.filter((item) => item !== value);
    renderDataExplorer();
  });
  renderChecklist("ratingFilters", ratings, state.selectedRatings, (value, checked) => {
    state.selectedRatings = checked
      ? [...state.selectedRatings, value]
      : state.selectedRatings.filter((item) => item !== value);
    renderDataExplorer();
  });
  renderChecklist("groupByFilters", GROUP_BY_OPTIONS, state.groupBy, (value, checked) => {
    state.groupBy = checked ? [...state.groupBy, value] : state.groupBy.filter((item) => item !== value);
    renderDataExplorer();
  });
  renderChecklist("aggregationFilters", Object.keys(AGG_OPTIONS), state.aggregations, (value, checked) => {
    state.aggregations = checked
      ? [...state.aggregations, value]
      : state.aggregations.filter((item) => item !== value);
    renderDataExplorer();
  });
}

function setupTabs() {
  const buttons = Array.from(document.querySelectorAll(".tab-btn"));
  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      const target = button.dataset.tab;
      buttons.forEach((btn) => btn.classList.remove("active"));
      button.classList.add("active");
      document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.remove("active"));
      document.getElementById(target).classList.add("active");
    });
  });
}

async function fetchData() {
  const response = await fetch(apiUrl("/api/data"));
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Failed to load data: ${body}`);
  }
  const payload = await response.json();
  state.rawData = payload.rows || [];
  state.metadata = payload.metadata || state.metadata;
}

function populateModelSelect(models, defaultModel) {
  const select = document.getElementById("model");
  select.innerHTML = "";
  models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model;
    if (model === defaultModel) option.selected = true;
    select.appendChild(option);
  });
}

async function loadModels() {
  setError("");
  setStatus("");
  const provider = document.getElementById("provider").value;
  const apiKey = document.getElementById("apiKey").value.trim();
  if (!apiKey) {
    setError("Please enter an API key before loading models.");
    return;
  }
  setStatus("Checking available models...");
  const response = await fetch(apiUrl("/api/models"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider, api_key: apiKey }),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "Unable to load models.");
  }
  state.availableModels = payload.models || [];
  populateModelSelect(state.availableModels, payload.default_model);
  setStatus(`Loaded ${state.availableModels.length} available model(s).`);
}

function resultsTable(title, rows) {
  if (!rows || !rows.length) return `<h4>${escapeHtml(title)}</h4><p class="muted">No rows.</p>`;
  const columns = Object.keys(rows[0]);
  const header = columns.map((col) => `<th>${escapeHtml(col)}</th>`).join("");
  const body = rows
    .map(
      (row) =>
        `<tr>${columns
          .map((col) => `<td>${escapeHtml(row[col] === null ? "-" : row[col])}</td>`)
          .join("")}</tr>`
    )
    .join("");
  return `
    <h4>${escapeHtml(title)}</h4>
    <div class="table-wrap">
      <table>
        <thead><tr>${header}</tr></thead>
        <tbody>${body}</tbody>
      </table>
    </div>
  `;
}

function renderExecutionWindow() {
  const logTab = document.getElementById("logTab");
  const sqlTab = document.getElementById("sqlTab");
  const resultTab = document.getElementById("resultTab");

  if (!state.analysisResult) {
    const placeholder = `<p class="muted">Run the analysis to print logs, SQL, execution, and results.</p>`;
    logTab.innerHTML = placeholder;
    sqlTab.innerHTML = placeholder;
    resultTab.innerHTML = placeholder;
    return;
  }

  const result = state.analysisResult;
  logTab.innerHTML = toPre((result.logs || []).join("\n"));

  const sqlSections = [];
  if (result.step1?.sql) sqlSections.push(`-- Step 1\n${result.step1.sql}`);
  if (result.step2?.sql) sqlSections.push(`-- Step 2\n${result.step2.sql}`);
  if (result.step3?.details_sql) sqlSections.push(`-- Step 3\n${result.step3.details_sql}`);
  sqlTab.innerHTML = toPre(sqlSections.join("\n\n"));

  const resultParts = [];
  resultParts.push(resultsTable("Step 1 Query Results", result.step1?.raw_rows || []));
  resultParts.push(resultsTable("Step 2 Query Results", result.step2?.raw_rows || []));
  resultParts.push(resultsTable("Step 2 Contribution Output", result.step2?.attribution_rows || []));
  resultParts.push(resultsTable("Step 3 Drill-Down Details", result.step3?.details_rows || []));
  resultParts.push(
    `<div class="note" style="margin-top:10px;">Final note: "${escapeHtml(result.summary || "")}"</div>`
  );
  resultTab.innerHTML = resultParts.join("");
}

function renderDrilldownCustomer(customer) {
  if (!customer) return "";
  const rows = state.rawData.filter((row) => row.customer_id === customer);
  if (!rows.length) return "<p class='muted'>No customer history rows.</p>";
  const columns = ["customer_id", "rating", "pd", "exposure", "asset_class", "reporting_date"];
  const head = columns.map((col) => `<th>${escapeHtml(col)}</th>`).join("");
  const body = rows
    .map((row) => `<tr>${columns.map((col) => `<td>${escapeHtml(row[col])}</td>`).join("")}</tr>`)
    .join("");
  return `<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function renderStepByStep() {
  const finalNote = document.getElementById("finalNote");
  const section = document.getElementById("stepSections");

  if (!state.analysisResult) {
    finalNote.style.display = "none";
    section.innerHTML = `<p class="muted">Run analysis to view step-by-step drill-down output.</p>`;
    return;
  }

  const result = state.analysisResult;
  finalNote.style.display = "block";
  finalNote.textContent = `Final note: "${result.summary}"`;

  const step1Rows = result.step1?.raw_rows || [];
  const step2Rows = result.step2?.attribution_rows || [];
  const step3Rows = result.step3?.details_rows || [];
  const topCustomer = result.step2?.top_customer || "";
  const firstCustomer = topCustomer || (step2Rows[0] ? step2Rows[0].customer_id : "");

  section.innerHTML = `
    <div class="panel">
      <div class="step-title"><h3>Step 1: Portfolio Level Trigger</h3></div>
      ${toPre(result.step1?.sql || "")}
      <div class="chips">
        <span class="chip">WAPD T0: ${fmtPct(result.step1?.wapd_t0, 4)}</span>
        <span class="chip">WAPD T1: ${fmtPct(result.step1?.wapd_t1, 4)}</span>
        <span class="chip">Diff: ${fmtPct(result.step1?.diff, 4)}</span>
      </div>
      ${resultsTable("Step 1 Results", step1Rows)}
    </div>
    <div class="panel">
      <div class="step-title"><h3>Step 2: Drill-Down Attribution</h3></div>
      ${toPre(result.step2?.sql || "")}
      <p><strong>Primary driver:</strong> ${escapeHtml(topCustomer)}</p>
      ${resultsTable("Attribution Rows", step2Rows)}
      <div class="control" style="margin-top:10px;">
        <label for="drilldownCustomer">Drill-down customer</label>
        <select id="drilldownCustomer">
          ${(step2Rows || [])
            .map((row) => {
              const selected = row.customer_id === firstCustomer ? "selected" : "";
              return `<option ${selected}>${escapeHtml(row.customer_id)}</option>`;
            })
            .join("")}
        </select>
      </div>
      <div id="drilldownCustomerHistory">${renderDrilldownCustomer(firstCustomer)}</div>
    </div>
    <div class="panel">
      <div class="step-title"><h3>Step 3: Root Cause & Summary</h3></div>
      ${toPre(result.step3?.details_sql || "")}
      ${resultsTable("Root Cause Details", step3Rows)}
      <p><strong>Detected change:</strong> ${escapeHtml(result.step3?.rating_change || "-")}</p>
      <p><strong>PD change:</strong> ${escapeHtml((result.step3?.pd_change || []).join(" -> "))}</p>
      <p><strong>Exposure change:</strong> ${escapeHtml((result.step3?.exposure_change || []).join(" -> "))}</p>
      <p><strong>Final report:</strong> "${escapeHtml(result.step3?.summary || "")}"</p>
    </div>
    <div class="panel">
      <h3>Drill-Down Output</h3>
      <p><strong>Top mover:</strong> ${escapeHtml(topCustomer)}</p>
      ${resultsTable("Drill-Down Details", step3Rows)}
      <div class="note">Final note: "${escapeHtml(result.step3?.summary || "")}"</div>
    </div>
    <div class="panel">
      <h3>Thought Process Log</h3>
      ${toPre((result.logs || []).join("\n"))}
    </div>
  `;

  const customerSelect = document.getElementById("drilldownCustomer");
  if (customerSelect) {
    customerSelect.addEventListener("change", (event) => {
      document.getElementById("drilldownCustomerHistory").innerHTML = renderDrilldownCustomer(
        event.target.value
      );
    });
  }
}

async function runAnalysis() {
  setError("");
  const provider = document.getElementById("provider").value;
  const apiKey = document.getElementById("apiKey").value.trim();
  const model = document.getElementById("model").value;

  if (!apiKey) {
    setError("Please enter an API key.");
    return;
  }
  if (!model) {
    setError("Please load and select a model before running analysis.");
    return;
  }

  setStatus("Running analysis...");
  const response = await fetch(apiUrl("/api/analyze"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider, api_key: apiKey, model, seed: 42 }),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "Analysis failed.");
  }

  state.analysisResult = payload;
  setStatus(`Analysis completed. Final note: "${payload.summary}"`);
  renderExecutionWindow();
  renderStepByStep();
}

function bindLlmControls() {
  document.getElementById("provider").addEventListener("change", () => {
    setError("");
    setStatus("");
    state.availableModels = [];
    document.getElementById("model").innerHTML = "";
  });

  document.getElementById("loadModelsBtn").addEventListener("click", async () => {
    try {
      await loadModels();
    } catch (err) {
      setError(err.message || String(err));
      setStatus("");
    }
  });

  document.getElementById("runBtn").addEventListener("click", async () => {
    try {
      await runAnalysis();
    } catch (err) {
      setError(err.message || String(err));
      setStatus("");
    }
  });
}

async function init() {
  setupTabs();
  bindFilterInputs();
  bindLlmControls();
  renderExecutionWindow();
  renderStepByStep();

  try {
    setStatus("Loading data...");
    await fetchData();
    initFilterControls();
    renderDataExplorer();
    setStatus("Data loaded.");
  } catch (err) {
    setError(err.message || String(err));
    setStatus("");
  }
}

document.addEventListener("DOMContentLoaded", init);
