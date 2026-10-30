/**
 * Rainfall Nowcasting & Inference Latency Dashboard - Client Logic
 */

// Application State
const state = {
  temp: 34.0,
  rhum: 53.0,
  dwpt: 23.1,
  pres: 1005.0,
  wspd: 7.0,
  prcp: 0.0,
  hour: 14,
  month: 10
};

// Preset Scenarios
const PRESETS = {
  normal: {
    temp: 34.0,
    rhum: 53.0,
    dwpt: 23.1,
    pres: 1005.0,
    wspd: 7.0,
    prcp: 0.0,
    hour: 14,
    month: 10
  },
  storm: {
    temp: 29.5,
    rhum: 88.0,
    dwpt: 27.2,
    pres: 1001.0,
    wspd: 22.0,
    prcp: 8.5,
    hour: 16,
    month: 10
  },
  monsoon: {
    temp: 26.0,
    rhum: 96.0,
    dwpt: 25.3,
    pres: 996.5,
    wspd: 34.0,
    prcp: 18.0,
    hour: 18,
    month: 7
  },
  cool: {
    temp: 22.5,
    rhum: 62.0,
    dwpt: 14.8,
    pres: 1014.0,
    wspd: 4.0,
    prcp: 0.0,
    hour: 23,
    month: 12
  }
};

let debounceTimer = null;

// Initialize when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  console.log("[*] Dashboard UI initialized.");
  // Trigger initial prediction
  requestPrediction();
});

// Update Parameter from Slider
function updateParam(param, value, unit) {
  state[param] = parseFloat(value);
  const badgeEl = document.getElementById(`val-${param}`);
  if (badgeEl) {
    badgeEl.textContent = `${value} ${unit}`;
  }

  // Clear active preset buttons
  document.querySelectorAll(".preset-btn").forEach(btn => btn.classList.remove("active"));

  // Debounce API call (60ms for smooth feel)
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    requestPrediction();
  }, 60);
}

// Update Hour and Month Selectors
function updateTime() {
  state.hour = parseInt(document.getElementById("select-hour").value, 10);
  state.month = parseInt(document.getElementById("select-month").value, 10);
  requestPrediction();
}

// Apply Scenario Preset
function applyPreset(presetKey) {
  const p = PRESETS[presetKey];
  if (!p) return;

  // Update State
  Object.assign(state, p);

  // Update UI Elements
  document.getElementById("slider-temp").value = p.temp;
  document.getElementById("val-temp").textContent = `${p.temp} °C`;

  document.getElementById("slider-rhum").value = p.rhum;
  document.getElementById("val-rhum").textContent = `${p.rhum} %`;

  document.getElementById("slider-dwpt").value = p.dwpt;
  document.getElementById("val-dwpt").textContent = `${p.dwpt} °C`;

  document.getElementById("slider-pres").value = p.pres;
  document.getElementById("val-pres").textContent = `${p.pres} hPa`;

  document.getElementById("slider-wspd").value = p.wspd;
  document.getElementById("val-wspd").textContent = `${p.wspd} km/h`;

  document.getElementById("slider-prcp").value = p.prcp;
  document.getElementById("val-prcp").textContent = `${p.prcp} mm/h`;

  document.getElementById("select-hour").value = p.hour;
  document.getElementById("select-month").value = p.month;

  // Toggle active button
  document.querySelectorAll(".preset-btn").forEach(btn => btn.classList.remove("active"));
  const activeBtn = document.getElementById(`preset-${presetKey}`);
  if (activeBtn) activeBtn.classList.add("active");

  requestPrediction();
}

// Reset to Defaults
function resetDefaults() {
  applyPreset("normal");
}

// Request Prediction from Backend API
async function requestPrediction() {
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state)
    });

    if (!res.ok) {
      console.warn("Prediction request returned status", res.status);
      return;
    }

    const data = await res.json();
    renderPredictionResults(data);
  } catch (err) {
    console.error("Failed to fetch prediction:", err);
  }
}

// Render Results to DOM
function renderPredictionResults(data) {
  const { pred_rain_mm, alert, latency } = data;

  // 1. Update Predicted Rain Number
  const rainEl = document.getElementById("pred-rain-val");
  if (rainEl) {
    rainEl.textContent = Number(pred_rain_mm).toFixed(2);
  }

  // 2. Update Alert Banner
  const alertBanner = document.getElementById("alert-banner");
  const alertIcon = document.getElementById("alert-icon");
  const alertTitle = document.getElementById("alert-title");
  const alertDesc = document.getElementById("alert-desc");

  if (alertBanner) {
    alertBanner.className = `alert-banner alert-${alert.level.toLowerCase()}`;
    alertIcon.textContent = alert.level === "RED" ? "🔴" : (alert.level === "ORANGE" ? "🟠" : (alert.level === "YELLOW" ? "🟡" : "🟢"));
    alertTitle.textContent = alert.title;
    alertDesc.textContent = alert.desc;
  }

  // 3. Update Latency Hardware Gauge
  const gaugeNum = document.getElementById("gauge-ms");
  const gaugeBudgetTag = document.getElementById("gauge-budget-tag");
  const gaugeCircle = document.getElementById("gauge-circle");
  const feasBadge = document.getElementById("feasibility-badge");

  if (gaugeNum) gaugeNum.textContent = Number(latency.total_ms).toFixed(2);
  if (gaugeBudgetTag) gaugeBudgetTag.textContent = `${latency.utilization_pct}% Ngân Sách`;

  // Circumference: 2 * PI * 65 = 408.4
  const circumference = 408.4;
  const ratio = Math.min(1.0, latency.total_ms / 5.0);
  const offset = circumference * (1 - ratio);
  if (gaugeCircle) {
    gaugeCircle.style.strokeDashoffset = offset;
    if (latency.total_ms <= 3.5) {
      gaugeCircle.style.stroke = "var(--status-green)";
    } else if (latency.total_ms <= 5.0) {
      gaugeCircle.style.stroke = "var(--primary-cyan)";
    } else {
      gaugeCircle.style.stroke = "var(--status-red)";
    }
  }

  // Feasibility status badge
  if (feasBadge) {
    if (latency.is_feasible) {
      feasBadge.className = "feasibility-badge feasible";
      feasBadge.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg><span>THỎA MÃN (&le; 5.0 ms)</span>`;
    } else {
      feasBadge.className = "feasibility-badge violated";
      feasBadge.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg><span>VI PHẠM (> 5.0 ms)</span>`;
    }
  }

  // 4. Update Latency Breakdown Bars
  const latFeatureVal = document.getElementById("lat-feature-val");
  const latPredictVal = document.getElementById("lat-predict-val");
  const barFeature = document.getElementById("bar-feature");
  const barPredict = document.getElementById("bar-predict");

  if (latFeatureVal) latFeatureVal.textContent = `${latency.feature_ms.toFixed(3)} ms`;
  if (latPredictVal) latPredictVal.textContent = `${latency.predict_ms.toFixed(3)} ms`;

  if (barFeature) barFeature.style.width = `${Math.min(100, (latency.feature_ms / 5.0) * 100)}%`;
  if (barPredict) barPredict.style.width = `${Math.min(100, (latency.predict_ms / 5.0) * 100)}%`;
}

// Run 200 Iterations Live Latency Benchmark
async function runBenchmark() {
  const btn = document.getElementById("btn-run-benchmark");
  if (!btn) return;

  const originalContent = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `<span class="pulse-dot"></span><span>Đang đo 200 lượt...</span>`;

  try {
    const res = await fetch("/api/benchmark", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...state, n_runs: 200, warmup: 20 })
    });

    if (res.ok) {
      const data = await res.json();
      document.getElementById("b-p50").textContent = `${data.p50_ms.toFixed(2)} ms`;
      document.getElementById("b-p95").textContent = `${data.p95_ms.toFixed(2)} ms`;
      document.getElementById("b-p99").textContent = `${data.p99_ms.toFixed(2)} ms`;
      document.getElementById("b-mean").textContent = `${data.mean_ms.toFixed(2)} ± ${data.std_ms.toFixed(2)} ms`;

      // Highlight green if feasible
      const p95El = document.getElementById("b-p95");
      if (data.is_feasible) {
        p95El.className = "b-metric-val text-green";
      } else {
        p95El.className = "b-metric-val text-red";
      }
    }
  } catch (err) {
    console.error("Benchmark error:", err);
  } finally {
    btn.disabled = false;
    btn.innerHTML = originalContent;
  }
}
