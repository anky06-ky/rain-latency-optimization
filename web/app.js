/**
 * Dự báo mưa ngắn hạn dưới ràng buộc độ trễ - Logic phía client
 *
 * Trang đi theo thứ tự demo của môn học: bài toán -> kết quả tìm kiếm -> kịch bản epsilon ->
 * kiểm chứng triển khai -> đối chuẩn nhiều mô hình. Giao tiếp với src/server.py:
 *   GET  /api/status        -> trạng thái sẵn sàng của server + mô hình
 *   GET  /api/optimization  -> lịch sử tìm kiếm, kịch bản epsilon, results/final_test_metrics.json
 *   GET  /api/models        -> các dòng của results/multi_model_benchmark.csv
 *   POST /api/predict       -> dự báo, mức cảnh báo, phân rã độ trễ của một request đơn lẻ
 *   POST /api/benchmark     -> P50/P95/P99 qua n_runs request
 */

const LATENCY_LIMIT_MS = 5.0;
const SAFETY_MARGIN_MS = 4.5;
const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 65;

// Quan trắc thật theo từng giờ từ tập Test được giữ riêng (H2/2025), lấy từ
// data/raw qua build_features(L=1). `actual` là lượng mưa quan trắc được trong giờ kế tiếp.
const PRESETS = [
  {
    id: "dry-afternoon", icon: "☀️", name: "Chiều nắng khô",
    time: "2025-07-25 15:00", actual: 0.0,
    obs: { temp: 34.0, dwpt: 24.0, pres: 1005.0, wspd: 22.0, prcp: 0.0, hour: 15, month: 7 },
  },
  {
    id: "heavy-rain", icon: "⛈️", name: "Đang mưa lớn",
    time: "2025-09-05 13:00", actual: 7.3,
    obs: { temp: 33.0, dwpt: 26.1, pres: 1005.0, wspd: 15.0, prcp: 12.0, hour: 13, month: 9 },
  },
  {
    id: "sudden-storm", icon: "⚡", name: "Dông bùng phát đột ngột",
    time: "2025-09-04 13:00", actual: 6.2,
    obs: { temp: 33.0, dwpt: 25.0, pres: 1005.0, wspd: 11.0, prcp: 0.0, hour: 13, month: 9 },
  },
  {
    id: "light-rain", icon: "🌦️", name: "Mưa nhẹ kéo dài",
    time: "2025-07-03 15:00", actual: 1.7,
    obs: { temp: 33.0, dwpt: 25.0, pres: 1006.0, wspd: 22.0, prcp: 2.0, hour: 15, month: 7 },
  },
  {
    id: "cool-night", icon: "🌙", name: "Rạng sáng mát, khô",
    time: "2025-12-08 04:00", actual: 0.0,
    obs: { temp: 21.0, dwpt: 16.0, pres: 1011.0, wspd: 7.0, prcp: 0.0, hour: 4, month: 12 },
  },
];

const SLIDERS = {
  temp: { unit: "°C", digits: 1 },
  dwpt: { unit: "°C", digits: 1 },
  pres: { unit: "hPa", digits: 1 },
  wspd: { unit: "km/h", digits: 0 },
  prcp: { unit: "mm", digits: 1 },
};

const ALERT_COPY = {
  GREEN: "Không mưa / mưa không đáng kể",
  YELLOW: "Mưa dông nhẹ",
  ORANGE: "Mưa rào nặng hạt",
  RED: "Mưa cực đoan",
};

// Mỗi phương pháp giữ cố định một màu (ô màu phân loại số 1 và 2), không đổi màu theo thứ hạng.
const METHODS = {
  OptunaTPE: { label: "Optuna TPE", cls: "tpe" },
  RandomSearch: { label: "Random Search", cls: "rs" },
};
const METHOD_ORDER = ["OptunaTPE", "RandomSearch"];

const state = { ...PRESETS[0].obs, rhum: 0 };
let activePreset = PRESETS[0];
let debounceTimer = null;
let latestRequestId = 0;

// Kết quả tối ưu (được điền từ GET /api/optimization)
const opt = {
  data: null,
  idx: null,
  rows: {},
  convergence: null,
  pareto: null,
  eps: null,
  scatter: null,
};

/* ---------- Hàm tiện ích ---------- */

const $ = (id) => document.getElementById(id);

function fmt(value, digits) {
  return Number(value).toFixed(digits);
}

function signed(value, digits) {
  return `${value > 0 ? "+" : ""}${fmt(value, digits)}`;
}

function fmtEps(value) {
  return String(Number(value));
}

// Độ ẩm tương đối tính từ nhiệt độ và điểm sương (công thức Magnus, hệ số của Alduchov & Eskridge).
// Làm tròn thành số nguyên, giống bản ghi trạm trên Meteostat.
function relativeHumidity(temp, dwpt) {
  const a = 17.625, b = 243.04;
  const rh = 100 * Math.exp((a * dwpt) / (b + dwpt) - (a * temp) / (b + temp));
  return Math.round(Math.min(100, Math.max(0, rh)));
}

function showError(message) {
  const el = $("error-notice");
  el.textContent = message;
  el.hidden = !message;
}

function setServerStatus(kind, text) {
  const pill = $("server-status");
  pill.className = `status-pill status-${kind}`;
  $("server-status-text").textContent = text;
}

function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ---------- Hiển thị đầu vào ---------- */

function buildSelects() {
  const hourSel = $("select-hour");
  for (let h = 0; h < 24; h++) {
    hourSel.add(new Option(`${String(h).padStart(2, "0")}:00`, String(h)));
  }
  const monthSel = $("select-month");
  for (let m = 1; m <= 12; m++) {
    const season = m >= 5 && m <= 11 ? "mùa mưa" : "mùa khô";
    monthSel.add(new Option(`Tháng ${m} (${season})`, String(m)));
  }
}

function buildPresets() {
  const grid = $("presets-grid");
  grid.innerHTML = "";
  for (const p of PRESETS) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "preset-btn";
    btn.dataset.preset = p.id;
    btn.setAttribute("aria-pressed", "false");
    const [date, time] = p.time.split(" ");
    const [y, mo, d] = date.split("-");
    btn.innerHTML = `
      <span class="preset-icon" aria-hidden="true">${p.icon}</span>
      <span class="preset-text">
        <span class="preset-name">${p.name}</span>
        <span class="preset-meta">${d}/${mo}/${y} ${time}</span>
        <span class="preset-meta">giờ sau thực tế: ${fmt(p.actual, 1)} mm</span>
      </span>`;
    btn.addEventListener("click", () => applyPreset(p));
    grid.appendChild(btn);
  }
}

function syncInputsFromState() {
  // dwpt không được vượt temp: cho giá trị max của thanh trượt bám theo nhiệt độ.
  const dwptSlider = $("slider-dwpt");
  dwptSlider.max = String(Math.min(30, state.temp));
  if (state.dwpt > state.temp) state.dwpt = state.temp;

  for (const [param, cfg] of Object.entries(SLIDERS)) {
    $(`slider-${param}`).value = state[param];
    $(`val-${param}`).textContent = `${fmt(state[param], cfg.digits)} ${cfg.unit}`;
  }
  $("select-hour").value = String(state.hour);
  $("select-month").value = String(state.month);

  state.rhum = relativeHumidity(state.temp, state.dwpt);
  $("val-rhum").textContent = `${state.rhum} %`;
  $("rhum-fill").style.width = `${state.rhum}%`;
  $("rhum-meter").setAttribute("aria-valuenow", String(state.rhum));
}

function syncPresetUI() {
  document.querySelectorAll(".preset-btn").forEach((btn) => {
    const on = activePreset && btn.dataset.preset === activePreset.id;
    btn.classList.toggle("active", Boolean(on));
    btn.setAttribute("aria-pressed", on ? "true" : "false");
  });
  const tag = $("mode-tag");
  tag.textContent = activePreset ? `Quan trắc ${activePreset.time}` : "Tùy chỉnh";
  tag.classList.toggle("mode-real", Boolean(activePreset));
}

/* ---------- Tương tác ---------- */

function applyPreset(preset) {
  activePreset = preset;
  Object.assign(state, preset.obs);
  syncInputsFromState();
  syncPresetUI();
  requestPrediction();
}

function onInputChanged(param, rawValue) {
  state[param] = param === "hour" || param === "month" ? parseInt(rawValue, 10) : parseFloat(rawValue);
  // Mọi thay đổi thủ công đều có nghĩa là đầu vào không còn khớp với một quan trắc đã ghi nhận.
  activePreset = null;
  syncInputsFromState();
  syncPresetUI();

  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(requestPrediction, 80);
}

function bindInputs() {
  for (const param of Object.keys(SLIDERS)) {
    $(`slider-${param}`).addEventListener("input", (e) => onInputChanged(param, e.target.value));
  }
  $("select-hour").addEventListener("change", (e) => onInputChanged("hour", e.target.value));
  $("select-month").addEventListener("change", (e) => onInputChanged("month", e.target.value));
  $("btn-run-benchmark").addEventListener("click", runBenchmark);
}

// Làm nổi bật bước có section đang cắt qua giữa viewport.
function bindStepNav() {
  const links = new Map();
  const list = document.querySelector(".step-nav-list");
  document.querySelectorAll(".step-link").forEach((a) => links.set(a.getAttribute("href").slice(1), a));
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      links.forEach((a, id) => {
        if (id === entry.target.id) {
          a.setAttribute("aria-current", "step");
          // Trên màn hình hẹp, thanh điều hướng cuộn ngang: giữ bước hiện tại luôn trong tầm nhìn.
          list.scrollTo({ left: a.parentElement.offsetLeft - 16, behavior: "smooth" });
        } else {
          a.removeAttribute("aria-current");
        }
      });
    }
  }, { rootMargin: "-45% 0px -50% 0px" });
  links.forEach((_, id) => {
    const section = $(id);
    if (section) observer.observe(section);
  });
}

/* ---------- Gọi API ---------- */

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.error || `Máy chủ trả về lỗi ${res.status}`);
  }
  return data;
}

async function requestPrediction() {
  // Khi kéo thanh trượt, phản hồi có thể về không theo thứ tự; chỉ hiển thị phản hồi của request mới nhất.
  const requestId = ++latestRequestId;
  try {
    const data = await postJSON("/api/predict", state);
    if (requestId !== latestRequestId) return;
    showError("");
    renderPrediction(data);
  } catch (err) {
    if (requestId !== latestRequestId) return;
    showError(`Không lấy được dự báo: ${err.message}. Kiểm tra rằng "python src/server.py" đang chạy.`);
  }
}

async function loadStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    if (data.model_ready) {
      setServerStatus("live", "Mô hình sẵn sàng");
    } else {
      setServerStatus("warn", "Chưa có mô hình");
      showError('Máy chủ chưa nạp mô hình. Chạy "python src/models/train.py" rồi khởi động lại server.');
    }
  } catch {
    setServerStatus("down", "Mất kết nối");
  }
}

async function loadModelTable() {
  const tbody = $("benchmark-table-body");
  try {
    const res = await fetch("/api/models");
    const rows = await res.json();
    if (!Array.isArray(rows) || rows.length === 0) throw new Error("empty");
    renderModelTable(rows);
  } catch {
    tbody.innerHTML = `<tr><td colspan="11" class="table-empty">Không tải được <code>results/multi_model_benchmark.csv</code>. Chạy <code>python src/evaluation/evaluate_final.py</code> để tạo bảng.</td></tr>`;
  }
}

async function loadOptimization() {
  try {
    const res = await fetch("/api/optimization");
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `Máy chủ trả về lỗi ${res.status}`);

    opt.data = data;
    opt.idx = Object.fromEntries(data.history.columns.map((c, i) => [c, i]));
    for (const m of METHOD_ORDER) opt.rows[m] = data.history[m];
    opt.convergence = Object.fromEntries(METHOD_ORDER.map((m) => [m, convergenceStats(opt.rows[m])]));
    opt.pareto = paretoFrontier(METHOD_ORDER.flatMap((m) => opt.rows[m]));
    opt.eps = data.constraint_ms;

    renderBudget();
    renderKPIs();
    renderConvergenceTable();
    renderScenarioTable();
    buildEpsControl();
    drawConvergence();
    drawScatter();
    selectEps(opt.eps);
    observeChartResize();
  } catch (err) {
    const msg = `Không tải được kết quả tối ưu (${err.message}). Chạy optimize_random.py, optimize_optuna.py và evaluate_final.py rồi khởi động lại server.`;
    for (const id of ["kpi-grid", "decision-card"]) {
      const p = document.createElement("p");
      p.className = "kpi-empty";
      p.textContent = msg;
      $(id).replaceChildren(p);
    }
  }
}

/* ---------- Hiển thị đầu ra: kiểm chứng triển khai ---------- */

function renderPrediction(data) {
  const { pred_rain_mm, alert, latency, model_info } = data;

  $("pred-rain-val").textContent = fmt(pred_rain_mm, 2);
  const nextHour = (state.hour + 1) % 24;
  $("target-time").textContent = `(${String(nextHour).padStart(2, "0")}:00)`;

  if (model_info) {
    const parts = [`L=${model_info.L_hours}h`];
    if (model_info.n_estimators) parts.push(`${model_info.n_estimators} cây`);
    if (model_info.max_depth) parts.push(`depth ${model_info.max_depth}`);
    $("model-meta").textContent = parts.join(" • ");
  }

  // Khi đầu vào là một quan trắc thật, so sánh với lượng mưa đã ghi nhận ở giờ kế tiếp.
  const actualBox = $("actual-box");
  if (activePreset) {
    actualBox.hidden = false;
    $("actual-val").textContent = fmt(activePreset.actual, 1);
    const diff = pred_rain_mm - activePreset.actual;
    const diffEl = $("actual-diff");
    diffEl.textContent = Math.abs(diff) < 0.05
      ? "Khớp thực tế"
      : `${diff > 0 ? "Dự báo cao hơn" : "Dự báo thấp hơn"} ${fmt(Math.abs(diff), 2)} mm`;
    diffEl.className = `actual-diff ${Math.abs(diff) <= 1.0 ? "diff-ok" : "diff-off"}`;
  } else {
    actualBox.hidden = true;
  }

  const banner = $("alert-banner");
  banner.className = `alert-banner alert-${alert.level.toLowerCase()}`;
  $("alert-title").textContent = alert.title || ALERT_COPY[alert.level];
  $("alert-desc").textContent = alert.desc;

  renderLatency(latency);
}

function renderLatency(latency) {
  const total = latency.total_ms;
  $("gauge-ms").textContent = fmt(total, total < 1 ? 3 : 2);
  $("gauge-budget-tag").textContent = `${fmt(latency.utilization_pct, 1)}% ngân sách`;

  const ratio = Math.min(1, total / LATENCY_LIMIT_MS);
  const gauge = $("gauge-circle");
  gauge.style.strokeDashoffset = String(GAUGE_CIRCUMFERENCE * (1 - ratio));
  gauge.dataset.level = total <= SAFETY_MARGIN_MS ? "safe" : total <= LATENCY_LIMIT_MS ? "tight" : "over";

  const badge = $("feasibility-badge");
  if (latency.is_feasible) {
    badge.className = "feasibility-badge feasible";
    badge.textContent = "✓ Thỏa ràng buộc";
  } else {
    badge.className = "feasibility-badge violated";
    badge.textContent = "✕ Vượt 5.0 ms";
  }

  $("lat-feature-val").textContent = `${fmt(latency.feature_ms, 3)} ms`;
  $("lat-predict-val").textContent = `${fmt(latency.predict_ms, 3)} ms`;
  // Các thanh dùng chung thang với đồng hồ đo (toàn chiều rộng = 5.0 ms) nên so sánh trực tiếp được độ dài.
  $("bar-feature").style.width = `${Math.min(100, (latency.feature_ms / LATENCY_LIMIT_MS) * 100)}%`;
  $("bar-predict").style.width = `${Math.min(100, (latency.predict_ms / LATENCY_LIMIT_MS) * 100)}%`;
}

function renderModelTable(rows) {
  // Giá trị tốt nhất mỗi cột: sai số càng thấp càng tốt, F1 và skill score càng cao càng tốt.
  const best = {
    Val_RMSE: Math.min(...rows.map((r) => r.Val_RMSE)),
    Test_RMSE: Math.min(...rows.map((r) => r.Test_RMSE)),
    Wet_Test_RMSE: Math.min(...rows.map((r) => r.Wet_Test_RMSE)),
    Rain_Event_F1: Math.max(...rows.map((r) => r.Rain_Event_F1)),
    Skill_Score_Persistence_pct: Math.max(...rows.map((r) => r.Skill_Score_Persistence_pct)),
    Skill_Score_Climatology_pct: Math.max(...rows.map((r) => r.Skill_Score_Climatology_pct)),
  };
  const cell = (row, key, digits, suffix = "") => {
    const v = row[key];
    const text = `${key.startsWith("Skill") && v > 0 ? "+" : ""}${fmt(v, digits)}${suffix}`;
    return `<td class="num${v === best[key] ? " is-best" : ""}">${text}</td>`;
  };

  $("benchmark-table-body").innerHTML = rows.map((row) => {
    const name = String(row.Model).replace(/^\d+\.\s*/, "");
    const champion = /optuna/i.test(name);
    const nominal = /persistence|climatology/i.test(name);
    const feasible = row.Feasible === true || row.Feasible === "True";
    return `
      <tr class="${champion ? "row-champion" : ""}">
        <th scope="row">${escapeHTML(name)}${champion ? ' <span class="badge-tag tag-champion">Chọn</span>' : ""}</th>
        <td class="num">${row.L}h</td>
        <td class="cfg">${escapeHTML(row["Cấu hình"])}</td>
        ${cell(row, "Val_RMSE", 4)}
        ${cell(row, "Test_RMSE", 4)}
        ${cell(row, "Wet_Test_RMSE", 4)}
        ${cell(row, "Rain_Event_F1", 4)}
        ${cell(row, "Skill_Score_Persistence_pct", 2, "%")}
        ${cell(row, "Skill_Score_Climatology_pct", 2, "%")}
        <td class="num">${fmt(row.Latency_P95, nominal ? 2 : 3)} ms${nominal ? "*" : ""}</td>
        <td>${feasible
          ? '<span class="badge-tag tag-green">Thỏa</span>'
          : '<span class="badge-tag tag-red">Vi phạm</span>'}</td>
      </tr>`;
  }).join("");
}

async function runBenchmark() {
  const btn = $("btn-run-benchmark");
  const originalContent = btn.innerHTML;
  btn.disabled = true;
  btn.setAttribute("aria-busy", "true");
  btn.innerHTML = `<span class="spinner" aria-hidden="true"></span><span>Đang đo…</span>`;

  try {
    const data = await postJSON("/api/benchmark", { ...state, n_runs: 200 });
    showError("");
    $("b-p50").textContent = `${fmt(data.p50_ms, 3)} ms`;
    $("b-p95").textContent = `${fmt(data.p95_ms, 3)} ms`;
    $("b-p99").textContent = `${fmt(data.p99_ms, 3)} ms`;
    $("b-mean").textContent = `${fmt(data.mean_ms, 3)} ± ${fmt(data.std_ms, 3)} ms`;
    $("b-p95").className = `b-metric-val ${data.is_feasible ? "text-green" : "text-red"}`;
    $("benchmark-empty").textContent = data.is_feasible
      ? `P₉₅ = ${fmt(data.p95_ms, 3)} ms, dùng ${fmt((data.p95_ms / LATENCY_LIMIT_MS) * 100, 1)}% ngân sách 5.0 ms.`
      : `P₉₅ = ${fmt(data.p95_ms, 3)} ms vượt ràng buộc 5.0 ms trên máy này.`;
  } catch (err) {
    showError(`Benchmark thất bại: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.removeAttribute("aria-busy");
    btn.innerHTML = originalContent;
  }
}

/* ---------- Kết quả tối ưu: dữ liệu dẫn xuất ---------- */

// Val RMSE tốt nhất tính đến hiện tại (chỉ xét trial khả thi) theo từng seed, rồi lấy trung bình và
// độ lệch chuẩn mẫu qua các seed. Giống plot_convergence_comparison trong src/evaluation/evaluate_final.py
// (các trial trước trial khả thi đầu tiên nhận giá trị của nó, như bfill của pandas ở đó). Độ lệch chuẩn
// mẫu (ddof=1) khớp các số "trung bình ± độ lệch chuẩn qua các seed" trong README / final_test_metrics.json.
function convergenceStats(rows) {
  const { seed: iS, trial: iT, val_rmse: iV, is_feasible: iF } = opt.idx;
  const bySeed = new Map();
  for (const r of rows) {
    if (!bySeed.has(r[iS])) bySeed.set(r[iS], []);
    bySeed.get(r[iS]).push(r);
  }
  const curves = [];
  for (const list of bySeed.values()) {
    list.sort((a, b) => a[iT] - b[iT]);
    let best = NaN;
    const curve = list.map((r) => {
      if (r[iF] && !(r[iV] >= best)) best = r[iV];
      return best;
    });
    const first = curve.find((v) => !Number.isNaN(v));
    if (first === undefined) continue; // seed không có trial khả thi nào thì không có đường cong
    for (let i = 0; i < curve.length && Number.isNaN(curve[i]); i++) curve[i] = first;
    curves.push(curve);
  }
  const n = Math.min(...curves.map((c) => c.length));
  const mean = [];
  const std = [];
  for (let t = 0; t < n; t++) {
    const vals = curves.map((c) => c[t]);
    const m = vals.reduce((a, b) => a + b, 0) / vals.length;
    const v = vals.length > 1 ? vals.reduce((a, b) => a + (b - m) ** 2, 0) / (vals.length - 1) : 0;
    mean.push(m);
    std.push(Math.sqrt(v));
  }
  return { mean, std, seeds: curves.length };
}

// Tập không bị trội trên toàn bộ trial trong log (RMSE và P95 đều càng thấp càng tốt).
function paretoFrontier(rows) {
  const { val_rmse: iV, p95_latency_ms: iP } = opt.idx;
  const sorted = [...rows].sort((a, b) => a[iP] - b[iP] || a[iV] - b[iV]);
  const front = [];
  let bestRmse = Infinity;
  for (const r of sorted) {
    if (r[iV] < bestRmse) {
      bestRmse = r[iV];
      front.push({ x: r[iP], y: r[iV] });
    }
  }
  return front;
}

// Cùng quy tắc với feasible_mask trong src/evaluation/epsilon_scenarios.py: tại ngưỡng chính thức,
// cờ is_feasible trong log quyết định (cờ này được tính trên P95 chưa làm tròn mà bộ tối ưu đã thấy).
function meetsEps(row, eps) {
  return eps === opt.data.constraint_ms ? Boolean(row[opt.idx.is_feasible]) : row[opt.idx.p95_latency_ms] <= eps;
}

function scenarioRows(eps) {
  return Object.fromEntries(opt.data.scenarios.filter((r) => r.epsilon_ms === eps).map((r) => [r.method, r]));
}

function configText(r) {
  return `L=${r.L}h · ${r.n_estimators} cây · depth ${r.max_depth} · ${r.num_leaves} lá · lr ${fmt(r.learning_rate, 4)} · min_child ${r.min_child_samples}`;
}

function epsTag(eps) {
  if (eps === opt.data.constraint_ms) return "SLA";
  if (eps === opt.data.safety_margin_ms) return "biên an toàn";
  return "";
}

/* ---------- Kết quả tối ưu: ô thống kê & bảng ---------- */

function renderBudget() {
  const { seed: iS } = opt.idx;
  const rows = opt.rows.OptunaTPE;
  const seeds = new Set(rows.map((r) => r[iS])).size;
  $("budget-text").textContent = `${rows.length / seeds} trials × ${seeds} seeds = ${rows.length}`;
}

function renderKPIs() {
  const summary = opt.data.summary;
  const grid = $("kpi-grid");
  if (!summary) {
    grid.innerHTML = `<p class="kpi-empty">Chưa có <code>results/final_test_metrics.json</code>. Chạy <code>python src/evaluation/evaluate_final.py</code> để có thống kê.</p>`;
    return;
  }
  const c = summary.optimization_comparison_across_seeds;
  const p1 = c.feasible_rate_comparison.phase_1_budget_40_trials;
  const p2 = c.feasible_rate_comparison.phase_2_budget_100_trials.threshold_5_0_ms;
  const perSeed = c.feasible_rate_comparison.phase_2_budget_100_trials.per_seed;
  const nSeeds = c.seeds_evaluated.length;

  // Mọi giá trị bên dưới là số từ báo cáo JSON, định dạng tại đây (không có văn bản tự do từ dữ liệu).
  const tiles = [
    {
      label: `Tỷ lệ trial khả thi · ${c.total_trials_per_method} trials mỗi phương pháp`,
      value: `${fmt(p2.optuna_feasible_rate_pct, 1)}%`,
      compare: `Random Search ${fmt(p2.random_feasible_rate_pct, 1)}%`,
      delta: `${signed(p2.gain_percentage_points, 1)} điểm %`,
      good: p2.gain_percentage_points > 0,
    },
    {
      label: "Tỷ lệ trial khả thi · Pha 1 (40 trials/seed)",
      value: `${fmt(p1.optuna_feasible_rate_pct, 1)}%`,
      compare: `Random Search ${fmt(p1.random_feasible_rate_pct, 1)}%`,
      delta: `${signed(p1.gain_percentage_points, 1)} điểm %`,
      good: p1.gain_percentage_points > 0,
    },
    {
      label: `Val RMSE tốt nhất · trung bình qua ${nSeeds} seeds`,
      value: `${fmt(c.optuna_tpe_val_rmse_mean, 4)} ± ${fmt(c.optuna_tpe_val_rmse_std, 4)}`,
      compare: `Random Search ${fmt(c.random_search_val_rmse_mean, 4)} ± ${fmt(c.random_search_val_rmse_std, 4)}`,
      note: `Tương đương (paired t, p = ${fmt(c.paired_t_p_value, 3)})`,
    },
    {
      label: "Số trial khả thi theo từng seed",
      value: `Dẫn ${perSeed.seeds_optuna_ahead}/${nSeeds} seeds`,
      compare: `Wilcoxon p = ${fmt(perSeed.wilcoxon_p_value, 4)}`,
      note: `Cùng hướng nhưng chưa đủ ý nghĩa thống kê với ${nSeeds} seeds`,
    },
  ];

  grid.innerHTML = tiles.map((t) => `
    <div class="kpi-tile">
      <span class="kpi-label">${t.label}</span>
      <span class="kpi-value"><i class="key-dot key-tpe" aria-hidden="true"></i><span class="visually-hidden">Optuna TPE: </span>${t.value}</span>
      <span class="kpi-compare"><i class="key-dot key-rs" aria-hidden="true"></i>${t.compare}</span>
      ${t.delta ? `<span class="kpi-delta ${t.good ? "delta-good" : ""}">${t.delta}</span>` : ""}
      ${t.note ? `<span class="kpi-note">${t.note}</span>` : ""}
    </div>`).join("");
}

function renderConvergenceTable() {
  const tpe = opt.convergence.OptunaTPE;
  const rs = opt.convergence.RandomSearch;
  const rows = tpe.mean.map((_, t) => `
    <tr>
      <th scope="row" class="num">${t + 1}</th>
      <td class="num">${fmt(tpe.mean[t], 4)} ± ${fmt(tpe.std[t], 4)}</td>
      <td class="num">${fmt(rs.mean[t], 4)} ± ${fmt(rs.std[t], 4)}</td>
    </tr>`).join("");
  $("conv-table").innerHTML = `
    <thead><tr><th scope="col" class="num">Trial</th><th scope="col" class="num">Optuna TPE</th><th scope="col" class="num">Random Search</th></tr></thead>
    <tbody>${rows}</tbody>`;
}

function renderScenarioTable() {
  const rows = opt.data.scenarios.map((r) => {
    const label = METHODS[r.method].label;
    const has = r.val_rmse !== null;
    return `
      <tr data-eps="${r.epsilon_ms}">
        <td class="num">${fmtEps(r.epsilon_ms)} ms</td>
        <th scope="row"><i class="key-dot key-${METHODS[r.method].cls}" aria-hidden="true"></i>${label}</th>
        <td class="num">${r.feasible_count} / ${r.total_trials} (${fmt(r.feasible_rate_pct, 1)}%)</td>
        <td class="num">${has ? fmt(r.val_rmse, 4) : "—"}</td>
        <td class="cfg">${has ? configText(r) : "Không có trial khả thi"}</td>
        <td class="num">${has ? `${fmt(r.p95_latency_ms, 3)} ms` : "—"}</td>
      </tr>`;
  }).join("");
  $("scenario-table").innerHTML = `
    <thead><tr>
      <th scope="col" class="num">ε</th><th scope="col">Phương pháp</th><th scope="col" class="num">Trial khả thi</th>
      <th scope="col" class="num">Val RMSE tốt nhất</th><th scope="col">Cấu hình được chọn</th><th scope="col" class="num">P₉₅ (pha tìm kiếm)</th>
    </tr></thead>
    <tbody>${rows}</tbody>`;
}

/* ---------- Bộ chọn epsilon & thẻ quyết định ---------- */

function buildEpsControl() {
  const group = $("eps-control");
  const levels = [...new Set(opt.data.scenarios.map((r) => r.epsilon_ms))].sort((a, b) => a - b);
  group.replaceChildren();
  for (const eps of levels) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "seg-btn";
    btn.setAttribute("role", "radio");
    btn.dataset.eps = String(eps);
    const tag = epsTag(eps);
    btn.innerHTML = `<span class="seg-main">${fmtEps(eps)} ms</span>${tag ? `<span class="seg-sub">${tag}</span>` : ""}`;
    btn.addEventListener("click", () => selectEps(eps));
    group.appendChild(btn);
  }
  // Điều hướng bàn phím kiểu radio group: phím mũi tên đổi lựa chọn, cả nhóm chỉ một điểm dừng Tab.
  group.addEventListener("keydown", (e) => {
    const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
    if (!step) return;
    e.preventDefault();
    const i = levels.indexOf(opt.eps);
    const next = levels[(i + step + levels.length) % levels.length];
    selectEps(next);
    group.querySelector(`[data-eps="${next}"]`).focus();
  });
}

function selectEps(eps) {
  opt.eps = eps;
  document.querySelectorAll("#eps-control .seg-btn").forEach((btn) => {
    const on = Number(btn.dataset.eps) === eps;
    btn.setAttribute("aria-checked", on ? "true" : "false");
    btn.tabIndex = on ? 0 : -1;
  });
  document.querySelectorAll("#scenario-table tbody tr").forEach((tr) => {
    tr.classList.toggle("is-current", Number(tr.dataset.eps) === eps);
  });
  renderDecision(eps);
  updateScatter(eps);
}

function renderDecision(eps) {
  const rows = scenarioRows(eps);
  const tpe = rows.OptunaTPE;
  const champ = scenarioRows(opt.data.constraint_ms).OptunaTPE;
  const has = tpe && tpe.val_rmse !== null;
  const tag = epsTag(eps);

  let note;
  if (!has) {
    const { p95_latency_ms: iP } = opt.idx;
    const floor = Math.min(...METHOD_ORDER.flatMap((m) => opt.rows[m].map((r) => r[iP])));
    const deployP95 = opt.data.summary?.latency_compliance?.test_p95_latency_ms;
    note = `Không trial nào trong log có P₉₅ ≤ ${fmtEps(eps)} ms: sàn đo pha tìm kiếm là ${fmt(floor, 3)} ms.`
      + (deployP95 ? ` Nghiệm được chọn đo trên đường triển khai chỉ ${fmt(deployP95, 3)} ms (bước 4), nên kịch bản này cần chạy lại tìm kiếm bằng code hiện tại, không thể kết luận là không khả thi.` : "");
  } else if (eps === opt.data.constraint_ms) {
    note = "Đây là nghiệm được triển khai và kiểm chứng ở bước 4.";
  } else if (champ && tpe.seed === champ.seed && tpe.trial === champ.trial) {
    note = `Trùng nghiệm ở ε = ${fmtEps(opt.data.constraint_ms)} ms: mức ngân sách này chưa làm thay đổi lựa chọn.`;
  } else if (champ) {
    const dRmse = tpe.val_rmse - champ.val_rmse;
    const dLat = (champ.p95_latency_ms - tpe.p95_latency_ms) / champ.p95_latency_ms * 100;
    note = `So với nghiệm ở ε = ${fmtEps(opt.data.constraint_ms)} ms: Val RMSE ${dRmse >= 0 ? "cao hơn" : "thấp hơn"} ${fmt(Math.abs(dRmse), 4)} mm/h `
      + `(${fmt(Math.abs(dRmse) / champ.val_rmse * 100, 2)}%), còn P₉₅ pha tìm kiếm giảm ${fmt(dLat, 1)}%.`;
  }

  const compare = METHOD_ORDER.map((m) => {
    const r = rows[m];
    const meta = METHODS[m];
    return `
      <div class="compare-row">
        <div class="compare-name"><i class="key-dot key-${meta.cls}" aria-hidden="true"></i>${meta.label}</div>
        <div class="compare-stat">
          <span><strong>${r.feasible_count}</strong> / ${r.total_trials} trial khả thi (${fmt(r.feasible_rate_pct, 1)}%)</span>
          <div class="rate-meter rate-${meta.cls}" aria-hidden="true"><span style="width:${r.feasible_rate_pct}%"></span></div>
          <span class="compare-best">Val RMSE tốt nhất: <strong>${r.val_rmse !== null ? fmt(r.val_rmse, 4) : "—"}</strong></span>
        </div>
      </div>`;
  }).join("");

  $("decision-card").innerHTML = `
    <div class="decision-head">
      <span class="header-badge badge-plain">ε = ${fmtEps(eps)} ms${tag ? ` · ${tag}` : ""}</span>
    </div>
    <h3 class="decision-title">${has ? "Quyết định: cấu hình của Optuna TPE" : "Không có nghiệm khả thi trong log"}</h3>
    ${has ? `
      <ul class="config-chips">
        <li>L = ${tpe.L} h</li><li>${tpe.n_estimators} cây</li><li>depth ${tpe.max_depth}</li>
        <li>${tpe.num_leaves} lá</li><li>lr ${fmt(tpe.learning_rate, 4)}</li><li>min_child ${tpe.min_child_samples}</li>
      </ul>
      <div class="decision-metrics">
        <div><span class="dm-label">Val RMSE</span><span class="dm-value">${fmt(tpe.val_rmse, 4)}</span><span class="dm-unit">mm/h</span></div>
        <div><span class="dm-label">P₉₅ pha tìm kiếm</span><span class="dm-value">${fmt(tpe.p95_latency_ms, 3)}</span><span class="dm-unit">ms</span></div>
      </div>
      <p class="decision-source">seed ${tpe.seed}, trial ${tpe.trial}</p>` : ""}
    <p class="decision-note">${note}</p>
    <h4 class="compare-title">Hai phương pháp ở mức ε này</h4>
    ${compare}`;
}

/* ---------- Thành phần cơ bản của biểu đồ SVG ---------- */

const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs, parent) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, v);
  if (parent) parent.appendChild(el);
  return el;
}

function linearScale(d0, d1, r0, r1) {
  const f = (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
  f.invert = (p) => d0 + ((p - r0) / (r1 - r0)) * (d1 - d0);
  return f;
}

function logScale(d0, d1, r0, r1) {
  const l0 = Math.log10(d0);
  const l1 = Math.log10(d1);
  return (v) => r0 + ((Math.log10(v) - l0) / (l1 - l0)) * (r1 - r0);
}

function niceStep(range, count) {
  const raw = range / count;
  const p = 10 ** Math.floor(Math.log10(raw));
  const f = raw / p;
  return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p;
}

function niceDomain(lo, hi, count) {
  const step = niceStep(hi - lo, count);
  const d0 = Math.floor(lo / step) * step;
  const d1 = Math.ceil(hi / step) * step;
  const ticks = [];
  for (let v = d0; v <= d1 + step / 2; v += step) ticks.push(Number(v.toFixed(10)));
  return { d0, d1, ticks, digits: Math.max(0, -Math.floor(Math.log10(step))) };
}

// Mỗi biểu đồ một khung: SVG có kích thước theo container tính bằng pixel CSS (để chữ hiển thị đúng
// cỡ thật), một hình chữ nhật vùng vẽ nằm trong lề, và một tooltip HTML đặt chồng lên trên.
function chartFrame(container, height, margin, label) {
  const width = Math.max(280, Math.floor(container.clientWidth));
  container.replaceChildren();
  const svg = svgEl("svg", { width, height, viewBox: `0 0 ${width} ${height}`, class: "chart-svg", role: "img", "aria-label": label }, container);
  const tooltip = document.createElement("div");
  tooltip.className = "chart-tooltip";
  tooltip.hidden = true;
  container.appendChild(tooltip);
  return { svg, tooltip, width, height, x0: margin.left, x1: width - margin.right, y0: height - margin.bottom, y1: margin.top };
}

function drawGridY(f, y, ticks, digits) {
  const g = svgEl("g", { class: "axis axis-y" }, f.svg);
  for (const t of ticks) {
    const py = y(t);
    svgEl("line", { x1: f.x0, x2: f.x1, y1: py, y2: py, class: "grid" }, g);
    svgEl("text", { x: f.x0 - 8, y: py, dy: "0.32em", "text-anchor": "end", class: "tick" }, g).textContent = fmt(t, digits);
  }
}

function drawAxisX(f, x, ticks, labelFn) {
  const g = svgEl("g", { class: "axis axis-x" }, f.svg);
  svgEl("line", { x1: f.x0, x2: f.x1, y1: f.y0, y2: f.y0, class: "baseline" }, g);
  for (const t of ticks) {
    svgEl("text", { x: x(t), y: f.y0 + 18, "text-anchor": "middle", class: "tick" }, g).textContent = labelFn(t);
  }
}

function drawAxisTitles(f, yTitle, xTitle) {
  svgEl("text", { x: 4, y: 12, class: "axis-title" }, f.svg).textContent = yTitle;
  svgEl("text", { x: (f.x0 + f.x1) / 2, y: f.height - 4, "text-anchor": "middle", class: "axis-title" }, f.svg).textContent = xTitle;
}

// Giữ tooltip trong khung biểu đồ: khi gần mép phải thì lật sang bên trái con trỏ.
function placeTooltip(f, px, py) {
  const tip = f.tooltip;
  tip.hidden = false;
  const w = tip.offsetWidth;
  const left = px + 14 + w > f.width ? px - 14 - w : px + 14;
  tip.style.left = `${Math.max(0, left)}px`;
  tip.style.top = `${Math.max(0, py)}px`;
}

function tooltipRow(cls, value, label) {
  return `<div class="tip-row"><i class="key-line key-${cls}" aria-hidden="true"></i><strong>${value}</strong><span>${label}</span></div>`;
}

/* ---------- Biểu đồ: hội tụ (tốt nhất tính đến hiện tại) ---------- */

function drawConvergence() {
  const container = $("conv-chart");
  const f = chartFrame(container, 300, { top: 26, right: 18, bottom: 44, left: 58 },
    "Đường hội tụ Val RMSE tốt nhất theo số trial của Optuna TPE và Random Search");
  const series = METHOD_ORDER.map((m) => ({ m, ...opt.convergence[m], cls: METHODS[m].cls }));
  const n = Math.min(...series.map((s) => s.mean.length));

  const lo = Math.min(...series.flatMap((s) => s.mean.map((v, i) => v - s.std[i])));
  const hi = Math.max(...series.flatMap((s) => s.mean.map((v, i) => v + s.std[i])));
  const yd = niceDomain(lo, hi, 5);
  const y = linearScale(yd.d0, yd.d1, f.y0, f.y1);
  const x = linearScale(1, n, f.x0, f.x1);

  drawGridY(f, y, yd.ticks, yd.digits);
  const xTicks = [1, ...Array.from({ length: Math.floor(n / 20) }, (_, i) => (i + 1) * 20)].filter((t) => t <= n);
  drawAxisX(f, x, xTicks, String);
  drawAxisTitles(f, "Val RMSE (mm/h)", "Số trial đã chạy (mỗi seed)");

  // Hết Pha 1 (so sánh cùng ngân sách)
  if (n >= 40) {
    svgEl("line", { x1: x(40), x2: x(40), y1: f.y1, y2: f.y0, class: "ref-line" }, f.svg);
    svgEl("text", { x: x(40) + 6, y: f.y1 + 10, class: "ref-label" }, f.svg).textContent = "Hết Pha 1 (40 trials)";
  }

  // Vẽ dải trước, rồi tới đường; phương pháp chính được vẽ sau cùng để nằm trên cùng.
  const drawOrder = [...series].reverse();
  for (const s of drawOrder) {
    const upper = s.mean.slice(0, n).map((v, i) => `${x(i + 1)},${y(v + s.std[i])}`);
    const lower = s.mean.slice(0, n).map((v, i) => `${x(i + 1)},${y(v - s.std[i])}`).reverse();
    svgEl("path", { d: `M${upper.join("L")}L${lower.join("L")}Z`, class: `band band-${s.cls}` }, f.svg);
  }
  for (const s of drawOrder) {
    const d = s.mean.slice(0, n).map((v, i) => `${i ? "L" : "M"}${x(i + 1)},${y(v)}`).join("");
    svgEl("path", { d, class: `line line-${s.cls}` }, f.svg);
  }

  // Lớp crosshair: bám vào trial gần nhất và đọc giá trị của mọi chuỗi dữ liệu (series) tại X đó.
  const cross = svgEl("g", { class: "crosshair", visibility: "hidden" }, f.svg);
  const vline = svgEl("line", { y1: f.y1, y2: f.y0, class: "cross-line" }, cross);
  const dots = Object.fromEntries(series.map((s) => [s.m, svgEl("circle", { r: 4.5, class: `dot dot-${s.cls}` }, cross)]));
  const hit = svgEl("rect", { x: f.x0, y: f.y1, width: f.x1 - f.x0, height: f.y0 - f.y1, class: "hit" }, f.svg);
  let current = n;

  const show = (t) => {
    current = Math.max(1, Math.min(n, t));
    const px = x(current);
    cross.setAttribute("visibility", "visible");
    vline.setAttribute("x1", px);
    vline.setAttribute("x2", px);
    for (const s of series) {
      dots[s.m].setAttribute("cx", px);
      dots[s.m].setAttribute("cy", y(s.mean[current - 1]));
    }
    f.tooltip.innerHTML = `<div class="tip-head">Trial ${current}</div>`
      + series.map((s) => tooltipRow(s.cls, fmt(s.mean[current - 1], 4), `± ${fmt(s.std[current - 1], 4)} · ${METHODS[s.m].label}`)).join("");
    placeTooltip(f, px, f.y1);
  };
  const hide = () => {
    cross.setAttribute("visibility", "hidden");
    f.tooltip.hidden = true;
  };

  hit.addEventListener("pointermove", (e) => {
    const px = e.clientX - f.svg.getBoundingClientRect().left;
    show(Math.round(x.invert(px)));
  });
  hit.addEventListener("pointerleave", hide);
  container.onkeydown = (e) => {
    const step = { ArrowRight: 1, ArrowLeft: -1, Home: -n, End: n }[e.key];
    if (!step) return;
    e.preventDefault();
    show(current + step);
  };
  container.onfocus = () => show(current);
  container.onblur = hide;
}

/* ---------- Biểu đồ: không gian nghiệm với ràng buộc epsilon ---------- */

function drawScatter() {
  const container = $("pareto-chart");
  const f = chartFrame(container, 360, { top: 26, right: 18, bottom: 44, left: 58 },
    "Biểu đồ phân tán Val RMSE theo độ trễ P95 của 1.000 trial, có đường ngân sách epsilon");
  const { val_rmse: iV, p95_latency_ms: iP } = opt.idx;
  const all = METHOD_ORDER.flatMap((m) => opt.rows[m]);

  const maxP = Math.max(...all.map((r) => r[iP]));
  const xDomain = [1, Math.max(10, maxP * 1.15)];
  const x = logScale(xDomain[0], xDomain[1], f.x0, f.x1);
  const yd = niceDomain(Math.min(...all.map((r) => r[iV])), Math.max(...all.map((r) => r[iV])), 5);
  const y = linearScale(yd.d0, yd.d1, f.y0, f.y1);

  // Vùng vi phạm (bên phải epsilon) nằm dưới mọi thành phần khác.
  const zone = svgEl("rect", { y: f.y1, height: f.y0 - f.y1, class: "zone-violate" }, f.svg);
  drawGridY(f, y, yd.ticks, yd.digits);
  const xTicks = [1, 2, 5, 10, 20, 50, 100, 200, 500].filter((t) => t >= xDomain[0] && t <= xDomain[1]);
  drawAxisX(f, x, xTicks, String);
  drawAxisTitles(f, "Val RMSE (mm/h)", "P₉₅ độ trễ pha tìm kiếm (ms, thang log)");

  // Vẽ Random Search trước để các điểm của phương pháp chính nằm trên cùng.
  const points = [];
  const gPts = svgEl("g", { class: "points" }, f.svg);
  for (const m of [...METHOD_ORDER].reverse()) {
    for (const r of opt.rows[m]) {
      const px = x(r[iP]);
      const py = y(r[iV]);
      const el = svgEl("circle", { cx: px, cy: py, r: 4, class: `pt pt-${METHODS[m].cls}` }, gPts);
      points.push({ m, r, px, py, el });
    }
  }

  // Biên Pareto bám sát dải điểm dày nhất, nên được vẽ đè lên các điểm để vẫn nhìn thấy được.
  const front = opt.pareto.map((p) => ({ px: x(p.x), py: y(p.y) }));
  let d = `M${front[0].px},${front[0].py}`;
  for (let i = 1; i < front.length; i++) d += `H${front[i].px}V${front[i].py}`;
  svgEl("path", { d, class: "pareto-line" }, f.svg);

  const epsLine = svgEl("line", { y1: f.y1, y2: f.y0, class: "eps-line" }, f.svg);
  const epsLabel = svgEl("text", { y: f.y1 - 8, class: "eps-label" }, f.svg);
  // Vẽ vòng của phương pháp chính sau cùng để nó nằm trên khi hai điểm được chọn ở sát nhau.
  const rings = Object.fromEntries([...METHOD_ORDER].reverse().map((m) => [m, svgEl("circle", { r: 8, class: `sel-ring sel-${METHODS[m].cls}` }, f.svg)]));
  const hover = svgEl("circle", { r: 6, class: "pt-hover", visibility: "hidden" }, f.svg);
  const hit = svgEl("rect", { x: f.x0, y: f.y1, width: f.x1 - f.x0, height: f.y0 - f.y1, class: "hit" }, f.svg);

  // Hover theo điểm gần nhất: con trỏ chỉ cần ở gần điểm nhất (trong vòng 24px), không cần trỏ trúng điểm.
  hit.addEventListener("pointermove", (e) => {
    const box = f.svg.getBoundingClientRect();
    const mx = e.clientX - box.left;
    const my = e.clientY - box.top;
    let best = null;
    let bestD = 24 * 24;
    for (const p of points) {
      const dd = (p.px - mx) ** 2 + (p.py - my) ** 2;
      if (dd < bestD) { bestD = dd; best = p; }
    }
    if (!best) { hover.setAttribute("visibility", "hidden"); f.tooltip.hidden = true; return; }
    const meta = METHODS[best.m];
    const r = best.r;
    const cfg = Object.fromEntries(opt.data.history.columns.map((c, i) => [c, r[i]]));
    hover.setAttribute("cx", best.px);
    hover.setAttribute("cy", best.py);
    hover.setAttribute("class", `pt-hover pt-${meta.cls}`);
    hover.setAttribute("visibility", "visible");
    const ok = meetsEps(r, opt.eps);
    f.tooltip.innerHTML = `
      <div class="tip-head">${meta.label} · seed ${cfg.seed}, trial ${cfg.trial}</div>
      ${tooltipRow(meta.cls, fmt(cfg.val_rmse, 4), "Val RMSE (mm/h)")}
      <div class="tip-row tip-plain"><strong>${fmt(cfg.p95_latency_ms, 3)} ms</strong><span>P₉₅ pha tìm kiếm</span></div>
      <div class="tip-cfg">${configText(cfg)}</div>
      <div class="tip-status ${ok ? "is-ok" : "is-bad"}">${ok ? "✓ Thỏa" : "✕ Vi phạm"} ε = ${fmtEps(opt.eps)} ms</div>`;
    placeTooltip(f, best.px, Math.max(f.y1, best.py - 40));
  });
  hit.addEventListener("pointerleave", () => {
    hover.setAttribute("visibility", "hidden");
    f.tooltip.hidden = true;
  });

  opt.scatter = { f, x, y, points, zone, epsLine, epsLabel, rings };
}

function updateScatter(eps) {
  const s = opt.scatter;
  if (!s) return;
  const px = s.x(eps);
  s.zone.setAttribute("x", px);
  s.zone.setAttribute("width", Math.max(0, s.f.x1 - px));
  s.epsLine.setAttribute("x1", px);
  s.epsLine.setAttribute("x2", px);
  const nearRight = px > s.f.x1 - 90;
  s.epsLabel.setAttribute("x", nearRight ? px - 6 : px + 6);
  s.epsLabel.setAttribute("text-anchor", nearRight ? "end" : "start");
  s.epsLabel.textContent = `ε = ${fmtEps(eps)} ms`;

  for (const p of s.points) p.el.classList.toggle("is-out", !meetsEps(p.r, eps));

  const rows = scenarioRows(eps);
  for (const m of METHOD_ORDER) {
    const r = rows[m];
    const ring = s.rings[m];
    if (r && r.val_rmse !== null) {
      ring.setAttribute("cx", s.x(r.p95_latency_ms));
      ring.setAttribute("cy", s.y(r.val_rmse));
      ring.setAttribute("visibility", "visible");
    } else {
      ring.setAttribute("visibility", "hidden");
    }
  }
}

function observeChartResize() {
  let frame = 0;
  let lastWidth = $("conv-chart").clientWidth;
  const ro = new ResizeObserver(() => {
    const w = $("conv-chart").clientWidth;
    if (w === lastWidth) return;
    lastWidth = w;
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(() => {
      drawConvergence();
      drawScatter();
      updateScatter(opt.eps);
    });
  });
  ro.observe($("conv-chart"));
  ro.observe($("pareto-chart"));
}

/* ---------- Khởi động ---------- */

document.addEventListener("DOMContentLoaded", () => {
  buildSelects();
  buildPresets();
  bindInputs();
  bindStepNav();
  syncInputsFromState();
  syncPresetUI();
  loadStatus();
  loadOptimization();
  loadModelTable();
  requestPrediction();
});
