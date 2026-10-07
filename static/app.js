window.addEventListener("error", (e) => {
  const banner = document.getElementById("statusPill");
  if (banner) { banner.textContent = "JS ERROR"; banner.className = "pill pill-bad"; }
  console.error("Uncaught error:", e.message, e.filename, e.lineno);
  const log = document.getElementById("log");
  if (log) {
    const line = document.createElement("div");
    line.className = "lost";
    line.textContent = `JS ERROR: ${e.message} (${e.filename}:${e.lineno})`;
    log.appendChild(line);
  }
});

let ws = null;
let ready = false;
let lostCount = 0;

const el = (id) => document.getElementById(id);
const WAVEFORM_MAX_POINTS = 2400;

// ---------------------------------------------------------- rolling waves --
function makeWaveState() {
  return { data: [], lost: [], x: 0 };
}
const waveTxState = makeWaveState();
const waveArState = makeWaveState();
const waveNnState = makeWaveState();
const diffState = makeWaveState();

function makeWaveChart(canvasId, color, state, yMin = -0.2, yMax = 0.2) {
  return new Chart(el(canvasId), {
    type: "line",
    data: { datasets: [{
      data: state.data, borderWidth: 1.1, pointRadius: 0, tension: 0,
      segment: { borderColor: (ctx) => state.lost[ctx.p0DataIndex] ? "#e74c3c" : color },
    }]},
    options: {
      animation: false, parsing: false,
      scales: {
        x: { type: "linear", ticks: { display: false }, grid: { color: "#1a232d" } },
        y: { min: yMin, max: yMax, ticks: { color: "#8b98a5", font: { size: 10 } }, grid: { color: "#1a232d" } },
      },
      plugins: { legend: { display: false } },
    },
  });
}

const waveTxChart = makeWaveChart("waveTx", "#4f9dff", waveTxState);
const waveArChart = makeWaveChart("waveAr", "#e8871e", waveArState);
const waveNnChart = makeWaveChart("waveNn", "#2ecc71", waveNnState);
const diffChart = makeWaveChart("diffChart", "#9b6bff", diffState, -0.05, 0.05);

const snrChart = new Chart(el("snrChart"), {
  type: "line",
  data: { datasets: [
    { label: "This LPC", data: [], borderColor: "#e8871e", pointRadius: 0, borderWidth: 1.5, tension: 0.1 },
    { label: "This NN",  data: [], borderColor: "#2ecc71", pointRadius: 0, borderWidth: 1.5, tension: 0.1 },
  ]},
  options: {
    animation: false, parsing: false,
    scales: {
      x: { type: "linear", ticks: { color: "#8b98a5", font: { size: 10 } }, grid: { color: "#1a232d" } },
      y: { ticks: { color: "#8b98a5", font: { size: 10 } }, grid: { color: "#1a232d" } },
    },
    plugins: { legend: { display: true, labels: { color: "#8b98a5", font: { size: 10 } } } },
  },
});

// Zoomed crossfade-boundary charts: not rolling -- each update fully replaces
// the data with the latest splice window, x-axis centered on the boundary (0ms).
// Same shape as the earlier matplotlib "crossfade effect at the gap boundary"
// plot: a dashed line at the splice point, a shaded crossfade-window box, and
// (on the AR chart) a "no crossfade" counterfactual overlaid for comparison.
function makeBoundaryChart(canvasId, color, withNoCrossOverlay) {
  const datasets = [{ label: "With crossfade", data: [], borderColor: color, pointRadius: 0, borderWidth: 1.4, tension: 0 }];
  if (withNoCrossOverlay) {
    datasets.push({ label: "No crossfade (hard splice)", data: [], borderColor: "#8b98a5",
                    borderDash: [4, 3], pointRadius: 0, borderWidth: 1.2, tension: 0 });
  }
  return new Chart(el(canvasId), {
    type: "line",
    data: { datasets },
    options: {
      animation: false, parsing: false,
      scales: {
        x: { type: "linear", title: { display: true, text: "ms from splice", color: "#8b98a5" },
             ticks: { color: "#8b98a5", font: { size: 10 } }, grid: { color: "#1a232d" } },
        y: { ticks: { color: "#8b98a5", font: { size: 10 } }, grid: { color: "#1a232d" } },
      },
      plugins: {
        legend: { display: withNoCrossOverlay, labels: { color: "#8b98a5", font: { size: 10 } } },
        annotation: { annotations: {
          spliceLine: { type: "line", xMin: 0, xMax: 0, borderColor: "#8b98a5", borderWidth: 1, borderDash: [6, 4],
                        label: { display: true, content: "Gap end (splice)", position: "start",
                                 color: "#8b98a5", font: { size: 9 }, backgroundColor: "rgba(0,0,0,0)" } },
          crossfadeBox: { type: "box", xMin: 0, xMax: 0, backgroundColor: "rgba(11,11,11,0.15)", borderWidth: 0 },
        }},
      },
    },
  });
}
const boundaryArChart = makeBoundaryChart("boundaryAr", "#e8871e", true);
const boundaryNnChart = makeBoundaryChart("boundaryNn", "#2ecc71", false);
let sampleRateHz = 44100; // updated from the "ready" message, used to convert boundary samples -> ms

function updateBoundaryChart(chart, samples, halfWidth, crossfadeMs, noCrossSamples) {
  if (!samples) return;
  const msPerSample = 1000 / sampleRateHz;
  chart.data.datasets[0].data = samples.map((y, i) => ({ x: (i - halfWidth) * msPerSample, y }));
  if (noCrossSamples && chart.data.datasets[1]) {
    chart.data.datasets[1].data = noCrossSamples.map((y, i) => ({ x: (i - halfWidth) * msPerSample, y }));
  }
  chart.options.plugins.annotation.annotations.crossfadeBox.xMax = crossfadeMs;
  chart.update();
}

function pushWave(state, samples, lostFlag, chart, maxPoints = WAVEFORM_MAX_POINTS) {
  samples.forEach((y) => {
    state.data.push({ x: state.x, y });
    state.lost.push(lostFlag);
    state.x++;
  });
  while (state.data.length > maxPoints) { state.data.shift(); state.lost.shift(); }
  chart.update("none");
}

// -------------------------------------------------------- spectrograms --
// Simple magma-like colormap: value in [0,1] -> "rgb(r,g,b)"
function magmaColor(t) {
  t = Math.max(0, Math.min(1, t));
  const stops = [
    [0, 0, 4], [81, 18, 124], [183, 55, 121], [252, 137, 97], [252, 253, 191],
  ];
  const seg = t * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(seg));
  const f = seg - i;
  const a = stops[i], b = stops[i + 1];
  const r = Math.round(a[0] + (b[0] - a[0]) * f);
  const g = Math.round(a[1] + (b[1] - a[1]) * f);
  const bl = Math.round(a[2] + (b[2] - a[2]) * f);
  return `rgb(${r},${g},${bl})`;
}

// Frames arrive already in dB, fixed-scale [-80, 0] (same convention as the
// matplotlib comparisons throughout this project: amplitude_to_db(ref=np.max)).
// No per-column renormalization -- one column's brightness means the same
// thing as every other column's, which is what makes it read as a coherent
// spectrogram instead of independently-flickering bars.
const DB_FLOOR = -80, DB_CEIL = 0;

function makeSpectrogram(canvasId) {
  const canvas = el(canvasId);
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  return {
    canvas, ctx,
    pushColumn(dbBins) {
      const w = canvas.width, h = canvas.height;
      const imgData = ctx.getImageData(1, 0, w - 1, h);
      ctx.putImageData(imgData, 0, 0);
      const rowH = h / dbBins.length;
      for (let i = 0; i < dbBins.length; i++) {
        const norm = (dbBins[i] - DB_FLOOR) / (DB_CEIL - DB_FLOOR);
        ctx.fillStyle = magmaColor(norm);
        const y = h - (i + 1) * rowH;
        ctx.fillRect(w - 1, y, 1, Math.ceil(rowH));
      }
    },
    // One packet yields several STFT frames (hop << packet_size) -- pushing
    // them one at a time is what makes the scroll move smoothly instead of
    // jumping a whole packet's width at once.
    pushColumns(frames) {
      frames.forEach((f) => this.pushColumn(f));
    },
  };
}

const specTx = makeSpectrogram("specTx");
const specAr = makeSpectrogram("specAr");
const specNn = makeSpectrogram("specNn");

// -------------------------------------------------------------- log/util --
function logLine(text, cls) {
  const log = el("log");
  const line = document.createElement("div");
  if (cls) line.className = cls;
  line.textContent = text;
  log.appendChild(line);
  if (log.children.length > 300) log.removeChild(log.firstChild);
  log.scrollTop = log.scrollHeight;
}

function setStatusPill(text, cls) {
  const p = el("statusPill");
  p.textContent = text;
  p.className = "pill " + cls;
}

function setNetworkPill(state) {
  const p = el("networkStatePill");
  p.textContent = state === "MANUAL" ? "DROPPING" : "OK";
  p.className = "pill " + (state === "MANUAL" ? "pill-bad" : "pill-good");
}

function setButtons({ init, start, pause, stop, fastForward }) {
  el("initBtn").disabled = !init;
  el("startBtn").disabled = !start;
  el("pauseBtn").disabled = !pause;
  el("stopBtn").disabled = !stop;
  el("fastForwardBtn").disabled = !fastForward;
}

// ----------------------------------------------------------------- setup --
async function loadFileLists() {
  const res = await fetch("/api/files");
  const data = await res.json();
  const fileSelect = el("fileSelect");
  data.files.forEach((f) => {
    const o = document.createElement("option"); o.value = f; o.textContent = f;
    fileSelect.appendChild(o);
  });
}

el("uploadInput").addEventListener("change", () => {
  const input = el("uploadInput");
  el("uploadFileName").textContent = input.files.length ? input.files[0].name : "No file chosen";
});

el("uploadBtn").addEventListener("click", async () => {
  const input = el("uploadInput");
  if (!input.files.length) return;
  const form = new FormData();
  form.append("file", input.files[0]);
  const res = await fetch("/api/upload", { method: "POST", body: form });
  const data = await res.json();
  const opt = document.createElement("option");
  opt.value = data.filename; opt.textContent = data.filename + " (uploaded)";
  el("cleanSelect").appendChild(opt);
  logLine(`Uploaded ${data.filename} -- select it as the clean reference.`);
});

function buildConfig() {
  return {
    order: parseInt(el("cfgOrder").value),
    diagonal_load: parseFloat(el("cfgDiag").value),
    context_dim: parseInt(el("cfgContext").value),
    extra_dim: parseInt(el("cfgExtra").value),
    equal_power: el("cfgEqualPower").checked,
    nn_context_dim: parseInt(el("cfgNnContext").value),
    nn_fade_dim: parseInt(el("cfgNnFade").value),
    buffer_depth_packets: parseInt(el("cfgBufferDepth").value),
    safety_margin_ms: parseFloat(el("cfgSafetyMargin").value),
  };
}

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => logLine("Connected to backend.");
  ws.onclose = () => { logLine("Disconnected."); setStatusPill("IDLE", "pill-idle"); };
  ws.onmessage = (evt) => handleMessage(JSON.parse(evt.data));
}
connect();
loadFileLists();

function resetVisuals() {
  [waveTxState, waveArState, waveNnState, diffState].forEach((s) => { s.data.length = 0; s.lost.length = 0; s.x = 0; });
  [waveTxChart, waveArChart, waveNnChart, diffChart].forEach((c) => c.update());
  snrChart.data.datasets[0].data = [];
  snrChart.data.datasets[1].data = [];
  snrChart.update();
  [specTx, specAr, specNn].forEach((s) => { s.ctx.fillStyle = "#000"; s.ctx.fillRect(0, 0, s.canvas.width, s.canvas.height); });
  [boundaryArChart, boundaryNnChart].forEach((c) => {
    c.data.datasets.forEach((ds) => { ds.data = []; });
    c.update();
  });
  lostCount = 0;
  el("roPacketIdx").textContent = "0";
  el("roLostCount").textContent = "0";
  el("roLostRate").textContent = "0";
  el("roGenAr").textContent = "-";
  el("roGenNn").textContent = "-";
  el("snrArValue").textContent = "N/A";
  el("snrNnValue").textContent = "N/A";
}

function applyCalibration(msg) {
  el("roPacketDur").textContent = msg.packet_duration_ms;
  el("roDeadline").textContent = msg.deadline_ms;
  el("roComputeAr").textContent = msg.measured_compute_ms;
  el("roWaitAr").textContent = msg.wait_budget_ar_ms;
  el("roComputeNn").textContent = msg.measured_nn_compute_ms;
  el("roWaitNn").textContent = msg.wait_budget_nn_ms;
  el("nnWarning").hidden = !msg.nn_overrun;
}

function handleMessage(msg) {
  if (msg.type === "ready") {
    ready = true;
    sampleRateHz = msg.sample_rate;
    applyCalibration(msg);
    el("roPacketTotal").textContent = msg.n_packets;
    setButtons({ init: true, start: true, pause: false, stop: false, fastForward: true });
    setStatusPill("READY", "pill-idle");
    logLine(`Loaded. ${msg.n_packets} packets, ${msg.packet_duration_ms}ms each. Deadline ${msg.deadline_ms}ms. ` +
            `AR wait budget ${msg.wait_budget_ar_ms}ms, NN wait budget ${msg.wait_budget_nn_ms}ms${msg.nn_overrun ? " (OVERRUN)" : ""}.`);

  } else if (msg.type === "calibration") {
    // A parameter changed mid-session (e.g. AR order) -- refresh the timing
    // readouts to match, without touching anything else that's running.
    applyCalibration(msg);

  } else if (msg.type === "tick") {
    renderTick(msg);

  } else if (msg.type === "tick_fast") {
    renderTickFast(msg);

  } else if (msg.type === "done") {
    setButtons({ init: true, start: false, pause: false, stop: false, fastForward: false });
    setStatusPill("DONE", "pill-idle");
    logLine("Simulation complete.");

  } else if (msg.type === "reset_done") {
    resetVisuals();
    el("log").innerHTML = "";
    setButtons({ init: true, start: ready, pause: false, stop: false, fastForward: ready });
    setStatusPill(ready ? "READY" : "IDLE", "pill-idle");

  } else if (msg.type === "error") {
    logLine("ERROR: " + msg.message, "lost");
    setStatusPill("ERROR", "pill-bad");
  }
}

// Fast-forward mode skips waveform/spectrogram payloads entirely -- only
// progress, loss, and running SNR are meaningful to update here.
function renderTickFast(msg) {
  el("roPacketIdx").textContent = msg.packet_index + 1;
  if (msg.lost) lostCount++;
  el("roLostCount").textContent = lostCount;
  el("roLostRate").textContent = (100 * lostCount / (msg.packet_index + 1)).toFixed(1);
  el("roGenAr").textContent = msg.gen_ms_ar;
  el("roGenNn").textContent = msg.gen_ms_nn;

  if (msg.snr_ar_db !== null) el("snrArValue").textContent = msg.snr_ar_db + " dB";
  if (msg.snr_nn_db !== null) el("snrNnValue").textContent = msg.snr_nn_db + " dB";
}

function renderTick(msg) {
  el("roPacketIdx").textContent = msg.packet_index + 1;
  if (msg.lost) lostCount++;
  el("roLostCount").textContent = lostCount;
  el("roLostRate").textContent = (100 * lostCount / (msg.packet_index + 1)).toFixed(1);
  el("roGenAr").textContent = msg.gen_ms_ar;
  el("roGenNn").textContent = msg.gen_ms_nn;
  setNetworkPill(msg.state);

  pushWave(waveTxState, msg.transmitted, msg.lost, waveTxChart);
  pushWave(waveArState, msg.this_lpc, msg.lost, waveArChart);
  pushWave(waveNnState, msg.this_nn, msg.lost, waveNnChart);
  pushWave(diffState, msg.diff, msg.lost, diffChart);

  specTx.pushColumns(msg.spec_tx);
  specAr.pushColumns(msg.spec_lpc);
  specNn.pushColumns(msg.spec_nn);

  if (msg.boundary_ar) {
    const crossfadeMs = (msg.crossfade_samples / sampleRateHz) * 1000;
    updateBoundaryChart(boundaryArChart, msg.boundary_ar, msg.boundary_half_width, crossfadeMs, msg.boundary_ar_nocross);
    updateBoundaryChart(boundaryNnChart, msg.boundary_nn, msg.boundary_half_width, crossfadeMs);
    logLine(`#${msg.packet_index}  crossfade boundary captured (concealment -> real audio)`, "ok");
  }

  if (msg.snr_ar_db !== null) {
    snrChart.data.datasets[0].data.push({ x: msg.packet_index, y: msg.snr_ar_db });
    el("snrArValue").textContent = msg.snr_ar_db + " dB";
  }
  if (msg.snr_nn_db !== null) {
    snrChart.data.datasets[1].data.push({ x: msg.packet_index, y: msg.snr_nn_db });
    el("snrNnValue").textContent = msg.snr_nn_db + " dB";
  }
  [0, 1].forEach((i) => { if (snrChart.data.datasets[i].data.length > 500) snrChart.data.datasets[i].data.shift(); });
  snrChart.update("none");

  // Only log drops -- every "received" packet is now identical (nothing random
  // decides that anymore), so logging each one would just be noise.
  if (msg.lost) {
    logLine(`#${msg.packet_index}  LOST (manual drop)  -> concealed (both methods)`, "lost");
  }
}

// ------------------------------------------------------------- controls --
el("speedSlider").addEventListener("input", (e) => {
  el("speedLabel").textContent = e.target.value + "×";
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "set_speed", slowdown: parseFloat(e.target.value) }));
});

el("initBtn").addEventListener("click", () => {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    logLine("Not connected to backend yet -- try again in a second.", "lost");
    setStatusPill("ERROR", "pill-bad");
    return;
  }
  const file = el("fileSelect").value;
  const clean = el("cleanSelect").value || null;
  if (!file) { logLine("Pick a file first."); return; }
  resetVisuals();
  logLine(`Requesting load: ${file}${clean ? " (clean: " + clean + ")" : ""}`);
  ws.send(JSON.stringify({ type: "init", file, clean_file: clean, config: buildConfig() }));
});

el("startBtn").addEventListener("click", () => {
  ws.send(JSON.stringify({ type: "start" }));
  setButtons({ init: false, start: false, pause: true, stop: true, fastForward: true });
  setStatusPill("RUNNING", "pill-good");
});

el("pauseBtn").addEventListener("click", () => {
  ws.send(JSON.stringify({ type: "pause" }));
  setButtons({ init: true, start: true, pause: false, stop: true, fastForward: true });
  setStatusPill("PAUSED", "pill-idle");
});

el("stopBtn").addEventListener("click", () => {
  ws.send(JSON.stringify({ type: "stop" }));
  setButtons({ init: true, start: true, pause: false, stop: false, fastForward: true });
});

el("fastForwardBtn").addEventListener("click", () => {
  ws.send(JSON.stringify({ type: "fast_forward" }));
  setButtons({ init: false, start: false, pause: true, stop: true, fastForward: false });
  setStatusPill("FAST FORWARDING", "pill-good");
  logLine(">> Fast-forwarding to the end (live plots paused, only progress/SNR update).");
});

document.querySelectorAll(".drop-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const count = parseInt(btn.dataset.count);
    ws.send(JSON.stringify({ type: "drop", count }));
    logLine(`>> Manually dropping next ${count} packet(s).`);
  });
});

el("dropCustomBtn").addEventListener("click", () => {
  const count = parseInt(el("dropCustomCount").value);
  ws.send(JSON.stringify({ type: "drop", count }));
  logLine(`>> Manually dropping next ${count} packet(s).`);
});

["cfgOrder","cfgDiag","cfgContext","cfgExtra","cfgEqualPower","cfgNnContext","cfgNnFade",
 "cfgBufferDepth","cfgSafetyMargin"].forEach((id) => {
  el(id).addEventListener("change", () => {
    if (ws && ws.readyState === WebSocket.OPEN && ready) {
      ws.send(JSON.stringify({ type: "update_config", config: buildConfig() }));
    }
  });
});
