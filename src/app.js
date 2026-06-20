/*
 * NeuroForge — application controller
 * -----------------------------------
 * Owns the UI state, builds the network from the controls, runs the
 * requestAnimationFrame training loop, and keeps every view (boundary,
 * network diagram, charts, telemetry) in sync each frame.
 */
(function () {
  "use strict";
  const { Network } = NF.nn;
  const D = NF.data;

  // ---------------- state ----------------
  const state = {
    dataset: "spiral",
    noise: 0.2,
    samples: 320,
    split: 0.5,
    seed: 7,
    features: ["x1", "x2", "sin1", "sin2"],
    layers: [8, 8],
    activation: "tanh",
    optimizer: "adam",
    lr: 0.03,
    batch: 16,
    l2: 0,
    running: false,
    epoch: 0,
    speed: 4,
  };

  let net, trainRaw = [], testRaw = [], drawPoints = [];
  let trainX = [], trainY = [], testX = [], testY = [], order = [];
  let dirtyField = true;

  // ---------------- DOM ----------------
  const $ = (id) => document.getElementById(id);
  const boundary = new NF.Boundary($("boundary"), D.DOMAIN);
  const netViz = new NF.NetworkViz($("network"));
  const lossChart = new NF.LineChart($("lossChart"), {
    series: [{ key: "train", color: "#f5a524" }, { key: "test", color: "#38bdf8", dash: [4, 4] }],
  });
  const accChart = new NF.LineChart($("accChart"), {
    yMin: 0, yMax: 1,
    series: [{ key: "train", color: "#f5a524" }, { key: "test", color: "#38bdf8", dash: [4, 4] }],
  });

  // ---------------- model + data ----------------
  function buildNet() {
    const specs = state.layers
      .map((u) => ({ units: u, activation: state.activation }))
      .concat([{ units: 1, activation: "sigmoid" }]);
    net = new Network(state.features.length, specs, {
      optimizer: state.optimizer,
      learningRate: state.lr,
      l2: state.l2,
      seed: state.seed,
    });
    $("paramCount").textContent = net.paramCount() + " params";
  }

  function regenData() {
    const { train, test } = D.generate(state.dataset, {
      n: state.samples, noise: state.noise, split: state.split, seed: state.seed,
    });
    trainRaw = train; testRaw = test;
    recomputeFeatures();
  }

  function recomputeFeatures() {
    const f = state.features;
    trainX = trainRaw.map((p) => D.featurize(p.x, p.y, f));
    trainY = trainRaw.map((p) => p.label);
    testX = testRaw.map((p) => D.featurize(p.x, p.y, f));
    testY = testRaw.map((p) => p.label);
    order = trainX.map((_, i) => i);
    drawPoints = trainRaw.map((p) => ({ x: p.x, y: p.y, label: p.label, test: false }))
      .concat(testRaw.map((p) => ({ x: p.x, y: p.y, label: p.label, test: true })));
  }

  function rebuildModel() {
    buildNet();
    state.epoch = 0;
    lossChart.reset();
    accChart.reset();
    dirtyField = true;
    refresh();
  }
  function newData() { regenData(); rebuildModel(); }

  // ---------------- training loop ----------------
  function runEpoch() {
    for (let i = order.length - 1; i > 0; i--) {
      const j = (Math.random() * (i + 1)) | 0;
      const t = order[i]; order[i] = order[j]; order[j] = t;
    }
    const bs = state.batch > 0 ? state.batch : order.length;
    for (let s = 0; s < order.length; s += bs) {
      const end = Math.min(s + bs, order.length);
      const Xb = [], Yb = [];
      for (let t = s; t < end; t++) { Xb.push(trainX[order[t]]); Yb.push(trainY[order[t]]); }
      net.trainBatch(Xb, Yb);
    }
    state.epoch++;
  }

  let lastTs = performance.now(), epochAcc = 0, epsShown = 0;
  function frame(ts) {
    if (state.running && trainX.length) {
      for (let i = 0; i < state.speed; i++) runEpoch();
      epochAcc += state.speed;
      dirtyField = true;
    }
    // epochs/sec telemetry
    if (ts - lastTs > 500) {
      epsShown = Math.round((epochAcc * 1000) / (ts - lastTs));
      $("eps").textContent = state.running ? epsShown + " epochs/s" : "paused";
      epochAcc = 0; lastTs = ts;
    }
    if (dirtyField) { boundary.computeField(net, D.featurize, state.features); dirtyField = false; }
    boundary.draw(drawPoints);
    netViz.draw(net, state.features.map((id) => D.FEATURE_BY_ID[id].label));

    if (state.running || pendingMetrics) { updateMetrics(); pendingMetrics = false; }
    requestAnimationFrame(frame);
  }

  let pendingMetrics = true;
  function updateMetrics() {
    const tr = net.evaluate(trainX, trainY);
    const te = net.evaluate(testX, testY);
    lossChart.push({ train: tr.loss, test: te.loss }); lossChart.draw();
    accChart.push({ train: tr.acc, test: te.acc }); accChart.draw();
    $("hEpoch").textContent = state.epoch;
    $("epochBig").textContent = state.epoch;
    $("hLoss").textContent = tr.loss.toFixed(3);
    $("hAcc").textContent = (te.acc * 100).toFixed(1) + "%";
    $("mTrainLoss").textContent = tr.loss.toFixed(3);
    $("mTestLoss").textContent = te.loss.toFixed(3);
    $("mTrainAcc").textContent = (tr.acc * 100).toFixed(1) + "%";
    $("mTestAcc").textContent = (te.acc * 100).toFixed(1) + "%";
  }

  // a lightweight refresh used after non-training edits
  function refresh() { pendingMetrics = true; }

  // ---------------- transport ----------------
  function setRunning(v) {
    state.running = v;
    const btn = $("play");
    btn.classList.toggle("running", v);
    $("playLabel").textContent = v ? "Pause" : "Train";
    if (!v) $("eps").textContent = "paused";
  }
  $("play").onclick = () => setRunning(!state.running);
  $("step").onclick = () => { if (!state.running) { runEpoch(); dirtyField = true; refresh(); } };
  $("reset").onclick = () => { state.seed++; setRunning(false); newData(); };
  $("resetWeights").onclick = () => { state.seed++; rebuildModel(); };
  $("speed").oninput = (e) => { state.speed = +e.target.value; };

  // ---------------- controls: data ----------------
  function buildDatasetTiles() {
    const grid = $("datasetGrid");
    grid.innerHTML = "";
    Object.entries(D.GENERATORS).forEach(([key, def]) => {
      const b = document.createElement("button");
      b.className = "tile" + (key === state.dataset ? " active" : "");
      b.dataset.k = key;
      const c = document.createElement("canvas");
      c.width = 84; c.height = 84;
      const span = document.createElement("span");
      span.textContent = def.label;
      b.appendChild(c); b.appendChild(span);
      b.onclick = () => {
        state.dataset = key;
        [...grid.children].forEach((el) => el.classList.toggle("active", el.dataset.k === key));
        setRunning(false); newData();
      };
      grid.appendChild(b);
      drawMini(c, key);
    });
  }
  function drawMini(canvas, kind) {
    const ctx = canvas.getContext("2d");
    const { all } = D.generate(kind, { n: 90, noise: 0.12, split: 1, seed: 3 });
    const s = canvas.width, dom = D.DOMAIN;
    ctx.clearRect(0, 0, s, s);
    for (const p of all) {
      const px = ((p.x + dom) / (2 * dom)) * s;
      const py = ((dom - p.y) / (2 * dom)) * s;
      ctx.beginPath(); ctx.arc(px, py, 2.1, 0, Math.PI * 2);
      ctx.fillStyle = p.label ? "#fb923c" : "#38bdf8";
      ctx.fill();
    }
  }

  bindRange("noise", "noiseVal", (v) => v.toFixed(2), (v) => { state.noise = v; }, true);
  bindRange("samples", "samplesVal", (v) => String(v), (v) => { state.samples = v; }, true);
  bindRange("split", "splitVal", (v) => Math.round(v * 100) + "%", (v) => { state.split = v; }, true);
  $("regen").onclick = () => { state.seed++; setRunning(false); newData(); };

  function bindRange(id, valId, fmt, set, regen) {
    const el = $(id);
    el.oninput = (e) => {
      const v = +e.target.value;
      set(v);
      $(valId).textContent = fmt(v);
    };
    el.onchange = () => { if (regen) { setRunning(false); newData(); } };
  }

  // ---------------- controls: features ----------------
  function buildFeatureChips() {
    const wrap = $("featureChips");
    wrap.innerHTML = "";
    D.FEATURES.forEach((f) => {
      const c = document.createElement("button");
      c.className = "chip" + (state.features.includes(f.id) ? " on" : "");
      c.textContent = f.label;
      c.onclick = () => {
        const i = state.features.indexOf(f.id);
        if (i >= 0) { if (state.features.length === 1) return; state.features.splice(i, 1); }
        else state.features.push(f.id);
        // keep feature order canonical
        state.features = D.FEATURES.filter((x) => state.features.includes(x.id)).map((x) => x.id);
        c.classList.toggle("on");
        recomputeFeatures(); rebuildModel();
      };
      wrap.appendChild(c);
    });
  }

  // ---------------- controls: architecture ----------------
  function bindSeg(id, set) {
    const seg = $(id);
    seg.querySelectorAll("button").forEach((b) => {
      b.classList.toggle("active", b.dataset.v === seg.dataset.value);
      b.onclick = () => {
        seg.dataset.value = b.dataset.v;
        seg.querySelectorAll("button").forEach((x) => x.classList.toggle("active", x === b));
        set(b.dataset.v);
      };
    });
  }
  bindSeg("activation", (v) => { state.activation = v; rebuildModel(); });
  bindSeg("optimizer", (v) => { state.optimizer = v; rebuildModel(); });

  function renderLayers() {
    const wrap = $("layers");
    wrap.innerHTML = "";
    if (state.layers.length === 0) {
      const empty = document.createElement("div");
      empty.className = "layer-row";
      empty.innerHTML = '<span class="ln">No hidden layers — pure logistic regression</span>';
      wrap.appendChild(empty);
    }
    state.layers.forEach((units, i) => {
      const row = document.createElement("div");
      row.className = "layer-row";
      row.innerHTML = `<span class="ln">Hidden ${i + 1}</span>
        <div class="stepper">
          <button data-d="-1">−</button>
          <span class="num mono">${units}</span>
          <button data-d="1">+</button>
        </div>`;
      row.querySelectorAll("button").forEach((b) => {
        b.onclick = () => {
          const d = +b.dataset.d;
          state.layers[i] = Math.max(1, Math.min(8, state.layers[i] + d));
          renderLayers(); rebuildModel();
        };
      });
      wrap.appendChild(row);
    });
  }
  $("addLayer").onclick = () => { if (state.layers.length < 5) { state.layers.push(4); renderLayers(); rebuildModel(); } };
  $("rmLayer").onclick = () => { if (state.layers.length > 0) { state.layers.pop(); renderLayers(); rebuildModel(); } };

  // ---------------- controls: optimization (live) ----------------
  $("lr").onchange = (e) => { state.lr = +e.target.value; net.lr = state.lr; };
  $("batch").onchange = (e) => { state.batch = +e.target.value; };
  $("l2").onchange = (e) => { state.l2 = +e.target.value; net.l2 = state.l2; };

  // ---------------- presets ----------------
  const PRESETS = [
    { name: "Solve the spiral", note: "8·8 tanh · Adam", cfg: { dataset: "spiral", features: ["x1", "x2", "sin1", "sin2"], layers: [8, 8], activation: "tanh", optimizer: "adam", lr: 0.03, l2: 0 } },
    { name: "XOR · tiny net", note: "1×4 tanh", cfg: { dataset: "xor", features: ["x1", "x2"], layers: [4], activation: "tanh", optimizer: "adam", lr: 0.1, l2: 0 } },
    { name: "Circle, no depth", note: "x² features only", cfg: { dataset: "circle", features: ["x1", "x2", "x1sq", "x2sq"], layers: [], activation: "tanh", optimizer: "adam", lr: 0.1, l2: 0 } },
    { name: "Deep ReLU", note: "8·8·8 · moons", cfg: { dataset: "moons", features: ["x1", "x2"], layers: [8, 8, 8], activation: "relu", optimizer: "adam", lr: 0.03, l2: 0 } },
  ];
  function buildPresets() {
    const wrap = $("presets");
    PRESETS.forEach((p) => {
      const b = document.createElement("button");
      b.className = "preset";
      b.innerHTML = `<b>${p.name}</b><small>${p.note}</small>`;
      b.onclick = () => applyPreset(p.cfg);
      wrap.appendChild(b);
    });
  }
  function applyPreset(cfg) {
    Object.assign(state, JSON.parse(JSON.stringify(cfg)));
    state.seed++;
    syncControlsFromState();
    setRunning(false);
    newData();
    setRunning(true); // instant gratification
  }

  function syncControlsFromState() {
    $("noise").value = state.noise; $("noiseVal").textContent = state.noise.toFixed(2);
    $("samples").value = state.samples; $("samplesVal").textContent = state.samples;
    $("split").value = state.split; $("splitVal").textContent = Math.round(state.split * 100) + "%";
    $("lr").value = String(state.lr);
    $("batch").value = String(state.batch);
    $("l2").value = String(state.l2);
    setSeg("activation", state.activation);
    setSeg("optimizer", state.optimizer);
    buildDatasetTiles();
    buildFeatureChips();
    renderLayers();
  }
  function setSeg(id, v) {
    const seg = $(id); seg.dataset.value = v;
    seg.querySelectorAll("button").forEach((b) => b.classList.toggle("active", b.dataset.v === v));
  }

  // ---------------- export ----------------
  $("exportBtn").onclick = () => {
    const blob = new Blob([JSON.stringify(net.toJSON(), null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "neuroforge-model.json";
    a.click();
    URL.revokeObjectURL(a.href);
  };

  // ---------------- keyboard ----------------
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
    if (e.code === "Space") { e.preventDefault(); setRunning(!state.running); }
    else if (e.key === "s" || e.key === "S") { if (!state.running) { runEpoch(); dirtyField = true; refresh(); } }
    else if (e.key === "r" || e.key === "R") { state.seed++; setRunning(false); newData(); }
  });

  // ---------------- boot ----------------
  buildDatasetTiles();
  buildFeatureChips();
  renderLayers();
  buildPresets();
  regenData();
  buildNet();
  $("paramCount").textContent = net.paramCount() + " params";
  updateMetrics();
  requestAnimationFrame(frame);
})();
