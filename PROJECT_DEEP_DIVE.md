# NeuroForge — Project Deep Dive

> The README sells the project. This document explains how it actually works — the architecture, the data flow, and the parts that took real thought to get right. If you're reviewing the code (or me), start here.

---

## 1. The one-sentence architecture

A small set of dependency-free ES modules share a global `NF` namespace; `app.js` owns all state and runs a `requestAnimationFrame` loop that trains the network a few epochs per frame and repaints four canvas-based views every frame. There is no framework, no bundler, and no server.

```
                    ┌──────────────────────────────────────────────┐
                    │  app.js  (controller + state + RAF loop)      │
                    └───────────────┬──────────────────────────────┘
            builds/trains           │ reads weights every frame
          ┌────────────────┐        │
          │  nn.js          │◄───────┤   ┌─────────────────────────┐
          │  Network        │        ├──►│ boundary.js (heatmap)   │
          │  forward/back   │        ├──►│ network-viz.js (graph)  │
          │  SGD/Mom/Adam   │        ├──►│ charts.js (loss/acc)    │
          └────────────────┘        │   └─────────────────────────┘
          ┌────────────────┐        │
          │ datasets.js     │◄───────┘
          │ generators +    │  provides train/test points + featurize()
          │ feature eng.    │
          └────────────────┘
```

Each module wraps itself in a tiny UMD-ish factory so it works as a classic `<script>` **and** could be `require()`d in Node for headless testing (the gradient check runs this way).

---

## 2. Folder structure & responsibilities

| File | Responsibility | Notes |
|---|---|---|
| `index.html` | App shell, layout, control markup | Three-column "instrument" grid: controls · stage · inspector |
| `styles.css` | Design system for the app | One file, hand-set tokens, dark-only |
| `pages.css` | Design system for content pages | Mirrors the same tokens; scrollable document layout |
| `about.html` / `about-project.html` | Narrative pages | Share `pages.css`; About-the-project carries the build story |
| `src/nn.js` | **The neural network** | Forward pass, backprop, three optimizers, serialization |
| `src/datasets.js` | Data + feature engineering | 5 generators, 7 input transforms, deterministic RNG |
| `src/boundary.js` | Decision-boundary renderer | Low-res field → upscaled heatmap + scatter |
| `src/network-viz.js` | Live weight graph | Edges coloured by sign, weighted by magnitude |
| `src/charts.js` | Loss/accuracy line charts | Sliding window, auto-scaled, no chart lib |
| `src/app.js` | Controller | State, controls, the training loop, telemetry |

---

## 3. Data flow, end to end

1. **Generate.** `datasets.generate(kind, {n, noise, split, seed})` produces `{x, y, label}` points in a `[-5, 5]²` domain, shuffles them with a seeded RNG, and splits into train/test.
2. **Featurize.** Raw coordinates are passed through `featurize(x, y, enabledIds)`, which maps them to the selected engineered features (e.g. `X₁², sin X₁`) normalised to roughly `[-1, 1]`. This produces the actual input vectors the network sees.
3. **Build.** `app.buildNet()` constructs a `Network` whose input dimension equals the number of enabled features, with the user's hidden layers plus a single sigmoid output unit.
4. **Train.** Each animation frame runs `state.speed` epochs. An epoch shuffles an index array (Fisher–Yates), slices it into mini-batches, and calls `net.trainBatch(Xb, Yb)`.
5. **Render.** The boundary field is recomputed only when weights changed (a `dirtyField` flag), then every view is redrawn from the current weights.
6. **Measure.** While running, `net.evaluate()` computes loss/accuracy on train and test sets; the numbers feed the header telemetry, the metric tiles, and the two charts.

The key invariant: **the boundary renderer feeds plane coordinates through the same `featurize()` the network was trained on.** What you see is, pixel for pixel, what the network believes.

---

## 4. The engine (`nn.js`) in detail

### Representation
- A layer is fully-connected: `a = act(W · a_prev + b)`.
- `W[l]` is `[units, inputDim]`, `b[l]` is `[units]` — plain nested JS arrays, not typed arrays. (Tradeoff: clarity over raw speed; the problem sizes are small enough that it doesn't matter, and the code reads like the math.)
- The output layer is a single sigmoid unit → binary classifier. Loss is binary cross-entropy with optional L2.

### Forward pass
`forward(x, keepCache)` walks the layers, computing `z` (pre-activation) and `a` (activation) for each. When `keepCache` is set it stores `inputs`, `z`, and `a` per layer — exactly what backprop needs.

### Backprop
The clean trick: for sigmoid output + BCE loss, the output-layer error collapses to `δ = (p − y)` — no explicit derivative of the loss or the sigmoid required. From there:

```
δ_l   = (Wᵀ_{l+1} · δ_{l+1}) ⊙ act'(z_l)
∂L/∂W = δ · aᵀ_prev      ∂L/∂b = δ
```

`_backprop()` accumulates per-example gradients into shared `gW`/`gb` buffers; `trainBatch()` averages them over the batch, adds the L2 term (`l2 * w`), and applies the optimizer.

Activation derivatives are expressed in terms of whatever value is cheapest to reuse — `1 − a²` for tanh, `a(1 − a)` for sigmoid, and the pre-activation sign for ReLU.

### Optimizers (built from the update rules up)
- **SGD** — `w -= lr · g`.
- **Momentum** — velocity `m = β₁·m − lr·g`; `w += m`.
- **Adam** — first/second moment estimates with **bias correction** (`m̂`, `v̂`) and the `√v̂ + ε` denominator. The timestep `t` increments per batch so the correction is exact.

### Reproducibility
`mulberry32(seed)` drives both data generation and weight init (He for ReLU, Xavier otherwise, via Box–Muller gaussians). Same seed ⇒ same run. The "Reset" actions bump the seed so you get a *fresh but still deterministic* run.

---

## 5. The hard parts (and how they were solved)

### 5.1 Silently-wrong gradients
**The problem.** First end-to-end run: no error, but the boundary never moved. A wrong-but-not-crashing gradient is the worst class of bug.

**The fix.** Verify the gradients numerically. For each weight, perturb by ±ε and measure the actual change in loss (central difference): `g_num ≈ (L(w+ε) − L(w−ε)) / 2ε`. Compare against the analytic gradient with a relative-error metric. The mismatch localised a batch-averaging error to one line. Post-fix, analytic vs. numerical agree to **~2e-10**, and that check stays in the toolbox as a regression guard. This is the engineering instinct from safety-critical control work: don't trust the clever code — measure it against ground truth.

### 5.2 6,400 forward passes per frame
**The problem.** The heatmap is an 80×80 grid — 6,400 forward passes. Recomputing every frame at full canvas resolution destroys the framerate.

**The fix.** Two moves:
1. Render the field into a tiny **80×80 offscreen canvas**, then `drawImage` it scaled up with smoothing on — the GPU does the interpolation for free.
2. Recompute the field **only when weights change** (`dirtyField`). On idle frames we redraw the cached field and the points, which is cheap. Net result: a steady 60fps while training at speed.

### 5.3 Keeping the picture honest
**The problem.** It's tempting to draw an "approximate" boundary. But a teaching tool that lies is worse than useless.

**The fix.** The renderer calls the network's real `forward()` on real featurized coordinates. Toggle `X₁²` and the boundary becomes a true circle because that's the actual function the net is now fitting — not a visual fake.

### 5.4 One design language for two ideas
Cool (cyan) and warm (amber) map to the two classes **everywhere** — the scatter points, the boundary field, the weight signs, the chart series. That consistency is a deliberate UX decision: the user learns the color code once and it pays off across every view.

---

## 6. Tradeoffs I made on purpose

| Decision | Upside | Cost I accepted |
|---|---|---|
| Zero dependencies | Readable, auditable, instant load, no supply chain | I re-implement charts, RNG, init by hand |
| Nested arrays, not typed arrays | Code mirrors the math notation | Leaves some performance on the table (fine at this scale) |
| Single sigmoid output | The clean `(p − y)` gradient; simple BCE | Binary classification only — no multi-class |
| Train in the RAF loop | Dead-simple, always in sync with rendering | Heavy training would block the main thread (out of scope here) |
| Two CSS files, mirrored tokens | Content pages can scroll without fighting the app's fixed layout | Tokens are duplicated; must keep them in sync |

---

## 7. What I'd build next

See [`INTERVIEW_PREP.md`](INTERVIEW_PREP.md) for the longer version, but in brief: move training to a Web Worker, swap nested arrays for `Float32Array`, add multi-class (softmax + cross-entropy), persist the last config to `localStorage`, and a "replay" scrubber that records weight snapshots so you can scrub through training history.
