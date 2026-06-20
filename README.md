# NeuroForge

**An interactive neural-network playground where the network is trained by a backpropagation engine written from scratch in vanilla JavaScript — no TensorFlow, no autograd, no dependencies.**

Build a network by hand, drop it on a dataset, hit **Train**, and watch it learn in real time: the decision boundary bends itself around the data, the connection weights thicken and fade as the optimizer works, and the loss/accuracy curves tick down live.

> **Live demo:** 
> The whole thing runs client-side in a single page — view source, and the math is right there.

---

## Why this exists

Most "neural network demos" are a thin UI over a library that does all the real work. NeuroForge is the opposite: **the learning algorithm is the project.** The forward pass, the gradients, the optimizers, the weight initialization — all of it is hand-written and readable in [`src/nn.js`](src/nn.js). The visualizer is wrapped around a real engine, not a toy.

If you want to understand backpropagation, reading 250 lines of commented JavaScript that demonstrably trains a spiral classifier beats another diagram.

## What it demonstrates

- A correct, from-scratch implementation of **backpropagation** through arbitrary fully-connected layers.
- Three optimizers built from the update rules up: **SGD**, **Momentum**, and **Adam** (with bias correction).
- The practical ML intuitions a playground is good at teaching: why depth matters, what feature engineering buys you, how learning rate and regularization change the story, and overfitting (watch the train/test gap).
- Real-time data visualization on raw `<canvas>` — decision-boundary heatmap, a live weight graph, and hand-drawn loss/accuracy charts, all without a charting library.

## Features

- **Five datasets** — spiral, circle, XOR, two gaussians, and interleaving moons, with adjustable noise, sample count, and train/test split.
- **Build the network live** — add/remove hidden layers, change neuron counts, and switch activation (tanh / ReLU / sigmoid). The parameter count updates as you go.
- **Feature engineering** — toggle inputs `X₁, X₂, X₁², X₂², X₁X₂, sin X₁, sin X₂`. Solve the circle with *no hidden layers* once you add the squared terms — a logistic-regression lesson in one click.
- **Full optimizer control** — optimizer, learning rate, batch size, and L2 regularization, all applied live.
- **Live decision boundary** — a diverging heatmap of the network's output across the whole plane, recomputed every frame.
- **Live weight graph** — every connection coloured by sign (amber +, cyan −) and weighted by magnitude, so you literally see the network reorganize itself.
- **Presets** — one-click configurations (e.g. "Solve the spiral") that load a known-good setup and start training.
- **Export** the trained model weights to JSON. Keyboard shortcuts (`space` train, `S` step, `R` reset).

## The engine, briefly

For a single sigmoid-output binary classifier with binary cross-entropy loss, the output-layer gradient collapses to the clean form `dL/dz = (p − y)`. From there the deltas propagate backward through each layer:

```
δ_out = p − y
δ_l   = (Wᵀ_{l+1} · δ_{l+1}) ⊙ act'(z_l)
∂L/∂W_l = δ_l · aᵀ_{l-1}
```

Gradients are averaged over a mini-batch, L2 weight decay is added, and the chosen optimizer applies the update. Weights are initialized with He (ReLU) or Xavier (tanh/sigmoid) scaling from a seeded RNG so any run is reproducible.

## Correctness

The gradients aren't taken on faith. A **numerical gradient check** (central differences vs. the analytic backprop) agrees to a maximum relative error of **~2e-10** — machine precision. The engine trains XOR, the circle, moons, and the two-spiral problem to ~100% train accuracy headlessly. See the verification notes in the commit history / `nn.js`.

## Architecture

```
neuroforge/
├── index.html            # structure
├── styles.css            # design system (one file, hand-written)
└── src/
    ├── nn.js             # the neural network: forward, backprop, optimizers  ← the core
    ├── datasets.js       # dataset generators + feature engineering
    ├── boundary.js       # decision-boundary heatmap renderer
    ├── network-viz.js    # live weight/architecture graph
    ├── charts.js         # loss & accuracy line charts
    └── app.js            # state, controls, and the training loop
```

The modules are plain `<script>`s sharing a small `NF` namespace, so the whole app runs by **opening `index.html` directly** — no build step, no `node_modules`, nothing to install.

## Run it

```bash
# simplest: just open the file
open index.html            # macOS  (or double-click it)

# or serve it (any static server works)
python3 -m http.server 8000
# then visit http://localhost:8000
```

## Deploy

It's a static site, so any host works — Vercel, Netlify, GitHub Pages, Cloudflare Pages. For GitHub Pages, push the folder and enable Pages on the branch; for Vercel/Netlify, point them at the directory with no build command.

## Tech

Vanilla JavaScript (ES2015+), HTML5 Canvas, CSS custom properties. **Zero runtime dependencies.** Fonts (Inter, JetBrains Mono) load from Google Fonts and degrade gracefully to system fonts offline.

## License

MIT — see [LICENSE](LICENSE).
