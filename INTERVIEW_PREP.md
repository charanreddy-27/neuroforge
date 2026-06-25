# NeuroForge — Interview Prep

> Rehearse from this. Read it out loud once before a call. The goal isn't to memorize lines — it's to have crisp, honest answers ready so you can stay present in the conversation.

---

## ⏱ The 30-second elevator pitch

> "NeuroForge is an interactive neural-network playground — but the twist is that there's no machine-learning library underneath it. The forward pass, backpropagation, and the optimizers — SGD, Momentum, Adam — are all hand-written in vanilla JavaScript, about 250 lines you can read top to bottom. You build a network in the browser, drop it on a dataset like a spiral, hit Train, and watch the decision boundary bend itself around the data in real time while the weights and loss curves update live. I built it because most ML demos hide the actual learning inside a library, and I wanted one where the algorithm *is* the project."

---

## 🎤 The 2-minute walkthrough

Use this structure: **what it is → why → how it works → the hard part → the result.**

1. **What it is (15s).** "It's a single-page web app — a neural-network sandbox. On the left you configure data and architecture, the center shows a live decision boundary, and the right shows the network's weights and the loss/accuracy curves as it trains."

2. **Why I built it (20s).** "I kept trying to explain backpropagation with diagrams and watching it not land. Almost every online demo is a thin UI over TensorFlow doing the real work — you learn nothing about the learning. So I inverted it: what if the library *was* the demo? You can open the source and the gradients are right there."

3. **How it works (40s).** "The engine in `nn.js` represents each layer as `a = act(W·a + b)`. Forward pass caches the pre-activations. Backprop uses the fact that for a sigmoid output with cross-entropy loss, the output error collapses to just `(prediction − label)` — then deltas propagate backward layer by layer. Gradients are averaged over a mini-batch, L2 is added, and the optimizer applies the update. I wrote Adam with proper bias correction. Everything's seeded with a deterministic RNG so runs reproduce."

4. **The hard part (30s).** "The first run had no errors but the boundary just wouldn't move — silently-wrong gradients, the worst kind of bug. I added a numerical gradient check — perturb each weight by epsilon, measure the actual loss change, compare to the analytic gradient. It pinned the bug to a batch-averaging line. Now the two agree to about 2e-10."

5. **The result (15s).** "It trains XOR, circles, moons, and the two-spiral problem to ~100% train accuracy, runs at 60fps with 6,400 forward passes a frame for the heatmap, and ships as a static site with zero dependencies and no build step."

---

## ⭐ STAR stories

### STAR 1 — The gradient bug (debugging under ambiguity)
- **Situation.** First time I wired the engine to the UI, training ran cleanly but the model never learned — flat loss, motionless boundary, no exception to chase.
- **Task.** Find a correctness bug with no stack trace and no obvious symptom beyond "it doesn't work."
- **Action.** Instead of guessing, I treated the gradients as a hypothesis to test. I implemented a numerical gradient check: nudge each parameter by ±ε, measure the real change in loss (central differences), and compare to what backprop produced via relative error. The numbers diverged in exactly one place — the mini-batch gradient averaging.
- **Result.** Fixed in one line. Analytic and numerical gradients now match to ~2e-10, and the check became a permanent regression guard. The lesson I tell people: when the clever code is silently wrong, stop reading it and measure it against ground truth.

### STAR 2 — Real-time rendering on a budget (performance)
- **Situation.** The decision-boundary heatmap needs a forward pass per grid cell — 80×80 = 6,400 per frame — and the first version stuttered badly.
- **Task.** Keep the visualization smooth (60fps) while the network trains continuously.
- **Action.** Two changes: render the field into an 80×80 offscreen canvas and let the GPU upscale it with smoothing (instead of computing at full resolution), and gate recomputation behind a dirty flag so the expensive field only rebuilds when weights actually change — idle frames just redraw the cache.
- **Result.** Steady 60fps even at high training speed. It reinforced a habit from my jet-engine control days: budget your milliseconds before you spend them.

### STAR 3 — The zero-dependency constraint (judgment / scoping)
- **Situation.** I could have reached for a chart library, a framework, a math library.
- **Task.** Decide what this project should optimize for.
- **Action.** I made readability the top priority and banned all dependencies — which meant hand-writing the line charts, the seeded RNG, the weight init, and the rendering. I kept the data structures close to the math (nested arrays mirroring matrix notation) even where typed arrays would be faster.
- **Result.** A repo a stranger can read in an afternoon, that loads instantly and has no supply chain. The constraint made the project *clearer*, which was the entire point — it's a teaching tool first.

---

## 💬 Likely technical Q&A

**Q: Walk me through backpropagation as you implemented it.**
A: Forward pass caches `z` and `a` per layer. The output is a sigmoid with BCE loss, so the output-layer error is `δ = p − y`. For each earlier layer, `δ_l = (Wᵀ_{l+1} · δ_{l+1}) ⊙ act'(z_l)`. The weight gradient for a layer is the outer product `δ · aᵀ_prev`, the bias gradient is just `δ`. I accumulate per-example, average over the batch, add L2, then apply the optimizer.

**Q: Why does the output error simplify to `(p − y)`?**
A: Because the derivative of binary cross-entropy with respect to the sigmoid's pre-activation cancels the sigmoid derivative term — `dL/dp · dp/dz` works out to `p − y`. It's the standard, clean result, and it means I never have to compute the sigmoid derivative on the output layer.

**Q: How did you verify the gradients are correct?**
A: Numerical gradient checking — central differences. For each weight, `g_num ≈ (L(w+ε) − L(w−ε)) / 2ε`, compared to the analytic gradient via relative error. They agree to ~2e-10, which is around floating-point precision for this.

**Q: How is Adam different from plain SGD here, and what's the bias correction for?**
A: SGD steps directly along the gradient. Adam keeps running estimates of the first moment (mean) and second moment (uncentered variance) of the gradients and scales the step by `m̂ / (√v̂ + ε)`. The bias correction divides by `1 − βᵗ` because the moment estimates start at zero and are biased toward zero early on — without it the first steps would be far too small.

**Q: Why no TensorFlow.js? Isn't that the obvious tool?**
A: For a production model, sure. But the entire value proposition here is that you can *see and read* the algorithm. A library would hide the one thing I'm trying to show. It's a deliberate inversion of the usual demo.

**Q: How do you keep the visualization in sync with the actual model?**
A: Everything reads from the same `Network` object each frame. Crucially, the boundary renderer runs real forward passes on coordinates put through the *same* `featurize()` the training data uses — so the picture can't drift from reality.

**Q: What are the performance characteristics / limits?**
A: It's `O(params)` per forward pass, run on the main thread inside a requestAnimationFrame loop. That's fine for these small 2-D problems. It would not scale to large nets — training would block the UI thread — which is why "move training to a Web Worker" is top of my next list.

**Q: Why nested arrays instead of typed arrays / a flat buffer?**
A: Readability. `W[l][j][k]` reads like the math. At this scale the speed difference is irrelevant. For a serious version I'd switch to `Float32Array` and flatten the layout for cache locality — I know the tradeoff, I just chose clarity here.

**Q: How does feature engineering let a network with no hidden layers solve a circle?**
A: A circle isn't linearly separable in `(x, y)`, but it is in `(x², y²)` — the boundary becomes a line in that transformed space. Adding the squared features turns the problem into plain logistic regression, which is a nice one-click lesson the playground makes visible.

**Q: How do you get reproducible runs?**
A: A `mulberry32` seeded PRNG drives both data generation and weight initialization (He for ReLU, Xavier otherwise). Same seed, same run. Reset bumps the seed for a fresh-but-deterministic run.

---

## 🔭 What I'd improve next (growth & self-awareness)

- **Web Worker training.** Move the training loop off the main thread so big configs don't jank the UI — the current main-thread loop is the clearest limitation.
- **Typed arrays + flat layout.** Swap nested arrays for `Float32Array` for cache-friendly math; measure the speedup.
- **Multi-class.** Generalize the output to softmax + categorical cross-entropy so it's not binary-only.
- **Training replay.** Snapshot weights periodically and add a scrubber to replay how the boundary evolved — great for teaching.
- **Persistence + sharing.** Save the last config to `localStorage` and encode setups in the URL so a specific experiment is shareable.
- **Automated test in CI.** Wire the existing gradient check and the "trains XOR to 100%" smoke test into a GitHub Action.

## 🙋 Smart questions to ask the interviewer

- "When something is silently wrong in production here — not throwing, just incorrect — how does the team usually catch it? What does your equivalent of a gradient check look like?"
- "How do you balance shipping fast against the kind of from-scratch understanding that makes debugging possible later?"
- "Where does this team sit on the build-vs-buy line for core infrastructure?"
- "What does the first 90 days look like for someone in this role, and what would 'clearly succeeding' look like at the end of it?"
- "What's a technical decision the team made recently that you'd revisit?"

---

> **Reminder before the call:** the strongest part of this project is the debugging story and the *why*. Lead with the inversion (the library is the demo), and don't be afraid to say "I chose clarity over speed here, and here's the tradeoff I accepted." Interviewers trust people who name their tradeoffs.
