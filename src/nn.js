/*
 * NeuroForge — neural network engine (from scratch, zero dependencies)
 * --------------------------------------------------------------------
 * A small but real feed-forward network with manual backpropagation.
 * Everything below — the forward pass, the gradients, the optimizers —
 * is implemented by hand. No TensorFlow, no autograd. The math is the
 * point.
 *
 * Conventions
 *   - A "layer" is fully-connected: a = act(W · a_prev + b)
 *   - W[l] is shaped [units, inputDim], b[l] is [units]
 *   - The output layer is a single sigmoid unit -> binary classifier
 *   - Loss is binary cross-entropy with L2 weight decay
 */
(function (root, factory) {
  const mod = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = mod;
  root.NF = root.NF || {};
  root.NF.nn = mod;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // ---- deterministic RNG so a given seed reproduces a run ---------------
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0;
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  // standard normal via Box–Muller
  function gaussian(rng) {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // ---- activations: value + derivative ----------------------------------
  // Derivatives are expressed in terms of the most convenient cached value:
  // for tanh/sigmoid that is the activation `a`, for relu the pre-activation `z`.
  const ACT = {
    tanh: {
      f: (z) => Math.tanh(z),
      // d/dz tanh = 1 - tanh^2 = 1 - a^2
      df: (z, a) => 1 - a * a,
    },
    relu: {
      f: (z) => (z > 0 ? z : 0),
      df: (z) => (z > 0 ? 1 : 0),
    },
    sigmoid: {
      f: (z) => 1 / (1 + Math.exp(-z)),
      // d/dz sigmoid = a(1 - a)
      df: (z, a) => a * (1 - a),
    },
    linear: {
      f: (z) => z,
      df: () => 1,
    },
  };

  function he(inDim, rng) {
    return gaussian(rng) * Math.sqrt(2 / inDim);
  }
  function xavier(inDim, outDim, rng) {
    return gaussian(rng) * Math.sqrt(2 / (inDim + outDim));
  }

  class Network {
    /**
     * @param {number} inputDim         number of input features
     * @param {Array<{units:number,activation:string}>} layerSpecs
     *        hidden layers + output layer (last entry should be the output)
     * @param {object} opts { optimizer, learningRate, l2, seed }
     */
    constructor(inputDim, layerSpecs, opts = {}) {
      this.inputDim = inputDim;
      this.specs = layerSpecs.map((s) => ({ ...s }));
      this.optimizer = opts.optimizer || "adam";
      this.lr = opts.learningRate != null ? opts.learningRate : 0.03;
      this.l2 = opts.l2 != null ? opts.l2 : 0;
      this.beta1 = 0.9;
      this.beta2 = 0.999;
      this.eps = 1e-8;
      this.t = 0; // Adam timestep
      const rng = mulberry32(opts.seed != null ? opts.seed : 1337);

      this.W = [];
      this.b = [];
      // optimizer accumulators
      this.mW = []; this.vW = []; this.mb = []; this.vb = [];

      let prev = inputDim;
      for (const spec of this.specs) {
        const units = spec.units;
        const act = spec.activation;
        const W = [];
        const mW = [], vW = [];
        for (let j = 0; j < units; j++) {
          const row = new Array(prev);
          const mrow = new Array(prev).fill(0);
          const vrow = new Array(prev).fill(0);
          for (let k = 0; k < prev; k++) {
            row[k] = act === "relu" ? he(prev, rng) : xavier(prev, units, rng);
          }
          W.push(row); mW.push(mrow); vW.push(vrow);
        }
        this.W.push(W);
        this.b.push(new Array(units).fill(0));
        this.mW.push(mW); this.vW.push(vW);
        this.mb.push(new Array(units).fill(0));
        this.vb.push(new Array(units).fill(0));
        prev = units;
      }
    }

    get numLayers() { return this.specs.length; }

    paramCount() {
      let n = 0;
      for (let l = 0; l < this.W.length; l++) {
        n += this.W[l].length * this.W[l][0].length + this.b[l].length;
      }
      return n;
    }

    /**
     * Forward pass for one example. Returns the output probability and,
     * optionally, the full cache of pre-activations/activations needed for
     * backprop and for the activation visualizer.
     */
    forward(x, keepCache) {
      let a = x;
      const cache = keepCache ? { inputs: [], z: [], a: [] } : null;
      for (let l = 0; l < this.specs.length; l++) {
        const act = ACT[this.specs[l].activation];
        const W = this.W[l], b = this.b[l];
        const units = W.length;
        const zL = new Array(units);
        const aL = new Array(units);
        for (let j = 0; j < units; j++) {
          let s = b[j];
          const wj = W[j];
          for (let k = 0; k < a.length; k++) s += wj[k] * a[k];
          zL[j] = s;
          aL[j] = act.f(s);
        }
        if (cache) { cache.inputs.push(a); cache.z.push(zL); cache.a.push(aL); }
        a = aL;
      }
      const output = a[0];
      return cache ? { output, cache } : { output };
    }

    predict(x) { return this.forward(x).output; }

    /** Accumulate gradients for a single example into gW/gb (in place). */
    _backprop(x, y, gW, gb) {
      const { output, cache } = this.forward(x, true);
      const L = this.specs.length;

      // output layer: sigmoid + BCE => dL/dz = (p - y)
      let delta = [output - y];

      for (let l = L - 1; l >= 0; l--) {
        const aPrev = cache.inputs[l];
        const gWl = gW[l], gbl = gb[l];
        const Wl = this.W[l];
        const units = Wl.length;

        for (let j = 0; j < units; j++) {
          const dj = delta[j];
          gbl[j] += dj;
          const gwj = gWl[j];
          for (let k = 0; k < aPrev.length; k++) gwj[k] += dj * aPrev[k];
        }

        if (l > 0) {
          // propagate delta to previous layer
          const prevSpec = this.specs[l - 1];
          const act = ACT[prevSpec.activation];
          const zPrev = cache.z[l - 1];
          const aPrevOut = cache.a[l - 1];
          const newDelta = new Array(aPrev.length).fill(0);
          for (let j = 0; j < units; j++) {
            const dj = delta[j];
            const wj = Wl[j];
            for (let k = 0; k < aPrev.length; k++) newDelta[k] += wj[k] * dj;
          }
          for (let k = 0; k < newDelta.length; k++) {
            newDelta[k] *= act.df(zPrev[k], aPrevOut[k]);
          }
          delta = newDelta;
        }
      }
      return output;
    }

    /** One optimizer step over a mini-batch. Returns mean batch loss. */
    trainBatch(X, Y) {
      const L = this.specs.length;
      const gW = this.W.map((Wl) => Wl.map((row) => new Array(row.length).fill(0)));
      const gb = this.b.map((bl) => new Array(bl.length).fill(0));

      let loss = 0;
      const n = X.length;
      for (let i = 0; i < n; i++) {
        const p = this._backprop(X[i], Y[i], gW, gb);
        const pc = Math.min(1 - 1e-7, Math.max(1e-7, p));
        loss += -(Y[i] * Math.log(pc) + (1 - Y[i]) * Math.log(1 - pc));
      }

      // average gradients, add L2, then apply the chosen optimizer
      this.t += 1;
      const inv = 1 / n;
      for (let l = 0; l < L; l++) {
        const Wl = this.W[l], bl = this.b[l];
        for (let j = 0; j < Wl.length; j++) {
          const wj = Wl[j], gwj = gW[l][j];
          for (let k = 0; k < wj.length; k++) {
            let g = gwj[k] * inv + this.l2 * wj[k];
            wj[k] -= this._step(g, this.mW[l][j], this.vW[l][j], k);
          }
          let gbj = gb[l][j] * inv;
          bl[j] -= this._stepScalar(gbj, this.mb[l], this.vb[l], j);
        }
      }
      return loss * inv;
    }

    // optimizer update for a weight entry stored in arrays m/v at index k
    _step(g, m, v, k) {
      if (this.optimizer === "sgd") return this.lr * g;
      if (this.optimizer === "momentum") {
        m[k] = this.beta1 * m[k] - this.lr * g;
        return -m[k];
      }
      // adam
      m[k] = this.beta1 * m[k] + (1 - this.beta1) * g;
      v[k] = this.beta2 * v[k] + (1 - this.beta2) * g * g;
      const mHat = m[k] / (1 - Math.pow(this.beta1, this.t));
      const vHat = v[k] / (1 - Math.pow(this.beta2, this.t));
      return (this.lr * mHat) / (Math.sqrt(vHat) + this.eps);
    }
    _stepScalar(g, m, v, k) { return this._step(g, m, v, k); }

    /** Evaluate mean loss + accuracy over a dataset. */
    evaluate(X, Y) {
      let loss = 0, correct = 0;
      for (let i = 0; i < X.length; i++) {
        const p = this.forward(X[i]).output;
        const pc = Math.min(1 - 1e-7, Math.max(1e-7, p));
        loss += -(Y[i] * Math.log(pc) + (1 - Y[i]) * Math.log(1 - pc));
        if ((p >= 0.5 ? 1 : 0) === Y[i]) correct++;
      }
      return { loss: loss / X.length, acc: correct / X.length };
    }

    /** Serialize weights so a trained model can be exported/reloaded. */
    toJSON() {
      return {
        inputDim: this.inputDim,
        specs: this.specs,
        optimizer: this.optimizer,
        learningRate: this.lr,
        l2: this.l2,
        W: this.W,
        b: this.b,
      };
    }
  }

  return { Network, ACT, mulberry32, gaussian };
});
