/*
 * NeuroForge — datasets & feature engineering
 * -------------------------------------------
 * Synthetic 2-D binary classification problems (the classic "can a net
 * learn this shape?" battery) plus the input-feature transforms that let
 * a small network reach beyond straight lines.
 *
 * Raw coordinates live in [-DOMAIN, DOMAIN]^2 (used for plotting). Features
 * are normalised to roughly [-1, 1] for stable training. The decision-boundary
 * renderer feeds raw coordinates through the exact same `featurize()` so what
 * you see is what the network sees.
 */
(function (root, factory) {
  const mod = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = mod;
  root.NF = root.NF || {};
  root.NF.data = mod;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const DOMAIN = 5;

  function rngFrom(seed) {
    let a = (seed >>> 0) || 1;
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(rng) {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // ---- engineered input features ----------------------------------------
  // Each maps raw (x1, x2) in [-DOMAIN, DOMAIN] to a value in ~[-1, 1].
  const FEATURES = [
    { id: "x1",   label: "X1",     fn: (x, y) => x / DOMAIN },
    { id: "x2",   label: "X2",     fn: (x, y) => y / DOMAIN },
    { id: "x1sq", label: "X1^2",   fn: (x, y) => (x / DOMAIN) ** 2 },
    { id: "x2sq", label: "X2^2",   fn: (x, y) => (y / DOMAIN) ** 2 },
    { id: "x1x2", label: "X1X2",   fn: (x, y) => (x / DOMAIN) * (y / DOMAIN) },
    { id: "sin1", label: "sin X1", fn: (x, y) => Math.sin(x) },
    { id: "sin2", label: "sin X2", fn: (x, y) => Math.sin(y) },
  ];
  const FEATURE_BY_ID = Object.fromEntries(FEATURES.map((f) => [f.id, f]));

  function featurize(x, y, enabledIds) {
    const out = new Array(enabledIds.length);
    for (let i = 0; i < enabledIds.length; i++) {
      out[i] = FEATURE_BY_ID[enabledIds[i]].fn(x, y);
    }
    return out;
  }

  // ---- dataset generators ------------------------------------------------
  // Each returns an array of { x, y, label } with label in {0, 1}.
  function circle(n, noise, rng) {
    const pts = [];
    const r = DOMAIN;
    for (let i = 0; i < n; i++) {
      const inner = i % 2 === 0;
      const radius = inner ? rng() * (r * 0.45) : r * 0.62 + rng() * (r * 0.38);
      const ang = rng() * 2 * Math.PI;
      const nx = (rng() - 0.5) * noise * r * 0.4;
      const ny = (rng() - 0.5) * noise * r * 0.4;
      pts.push({ x: radius * Math.cos(ang) + nx, y: radius * Math.sin(ang) + ny, label: inner ? 1 : 0 });
    }
    return pts;
  }

  function xor(n, noise, rng) {
    const pts = [];
    for (let i = 0; i < n; i++) {
      let x = (rng() - 0.5) * 2 * DOMAIN;
      let y = (rng() - 0.5) * 2 * DOMAIN;
      x += x > 0 ? 0.3 : -0.3;
      y += y > 0 ? 0.3 : -0.3;
      const nx = (rng() - 0.5) * noise * DOMAIN * 0.5;
      const ny = (rng() - 0.5) * noise * DOMAIN * 0.5;
      const label = (x * y >= 0) ? 1 : 0;
      pts.push({ x: x + nx, y: y + ny, label });
    }
    return pts;
  }

  function gaussians(n, noise, rng) {
    const pts = [];
    const spread = 1 + noise * 2.2;
    for (let i = 0; i < n; i++) {
      const a = i % 2 === 0;
      const cx = a ? 2.4 : -2.4;
      const cy = a ? 2.4 : -2.4;
      pts.push({ x: cx + gauss(rng) * spread, y: cy + gauss(rng) * spread, label: a ? 1 : 0 });
    }
    return pts;
  }

  function spiral(n, noise, rng) {
    const pts = [];
    const half = Math.floor(n / 2);
    for (let s = 0; s < 2; s++) {
      const count = s === 0 ? half : n - half;
      for (let i = 0; i < count; i++) {
        const t = (i / count) * 3.5;
        const radius = (t / 3.5) * DOMAIN;
        const ang = t * Math.PI + s * Math.PI;
        const nx = (rng() - 0.5) * noise * 1.6;
        const ny = (rng() - 0.5) * noise * 1.6;
        pts.push({ x: radius * Math.cos(ang) + nx, y: radius * Math.sin(ang) + ny, label: s });
      }
    }
    return pts;
  }

  function moons(n, noise, rng) {
    // Classic two interleaving half-moons (scikit-learn style), scaled + centred.
    const pts = [];
    const half = Math.floor(n / 2);
    const r = 3.6;
    for (let i = 0; i < n; i++) {
      const top = i < half;
      const t = rng() * Math.PI;
      let x, y;
      if (top) { x = r * Math.cos(t); y = r * Math.sin(t); }
      else { x = r * (1 - Math.cos(t)); y = r * (0.5 - Math.sin(t)); }
      x -= r * 0.5; y -= r * 0.25;
      const nx = gauss(rng) * noise * 1.2;
      const ny = gauss(rng) * noise * 1.2;
      pts.push({ x: x + nx, y: y + ny, label: top ? 1 : 0 });
    }
    return pts;
  }

  const GENERATORS = {
    circle: { fn: circle, label: "Circle" },
    xor: { fn: xor, label: "XOR" },
    gauss: { fn: gaussians, label: "Gaussian" },
    spiral: { fn: spiral, label: "Spiral" },
    moons: { fn: moons, label: "Moons" },
  };

  function generate(kind, opts) {
    const o = opts || {};
    const n = o.n != null ? o.n : 320;
    const noise = o.noise != null ? o.noise : 0.2;
    const split = o.split != null ? o.split : 0.5;
    const seed = o.seed != null ? o.seed : 1;
    const rng = rngFrom(seed);
    const gen = (GENERATORS[kind] || GENERATORS.circle).fn;
    const all = gen(n, noise, rng);
    for (let i = all.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = all[i]; all[i] = all[j]; all[j] = tmp;
    }
    const cut = Math.floor(all.length * split);
    return { train: all.slice(0, cut), test: all.slice(cut), all };
  }

  return { DOMAIN, FEATURES, FEATURE_BY_ID, featurize, generate, GENERATORS };
});
