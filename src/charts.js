/*
 * NeuroForge — minimal live line charts (hand-drawn on canvas, no chart lib)
 * Tracks train vs. test curves for loss and accuracy as training progresses.
 */
(function (root, factory) {
  const mod = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = mod;
  root.NF = root.NF || {};
  root.NF.LineChart = mod;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  class LineChart {
    /**
     * @param {HTMLCanvasElement} canvas
     * @param {object} opts { series:[{key,color,dash}], yMin, yMax, fmt }
     */
    constructor(canvas, opts) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.series = opts.series;
      this.yMin = opts.yMin;
      this.yMax = opts.yMax;
      this.fmt = opts.fmt || ((v) => v.toFixed(3));
      this.maxLen = opts.maxLen || 600; // sliding window so memory stays bounded
      this.data = {};
      this.series.forEach((s) => (this.data[s.key] = []));
      this._resize();
      window.addEventListener("resize", () => { this._resize(); this.draw(); });
    }

    _resize() {
      const dpr = window.devicePixelRatio || 1;
      const rect = this.canvas.getBoundingClientRect();
      this.w = Math.max(1, rect.width);
      this.h = Math.max(1, rect.height);
      this.canvas.width = Math.round(this.w * dpr);
      this.canvas.height = Math.round(this.h * dpr);
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    reset() { this.series.forEach((s) => (this.data[s.key] = [])); this.draw(); }

    push(values) {
      this.series.forEach((s) => {
        if (values[s.key] != null) {
          const arr = this.data[s.key];
          arr.push(values[s.key]);
          if (arr.length > this.maxLen) arr.shift();
        }
      });
    }

    draw() {
      const ctx = this.ctx, w = this.w, h = this.h;
      const padL = 6, padR = 6, padT = 10, padB = 6;
      ctx.clearRect(0, 0, w, h);

      let lo = this.yMin, hi = this.yMax;
      if (lo == null || hi == null) {
        lo = Infinity; hi = -Infinity;
        for (const s of this.series) for (const v of this.data[s.key]) {
          if (v < lo) lo = v; if (v > hi) hi = v;
        }
        if (!isFinite(lo)) { lo = 0; hi = 1; }
        const pad = (hi - lo) * 0.12 || 0.1;
        lo -= pad; hi += pad;
        if (this.yMin != null) lo = this.yMin;
      }
      const span = hi - lo || 1;

      // gridlines
      ctx.strokeStyle = "rgba(148,163,184,0.10)";
      ctx.lineWidth = 1;
      for (let g = 0; g <= 3; g++) {
        const y = padT + ((h - padT - padB) * g) / 3;
        ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
      }

      const maxLen = Math.max(2, ...this.series.map((s) => this.data[s.key].length));
      const xAt = (i) => padL + ((w - padL - padR) * i) / Math.max(1, maxLen - 1);
      const yAt = (v) => padT + (h - padT - padB) * (1 - (v - lo) / span);

      for (const s of this.series) {
        const arr = this.data[s.key];
        if (arr.length < 1) continue;
        ctx.beginPath();
        ctx.setLineDash(s.dash || []);
        ctx.strokeStyle = s.color;
        ctx.lineWidth = 2;
        ctx.lineJoin = "round";
        for (let i = 0; i < arr.length; i++) {
          const x = xAt(i), y = yAt(arr[i]);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    last(key) {
      const a = this.data[key];
      return a.length ? a[a.length - 1] : null;
    }
  }

  return LineChart;
});
