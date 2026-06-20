/*
 * NeuroForge — decision-boundary renderer
 * ---------------------------------------
 * Paints what the network "believes" across the whole input plane: a low-res
 * grid of forward passes, upscaled with smoothing into a soft diverging field,
 * with the training/test points scattered on top. This is the centrepiece —
 * watching the field bend itself around a spiral is the whole show.
 */
(function (root, factory) {
  const mod = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = mod;
  root.NF = root.NF || {};
  root.NF.Boundary = mod;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // diverging palette: confident class-0 (cyan) <- dark midline -> class-1 (amber)
  const C0 = [56, 189, 248];   // cyan
  const C1 = [251, 146, 60];   // amber
  const MID = [12, 18, 32];    // near-black slate

  function colorFor(p) {
    const d = p - 0.5;
    const intensity = Math.min(1, Math.abs(d) * 2);
    const end = d < 0 ? C0 : C1;
    const r = MID[0] + (end[0] - MID[0]) * intensity;
    const g = MID[1] + (end[1] - MID[1]) * intensity;
    const b = MID[2] + (end[2] - MID[2]) * intensity;
    return [r, g, b];
  }

  class Boundary {
    constructor(canvas, domain) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.domain = domain;
      this.res = 80; // grid cells per axis
      this.grid = document.createElement("canvas");
      this.grid.width = this.res;
      this.grid.height = this.res;
      this.gctx = this.grid.getContext("2d");
      this._img = this.gctx.createImageData(this.res, this.res);
      this._resize();
      window.addEventListener("resize", () => this._resize());
    }

    _resize() {
      const dpr = window.devicePixelRatio || 1;
      const rect = this.canvas.getBoundingClientRect();
      const size = Math.max(1, Math.min(rect.width, rect.height));
      this.cssSize = size;
      this.canvas.width = Math.round(size * dpr);
      this.canvas.height = Math.round(size * dpr);
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    _toPx(x, y) {
      const s = this.cssSize;
      const px = ((x + this.domain) / (2 * this.domain)) * s;
      const py = ((this.domain - y) / (2 * this.domain)) * s; // flip Y
      return [px, py];
    }

    /** Recompute the heatmap field (call sparingly — every few epochs). */
    computeField(net, featurize, enabled) {
      const res = this.res, data = this._img.data;
      const D = this.domain;
      for (let j = 0; j < res; j++) {
        // map grid row -> world y (top row = +D)
        const wy = D - ((j + 0.5) / res) * 2 * D;
        for (let i = 0; i < res; i++) {
          const wx = -D + ((i + 0.5) / res) * 2 * D;
          const p = net.forward(featurize(wx, wy, enabled)).output;
          const [r, g, b] = colorFor(p);
          const idx = (j * res + i) * 4;
          data[idx] = r; data[idx + 1] = g; data[idx + 2] = b; data[idx + 3] = 255;
        }
      }
      this.gctx.putImageData(this._img, 0, 0);
    }

    /** Draw the cached field + points. Cheap; call every frame. */
    draw(points) {
      const ctx = this.ctx, s = this.cssSize;
      ctx.clearRect(0, 0, s, s);

      // upscaled, smoothed heatmap
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.globalAlpha = 0.92;
      ctx.drawImage(this.grid, 0, 0, s, s);
      ctx.globalAlpha = 1;

      // faint world axes
      ctx.strokeStyle = "rgba(148,163,184,0.16)";
      ctx.lineWidth = 1;
      const [ox, oy] = this._toPx(0, 0);
      ctx.beginPath();
      ctx.moveTo(ox, 0); ctx.lineTo(ox, s);
      ctx.moveTo(0, oy); ctx.lineTo(s, oy);
      ctx.stroke();

      // data points
      if (points) {
        for (const pt of points) {
          const [px, py] = this._toPx(pt.x, pt.y);
          const base = pt.label === 1 ? "#fb923c" : "#38bdf8";
          ctx.beginPath();
          ctx.arc(px, py, 4.2, 0, Math.PI * 2);
          ctx.fillStyle = base;
          ctx.fill();
          ctx.lineWidth = 1.2;
          // test points get a bright ring so the split is visible
          ctx.strokeStyle = pt.test ? "rgba(255,255,255,0.95)" : "rgba(2,6,23,0.65)";
          ctx.stroke();
        }
      }
    }
  }

  return Boundary;
});
