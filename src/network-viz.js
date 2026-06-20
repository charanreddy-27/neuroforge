/*
 * NeuroForge — live network diagram
 * ---------------------------------
 * Draws the architecture and, more importantly, the *learned weights*:
 * every connection's colour encodes sign (amber +, cyan −) and its
 * thickness/opacity encodes magnitude. As the optimizer works, the graph
 * visibly reorganises itself — dead connections fade, important ones thicken.
 */
(function (root, factory) {
  const mod = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = mod;
  root.NF = root.NF || {};
  root.NF.NetworkViz = mod;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  class NetworkViz {
    constructor(canvas) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this._resize();
      window.addEventListener("resize", () => this._resize());
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

    _layout(net, inputLabels) {
      const counts = [net.inputDim, ...net.specs.map((s) => s.units)];
      const padX = 54, padY = 26;
      const usableW = this.w - padX * 2;
      const cols = counts.length;
      const columns = counts.map((count, c) => {
        const x = cols === 1 ? this.w / 2 : padX + (usableW * c) / (cols - 1);
        const gap = (this.h - padY * 2) / Math.max(1, count);
        const nodes = [];
        for (let i = 0; i < count; i++) {
          nodes.push({ x, y: padY + gap * (i + 0.5) });
        }
        return { x, nodes, count };
      });
      return columns;
    }

    draw(net, inputLabels) {
      const ctx = this.ctx;
      ctx.clearRect(0, 0, this.w, this.h);
      const columns = this._layout(net, inputLabels);

      // global max |weight| for normalisation
      let maxAbs = 1e-6;
      for (const Wl of net.W) for (const row of Wl) for (const w of row) {
        if (Math.abs(w) > maxAbs) maxAbs = Math.abs(w);
      }

      const nodeR = Math.max(5, Math.min(11, 150 / Math.max(...columns.map((c) => c.count))));

      // edges
      for (let l = 0; l < net.W.length; l++) {
        const Wl = net.W[l];
        const prev = columns[l].nodes;
        const cur = columns[l + 1].nodes;
        for (let j = 0; j < Wl.length; j++) {
          for (let k = 0; k < Wl[j].length; k++) {
            const w = Wl[j][k];
            const norm = Math.abs(w) / maxAbs;
            if (norm < 0.04) continue; // skip near-dead links for clarity
            ctx.beginPath();
            ctx.moveTo(prev[k].x, prev[k].y);
            ctx.lineTo(cur[j].x, cur[j].y);
            ctx.strokeStyle = w >= 0
              ? `rgba(251,146,60,${0.06 + norm * 0.6})`
              : `rgba(56,189,248,${0.06 + norm * 0.6})`;
            ctx.lineWidth = 0.4 + norm * 3.4;
            ctx.stroke();
          }
        }
      }

      // nodes
      for (let c = 0; c < columns.length; c++) {
        const col = columns[c];
        for (let i = 0; i < col.nodes.length; i++) {
          const n = col.nodes[i];
          ctx.beginPath();
          ctx.arc(n.x, n.y, nodeR, 0, Math.PI * 2);
          ctx.fillStyle = "#0b1326";
          ctx.fill();
          const isOut = c === columns.length - 1;
          ctx.lineWidth = isOut ? 2.4 : 1.6;
          ctx.strokeStyle = isOut ? "#e2e8f0" : "rgba(148,163,184,0.7)";
          ctx.stroke();
        }
      }

      // labels for input + output
      ctx.fillStyle = "rgba(226,232,240,0.85)";
      ctx.font = "600 11px ui-monospace, 'JetBrains Mono', monospace";
      ctx.textBaseline = "middle";
      ctx.textAlign = "right";
      const inCol = columns[0];
      for (let i = 0; i < inCol.nodes.length; i++) {
        const label = (inputLabels && inputLabels[i]) || ("x" + i);
        ctx.fillText(label, inCol.nodes[i].x - nodeR - 8, inCol.nodes[i].y);
      }
      ctx.textAlign = "left";
      const outCol = columns[columns.length - 1];
      ctx.fillText("ŷ", outCol.nodes[0].x + nodeR + 8, outCol.nodes[0].y);

      // column captions
      ctx.fillStyle = "rgba(148,163,184,0.55)";
      ctx.font = "500 10px ui-monospace, monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (let c = 0; c < columns.length; c++) {
        const cap = c === 0 ? "input" : c === columns.length - 1 ? "output" : `h${c}`;
        ctx.fillText(cap, columns[c].x, 6);
      }
    }
  }

  return NetworkViz;
});
