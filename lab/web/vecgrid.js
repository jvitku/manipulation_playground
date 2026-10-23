/* Vector grids: a vector as a small grid of cells, diverging palette
 * (--vec-neg red = -vmax, --vec-zero near-black = 0, --vec-pos green = +vmax).
 *   LabVecGrid.render(container, values, opts) -> canvas
 *     values: flat array, or array of rows (rows = grid rows, cols = row length)
 *     opts:   {cols, labels[], title, vmax, cell}   vmax default = max|v| (>= 1e-6)
 *     hover shows "[index] value" (or "[row,col]" for array-of-rows).
 *   the returned wrapper has .update(values) to redraw in place (keeps hover listeners; use it per animation frame).
 *   LabVecGrid.panel(ctx) -> {el, add(title, values, opts), clear(), enabled} or null when the task's
 *     toggle is off (viewers: `const p = LabVecGrid.panel(ctx); if (p) {...}`). Append p.el yourself.
 *   LabVecGrid.enabled(taskId) / setEnabled(taskId, bool): per-task toggle, localStorage
 *     "lab.vecgrid.<taskId>", default ON. */
(function () {
  const KEY = (t) => `lab.vecgrid.${t}`;
  function enabled(taskId) {
    try { return localStorage.getItem(KEY(taskId)) !== "0"; } catch (e) { return true; }
  }
  function setEnabled(taskId, on) {
    try { localStorage.setItem(KEY(taskId), on ? "1" : "0"); } catch (e) { /* ignore */ }
  }
  function tok(name, fb) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fb;
  }
  let conv = null, palCache = null, palAt = 0;
  function rgb(c) {
    const d = conv || (conv = document.createElement("canvas").getContext("2d", { willReadFrequently: true }));
    d.fillStyle = "#000"; d.fillStyle = c; d.fillRect(0, 0, 1, 1);
    const p = d.getImageData(0, 0, 1, 1).data;
    return [p[0], p[1], p[2]];
  }
  function palette() { // cached ~1 s: theme changes show up on the next repaint after that
    const now = Date.now();
    if (palCache && now - palAt < 1000) return palCache;
    palAt = now;
    return (palCache = paletteNow());
  }
  function paletteNow() {
    return [rgb(tok("--vec-neg", "#d33")), rgb(tok("--vec-zero", "#101114")), rgb(tok("--vec-pos", "#2a4"))];
  }
  function color(pal, v, vmax) {
    const t = Math.max(-1, Math.min(1, v / vmax)), end = pal[t < 0 ? 0 : 2], z = pal[1], a = Math.abs(t);
    return `rgb(${z.map((c, i) => Math.round(c + (end[i] - c) * a)).join(",")})`;
  }

  function render(container, values, opts) {
    opts = opts || {};
    const wrap = document.createElement("div");
    wrap.className = "vecgrid";
    if (opts.title) { const t = document.createElement("div"); t.className = "vg-title"; t.textContent = opts.title; wrap.append(t); }
    const cv = document.createElement("canvas");
    const tip = document.createElement("div");
    tip.className = "vg-tip";
    wrap.append(cv, tip);
    const g = cv.getContext("2d");
    let flat = [], cols = 1, rows2d = false, cell = 10, vmax = 1;
    const idle = () => `±${vmax.toPrecision(3)} full scale`;

    // (re)draw the cells in place; the canvas and its listeners stay mounted
    function update(vals, vm) {
      rows2d = Array.isArray(vals) && vals.length > 0 && Array.isArray(vals[0]);
      flat = rows2d ? vals.flat() : Array.from(vals || []);
      cols = rows2d ? vals[0].length : Math.max(1, opts.cols || Math.ceil(Math.sqrt(flat.length)));
      const nrows = Math.ceil(flat.length / cols);
      cell = opts.cell || Math.max(10, Math.min(22, Math.floor(260 / cols)));
      vmax = vm != null ? vm : opts.vmax;
      if (!(Number.isFinite(vmax) && vmax > 0)) {
        vmax = Math.max(1e-6, ...flat.filter((v) => Number.isFinite(v)).map((v) => Math.abs(v)));
      }
      if (cv.width !== cols * cell || cv.height !== nrows * cell) { cv.width = cols * cell; cv.height = nrows * cell; }
      else g.clearRect(0, 0, cv.width, cv.height);
      const pal = palette(), na = tok("--line", "#888");
      flat.forEach((v, i) => {
        g.fillStyle = Number.isFinite(v) ? color(pal, v, vmax) : na;
        g.fillRect((i % cols) * cell, Math.floor(i / cols) * cell, cell - 1, cell - 1);
      });
      if (!tip.dataset.hover) tip.textContent = idle();
    }
    cv.addEventListener("mousemove", (ev) => {
      const r = cv.getBoundingClientRect(), sx = cv.width / r.width;
      const c = Math.floor(((ev.clientX - r.left) * sx) / cell), rw = Math.floor(((ev.clientY - r.top) * sx) / cell), i = rw * cols + c;
      if (c >= 0 && c < cols && i < flat.length) {
        const lab = opts.labels && opts.labels[i] != null ? opts.labels[i] : (rows2d ? `[${rw},${c}]` : `[${i}]`);
        tip.dataset.hover = "1";
        tip.textContent = Number.isFinite(flat[i]) ? `${lab} = ${Number(flat[i]).toFixed(4)}` : `${lab} = n/a`;
      }
    });
    cv.addEventListener("mouseleave", () => { delete tip.dataset.hover; tip.textContent = idle(); });
    update(values);
    if (container) container.append(wrap);
    wrap.canvas = cv;
    wrap.update = update;
    return wrap;
  }

  function panel(ctx) {
    if (!ctx || ctx.vecgrids === false) return null;
    const el = document.createElement("div");
    el.className = "vecgrid-panel";
    return {
      el, enabled: true,
      add(title, values, opts) { return render(el, values, Object.assign({ title }, opts || {})); },
      clear() { el.replaceChildren(); },
    };
  }

  window.LabVecGrid = { render, panel, enabled, setEnabled };
})();
