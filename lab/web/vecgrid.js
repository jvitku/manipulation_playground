/* Vector grids: a vector as a small grid of cells, diverging palette
 * (--vec-neg red = -vmax, --vec-zero near-black = 0, --vec-pos green = +vmax).
 *   LabVecGrid.render(container, values, opts) -> canvas
 *     values: flat array, or array of rows (rows = grid rows, cols = row length)
 *     opts:   {cols, labels[], title, vmax, cell}   vmax default = max|v| (>= 1e-6)
 *     hover shows "[index] value" (or "[row,col]" for array-of-rows).
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
  function rgb(c) {
    const d = document.createElement("canvas").getContext("2d");
    d.fillStyle = "#000"; d.fillStyle = c; d.fillRect(0, 0, 1, 1);
    const p = d.getImageData(0, 0, 1, 1).data;
    return [p[0], p[1], p[2]];
  }
  function palette() {
    return [rgb(tok("--vec-neg", "#d33")), rgb(tok("--vec-zero", "#101114")), rgb(tok("--vec-pos", "#2a4"))];
  }
  function color(pal, v, vmax) {
    const t = Math.max(-1, Math.min(1, v / vmax)), end = pal[t < 0 ? 0 : 2], z = pal[1], a = Math.abs(t);
    return `rgb(${z.map((c, i) => Math.round(c + (end[i] - c) * a)).join(",")})`;
  }

  function render(container, values, opts) {
    opts = opts || {};
    const rows2d = Array.isArray(values) && values.length && Array.isArray(values[0]);
    const flat = rows2d ? values.flat() : Array.from(values || []);
    const cols = rows2d ? values[0].length : Math.max(1, opts.cols || Math.ceil(Math.sqrt(flat.length)));
    const nrows = Math.ceil(flat.length / cols);
    const cell = opts.cell || Math.max(10, Math.min(22, Math.floor(260 / cols)));
    let vmax = opts.vmax;
    if (!(vmax > 0)) vmax = Math.max(1e-6, ...flat.map((v) => Math.abs(v)).filter(Number.isFinite));
    const wrap = document.createElement("div");
    wrap.className = "vecgrid";
    if (opts.title) { const t = document.createElement("div"); t.className = "vg-title"; t.textContent = opts.title; wrap.append(t); }
    const cv = document.createElement("canvas");
    cv.width = cols * cell; cv.height = nrows * cell;
    const g = cv.getContext("2d"), pal = palette();
    flat.forEach((v, i) => {
      g.fillStyle = Number.isFinite(v) ? color(pal, v, vmax) : tok("--line", "#888");
      g.fillRect((i % cols) * cell, Math.floor(i / cols) * cell, cell - 1, cell - 1);
    });
    const tip = document.createElement("div");
    tip.className = "vg-tip";
    tip.textContent = `±${vmax.toPrecision(3)} full scale`;
    cv.addEventListener("mousemove", (ev) => {
      const r = cv.getBoundingClientRect(), sx = cv.width / r.width;
      const c = Math.floor(((ev.clientX - r.left) * sx) / cell), rw = Math.floor(((ev.clientY - r.top) * sx) / cell), i = rw * cols + c;
      if (c >= 0 && c < cols && i < flat.length) {
        const lab = opts.labels && opts.labels[i] != null ? opts.labels[i] : (rows2d ? `[${rw},${c}]` : `[${i}]`);
        tip.textContent = `${lab} = ${Number(flat[i]).toFixed(4)}`;
      }
    });
    cv.addEventListener("mouseleave", () => { tip.textContent = `±${vmax.toPrecision(3)} full scale`; });
    wrap.append(cv, tip);
    if (container) container.append(wrap);
    wrap.canvas = cv;
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
