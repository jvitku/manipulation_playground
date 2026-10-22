/* Result viewers. Contract for real viewers (added by later tasks):
 *   LabViewers.register(name, fn)  where fn(resultObj, container, ctx) renders into container.
 *   ctx = {task, evalId, fileUrl(name) -> "/api/evals/<task>/<eval>/file/<name>"}.
 *   LabViewers.render(viewerName, resultObj, container, ctx) dispatches by spec.viewer;
 *   unknown/unregistered names fall back to pretty-printed JSON. */
(function () {
  const registry = {};

  function jsonFallback(result, container) {
    const pre = document.createElement("pre");
    pre.className = "json";
    pre.textContent = JSON.stringify(result, null, 2);
    container.replaceChildren(pre);
  }

  function render(viewer, result, container, ctx) {
    const fn = registry[viewer] || jsonFallback;
    try {
      fn(result, container, ctx || {});
    } catch (e) {
      jsonFallback(result, container);
    }
  }

  // ---- shared helpers -------------------------------------------------------------------
  const NS = "http://www.w3.org/2000/svg";
  const T1_OBSTACLE = 0.25, T1_HALF_GAP = 0.5; // toy.py: T1_OBSTACLE, T1_HALF_GAP

  function h(tag, attrs, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "text") e.textContent = v; else e.setAttribute(k, v);
    }
    for (const c of kids) if (c) e.append(c);
    return e;
  }
  function s(tag, attrs, text) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
    if (text != null) e.textContent = text;
    return e;
  }
  function css(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }
  const num = (x) => (typeof x === "number" && isFinite(x) ? x : null);
  const pct = (x) => (num(x) == null ? "n/a" : (100 * x).toFixed(1) + "%");
  function chip(label, value) {
    return h("span", { class: "chip", style: "background:var(--chip);border-radius:999px;padding:2px 10px;font:12px var(--mono);margin-right:8px;display:inline-block" },
      document.createTextNode(label + " "), h("b", { text: value }));
  }

  // ---- t1scatter: observation c vs sampled a_x ------------------------------------------
  function t1scatter(result, container) {
    const t1 = (result && result.t1) || {};
    const pts = Array.isArray(t1.points) ? t1.points.filter((p) => Array.isArray(p) && num(p[0]) != null && num(p[1]) != null) : [];
    const wrap = h("div", { class: "viewer-t1" });
    const stats = h("div", { style: "margin-bottom:6px" });
    stats.append(chip("collision", pct(t1.collision)), chip("left fraction", pct(t1.left_frac)),
      chip("mode error", num(t1.mode_err) == null ? "n/a" : t1.mode_err.toFixed(3)));
    wrap.append(stats);

    const W = 420, H = 300, L = 40, R = 10, T = 10, B = 32;
    const cx0 = -0.4, cx1 = 0.4, ay0 = -1, ay1 = 1;
    const X = (c) => L + ((c - cx0) / (cx1 - cx0)) * (W - L - R);
    const Y = (a) => T + (1 - (a - ay0) / (ay1 - ay0)) * (H - T - B);
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, style: "width:100%;max-width:520px;height:auto;font:10px var(--mono)", role: "img", "aria-label": "T1 samples: obstacle position vs sampled action" });
    svg.append(s("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "none", stroke: "var(--line)" }));
    // obstacle band |a_x - c| < 0.25, as a polygon along the diagonal a_x = c
    const band = [[cx0, cx0 - T1_OBSTACLE], [cx1, cx1 - T1_OBSTACLE], [cx1, cx1 + T1_OBSTACLE], [cx0, cx0 + T1_OBSTACLE]];
    svg.append(s("polygon", { points: band.map(([c, a]) => `${X(c)},${Y(a)}`).join(" "), fill: "var(--bad-soft)", stroke: "var(--bad)", "stroke-dasharray": "3 3", "stroke-opacity": "0.6" }));
    for (const sign of [-1, 1]) {
      svg.append(s("line", { x1: X(cx0), y1: Y(cx0 + sign * T1_HALF_GAP), x2: X(cx1), y2: Y(cx1 + sign * T1_HALF_GAP), stroke: "var(--good)", "stroke-width": 1.5, "stroke-dasharray": "5 3" }));
    }
    for (const a of [-1, -0.5, 0, 0.5, 1]) {
      svg.append(s("text", { x: L - 5, y: Y(a) + 3, "text-anchor": "end", fill: "var(--muted)" }, String(a)));
    }
    for (const c of [-0.4, -0.2, 0, 0.2, 0.4]) {
      svg.append(s("text", { x: X(c), y: H - B + 13, "text-anchor": "middle", fill: "var(--muted)" }, String(c)));
    }
    svg.append(s("text", { x: (L + W - R) / 2, y: H - 4, "text-anchor": "middle", fill: "var(--muted)" }, "obstacle position c"));
    svg.append(s("text", { x: 11, y: (T + H - B) / 2, "text-anchor": "middle", fill: "var(--muted)", transform: `rotate(-90 11 ${(T + H - B) / 2})` }, "sampled a_x"));

    const tip = h("div", { style: "font:12px var(--mono);color:var(--muted);min-height:1.4em;margin-top:4px", text: pts.length ? "hover a point for its values" : "no samples in this result" });
    for (const p of pts) {
      const hit = Math.abs(p[1] - p[0]) < T1_OBSTACLE;
      const dot = s("circle", { cx: X(p[0]), cy: Y(Math.max(ay0, Math.min(ay1, p[1]))), r: 3, fill: hit ? "var(--bad)" : "var(--good)", "fill-opacity": 0.75 });
      const show = () => { tip.textContent = `c=${p[0].toFixed(3)}  a_x=${p[1].toFixed(3)}  a_y=${num(p[2]) == null ? "n/a" : p[2].toFixed(3)}  ${hit ? "collision" : "clear"}`; };
      dot.addEventListener("mouseenter", show);
      dot.addEventListener("mousemove", show);
      dot.append(s("title", {}, `c=${p[0].toFixed(3)}, a_x=${p[1].toFixed(3)}${hit ? " (collision)" : ""}`));
      svg.append(dot);
    }
    wrap.append(svg, tip);
    wrap.append(h("div", { class: "leg", style: "font-size:11px;color:var(--muted)", text: "shaded band = obstacle (|a_x - c| < 0.25), dashed green = the two expert modes c ± 0.5; red points collide" }));
    container.replaceChildren(wrap);
  }

  // ---- traj2d: animated rollouts --------------------------------------------------------
  function traj2d(result, container) {
    const t2 = (result && result.t2) || {};
    const eps = Array.isArray(t2.episodes) ? t2.episodes.filter((e) => e && Array.isArray(e.p) && e.p.length) : [];
    const wrap = h("div", { class: "viewer-traj2d" });
    const stats = h("div", { style: "margin-bottom:6px" });
    stats.append(chip("success", pct(t2.success)), chip("collision", pct(t2.collision)), chip("timeout", pct(t2.timeout)));
    if (Array.isArray(t2.wilson95) && t2.wilson95.length === 2) {
      stats.append(chip("95% CI", `${pct(t2.wilson95[0])} – ${pct(t2.wilson95[1])}`));
    }
    wrap.append(stats);
    if (!eps.length) {
      wrap.append(h("p", { style: "color:var(--muted)", text: "No episodes to animate (0 episodes in this result)." }));
      container.replaceChildren(wrap);
      return;
    }
    const maxLen = Math.max(...eps.map((e) => e.p.length));
    const R = num(t2.obstacle_radius) || 0.3;
    const SIZE = 360;
    const canvas = h("canvas", { width: SIZE, height: SIZE, style: "width:100%;max-width:360px;height:auto;border:1px solid var(--line);border-radius:8px;display:block" });
    const label = h("span", { style: "font:12px var(--mono);color:var(--muted)" });
    const btn = h("button", { type: "button", text: "Pause" });
    const slider = h("input", { type: "range", min: 0, max: maxLen - 1, value: 0, step: 1, style: "flex:1;min-width:120px" });
    const row = h("div", { style: "display:flex;gap:10px;align-items:center;margin-top:8px;max-width:520px" }, btn, slider);
    wrap.append(canvas, row, label);
    container.replaceChildren(wrap);

    const ctx2 = canvas.getContext("2d");
    // world [-1.2, 1.2]^2 -> canvas, y up
    const WORLD = 1.2;
    const px = (x) => ((x + WORLD) / (2 * WORLD)) * SIZE;
    const py = (y) => SIZE - ((y + WORLD) / (2 * WORLD)) * SIZE;
    const sc = (r) => (r / (2 * WORLD)) * SIZE;
    const color = (o) => (o === "success" ? css("--good", "#1a7f4b") : o === "collision" ? css("--bad", "#b3261e") : css("--muted", "#6b7280"));
    let step = 0, playing = true, timer = null;

    function draw() {
      const bg = css("--panel", "#fff");
      ctx2.fillStyle = bg;
      ctx2.fillRect(0, 0, SIZE, SIZE);
      ctx2.strokeStyle = css("--line", "#e2e4e9");
      ctx2.strokeRect(0.5, 0.5, SIZE - 1, SIZE - 1);
      // obstacles (one disc per distinct centre) and goals
      const seen = new Set();
      for (const e of eps) {
        const o = Array.isArray(e.o) ? e.o : [0, 0];
        const k = o.join(",");
        if (seen.has(k)) continue;
        seen.add(k);
        ctx2.beginPath();
        ctx2.arc(px(o[0]), py(o[1]), sc(R), 0, 2 * Math.PI);
        ctx2.fillStyle = css("--bad-soft", "#fbe9e7");
        ctx2.globalAlpha = 0.5;
        ctx2.fill();
        ctx2.globalAlpha = 1;
        ctx2.strokeStyle = css("--bad", "#b3261e");
        ctx2.setLineDash([3, 3]);
        ctx2.stroke();
        ctx2.setLineDash([]);
      }
      for (const e of eps) {
        if (Array.isArray(e.g)) {
          ctx2.fillStyle = css("--muted", "#6b7280");
          ctx2.fillRect(px(e.g[0]) - 3, py(e.g[1]) - 3, 6, 6);
        }
      }
      for (const e of eps) {
        const n = Math.min(step + 1, e.p.length);
        ctx2.beginPath();
        for (let i = 0; i < n; i++) {
          const [x, y] = e.p[i];
          if (i === 0) ctx2.moveTo(px(x), py(y)); else ctx2.lineTo(px(x), py(y));
        }
        ctx2.strokeStyle = color(e.outcome);
        ctx2.globalAlpha = e.outcome === "timeout" ? 0.45 : 0.7;
        ctx2.lineWidth = 1.5;
        ctx2.stroke();
        ctx2.globalAlpha = 1;
        const last = e.p[n - 1];
        ctx2.beginPath();
        ctx2.arc(px(last[0]), py(last[1]), 2.5, 0, 2 * Math.PI);
        ctx2.fillStyle = color(e.outcome);
        ctx2.fill();
      }
      const cnt = { success: 0, collision: 0, timeout: 0 };
      for (const e of eps) if (cnt[e.outcome] != null) cnt[e.outcome]++;
      label.textContent = `${eps.length} episodes shown (${cnt.success} success, ${cnt.collision} collision, ${cnt.timeout} timeout) · step ${step}/${maxLen - 1}`;
      slider.value = step;
    }
    function tick() {
      if (!playing) return;
      if (step >= maxLen - 1) { playing = false; btn.textContent = "Replay"; return; }
      step += 1;
      draw();
    }
    // stop the timer when the viewer leaves the DOM
    timer = setInterval(() => { if (!canvas.isConnected) { clearInterval(timer); return; } tick(); }, 40);
    btn.addEventListener("click", () => {
      if (!playing && step >= maxLen - 1) step = 0;
      playing = !playing;
      btn.textContent = playing ? "Pause" : "Play";
      draw();
    });
    slider.addEventListener("input", () => {
      step = Number(slider.value);
      playing = false;
      btn.textContent = "Play";
      draw();
    });
    draw();
  }

  // ---- t1: composite for the task-1 spec (scatter + rollouts side by side) --------------
  function t1(result, container, ctx) {
    const a = h("div", { class: "chart", style: "min-width:0" }, h("h3", { text: "T1: sampled actions" }), h("div"));
    const b = h("div", { class: "chart", style: "min-width:0" }, h("h3", { text: "T2: rollouts" }), h("div"));
    const grid = h("div", { style: "display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px" }, a, b);
    container.replaceChildren(grid);
    t1scatter(result, a.lastChild, ctx);
    traj2d(result, b.lastChild, ctx);
  }

  // ---- flowtransport: noise -> actions, animated over the K sampling steps ---------------
  function flowtransport(result, container) {
    const tr = (result && result.trace) || {};
    const xs = Array.isArray(tr.xs) ? tr.xs : [];
    const cs = Array.isArray(tr.c) ? tr.c : [];
    const wrap = h("div", { class: "viewer-flowtransport" });
    if (!xs.length || !cs.length) {
      wrap.append(h("p", { style: "color:var(--muted)", text: "No sampling trace in this result." }));
      container.replaceChildren(wrap);
      return;
    }
    const K = xs.length - 1;
    const W = 420, H = 300, L = 40, R = 10, T = 10, B = 32;
    const cx0 = -0.4, cx1 = 0.4, ay0 = -2.5, ay1 = 2.5;
    const X = (c) => L + ((c - cx0) / (cx1 - cx0)) * (W - L - R);
    const Y = (a) => T + (1 - (Math.max(ay0, Math.min(ay1, a)) - ay0) / (ay1 - ay0)) * (H - T - B);
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, style: "width:100%;max-width:520px;height:auto;font:10px var(--mono)", role: "img", "aria-label": "Sampling trace: noise to actions" });
    svg.append(s("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "none", stroke: "var(--line)" }));
    const band = [[cx0, cx0 - T1_OBSTACLE], [cx1, cx1 - T1_OBSTACLE], [cx1, cx1 + T1_OBSTACLE], [cx0, cx0 + T1_OBSTACLE]];
    svg.append(s("polygon", { points: band.map(([c, a]) => `${X(c)},${Y(a)}`).join(" "), fill: "var(--bad-soft)", stroke: "var(--bad)", "stroke-dasharray": "3 3", "stroke-opacity": "0.6" }));
    for (const sign of [-1, 1]) {
      svg.append(s("line", { x1: X(cx0), y1: Y(cx0 + sign * T1_HALF_GAP), x2: X(cx1), y2: Y(cx1 + sign * T1_HALF_GAP), stroke: "var(--good)", "stroke-width": 1.5, "stroke-dasharray": "5 3" }));
    }
    for (const a of [-2, -1, 0, 1, 2]) svg.append(s("text", { x: L - 5, y: Y(a) + 3, "text-anchor": "end", fill: "var(--muted)" }, String(a)));
    for (const c of [-0.4, -0.2, 0, 0.2, 0.4]) svg.append(s("text", { x: X(c), y: H - B + 13, "text-anchor": "middle", fill: "var(--muted)" }, String(c)));
    svg.append(s("text", { x: (L + W - R) / 2, y: H - 4, "text-anchor": "middle", fill: "var(--muted)" }, "obstacle position c"));
    svg.append(s("text", { x: 11, y: (T + H - B) / 2, "text-anchor": "middle", fill: "var(--muted)", transform: `rotate(-90 11 ${(T + H - B) / 2})` }, "sample a_x"));
    const dots = cs.map(() => { const d = s("circle", { r: 3, fill: "var(--accent)", "fill-opacity": 0.75 }); svg.append(d); return d; });
    const label = h("span", { style: "font:12px var(--mono);color:var(--muted)" });
    const btn = h("button", { type: "button", text: "Pause" });
    const slider = h("input", { type: "range", min: 0, max: K, value: 0, step: 1, style: "flex:1;min-width:120px" });
    const row = h("div", { style: "display:flex;gap:10px;align-items:center;margin-top:8px;max-width:520px" }, btn, slider);
    wrap.append(svg, row, label);
    container.replaceChildren(wrap);
    let step = 0, playing = true;
    function draw() {
      xs[step].forEach((p, i) => {
        const hit = Math.abs(p[0] - cs[i]) < T1_OBSTACLE;
        dots[i].setAttribute("cx", X(cs[i]));
        dots[i].setAttribute("cy", Y(p[0]));
        dots[i].setAttribute("fill", step === K ? (hit ? "var(--bad)" : "var(--good)") : "var(--accent)");
      });
      label.textContent = step === 0 ? `step 0/${K}: pure noise` : `step ${step}/${K}${step === K ? ": final actions" : ""}`;
      slider.value = step;
    }
    const timer = setInterval(() => {
      if (!svg.isConnected) { clearInterval(timer); return; }
      if (!playing) return;
      if (step >= K) { playing = false; btn.textContent = "Replay"; return; }
      step += 1; draw();
    }, 450);
    btn.addEventListener("click", () => {
      if (!playing && step >= K) step = 0;
      playing = !playing; btn.textContent = playing ? "Pause" : "Play"; draw();
    });
    slider.addEventListener("input", () => { step = Number(slider.value); playing = false; btn.textContent = "Play"; draw(); });
    draw();
  }

  // ---- stepsweep: T1 collision vs number of sampling steps (eval-only data) --------------
  function stepsweep(result, container) {
    const sw = (result && result.sweep) || {};
    const data = sw.euler || {};
    const ks = Object.keys(data).map(Number).filter((k) => num(data[String(k)]) != null).sort((a, b) => a - b);
    const wrap = h("div", { class: "viewer-stepsweep" });
    if (!ks.length) {
      wrap.append(h("p", { style: "color:var(--muted)", text: "No step sweep in this result." }));
      container.replaceChildren(wrap);
      return;
    }
    const W = 420, H = 240, L = 44, R = 12, T = 12, B = 34;
    const X = (i) => L + (i / Math.max(1, ks.length - 1)) * (W - L - R);
    const Y = (v) => T + (1 - v) * (H - T - B);
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, style: "width:100%;max-width:520px;height:auto;font:10px var(--mono)", role: "img", "aria-label": "T1 collision rate vs sampling steps" });
    svg.append(s("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "none", stroke: "var(--line)" }));
    for (const v of [0, 0.25, 0.5, 0.75, 1]) {
      svg.append(s("text", { x: L - 5, y: Y(v) + 3, "text-anchor": "end", fill: "var(--muted)" }, pct(v)));
      svg.append(s("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), stroke: "var(--line)", "stroke-opacity": 0.5 }));
    }
    const cur = num(result.sample_steps);
    ks.forEach((k, i) => svg.append(s("text", { x: X(i), y: H - B + 14, "text-anchor": "middle", fill: k === cur ? "var(--ink)" : "var(--muted)", "font-weight": k === cur ? "700" : "400" }, String(k))));
    svg.append(s("text", { x: (L + W - R) / 2, y: H - 4, "text-anchor": "middle", fill: "var(--muted)" }, `sampling steps K (${sw.kind || "euler"})`));
    svg.append(s("polyline", { points: ks.map((k, i) => `${X(i)},${Y(data[String(k)])}`).join(" "), fill: "none", stroke: "var(--accent)", "stroke-width": 2 }));
    ks.forEach((k, i) => {
      const v = data[String(k)];
      const d = s("circle", { cx: X(i), cy: Y(v), r: k === cur ? 5 : 3.5, fill: "var(--accent)" });
      d.append(s("title", {}, `K=${k}: T1 collision ${pct(v)}`));
      svg.append(d);
      svg.append(s("text", { x: X(i), y: Y(v) - 8, "text-anchor": "middle", fill: "var(--ink)" }, pct(v)));
    });
    wrap.append(svg, h("div", { class: "leg", style: "font-size:11px;color:var(--muted)", text: "T1 collision rate of the same trained net, sampled with K steps (bold = the K used for the scatter and rollouts)." }));
    container.replaceChildren(wrap);
  }

  // ---- flow: composite for the task-2 spec -----------------------------------------------
  function flow(result, container, ctx) {
    const card = (title) => h("div", { class: "chart", style: "min-width:0" }, h("h3", { text: title }), h("div"));
    const head = result && result.head ? `${result.head}, K=${result.sample_steps}` : "";
    const a = card(`T1: noise to actions (${head})`), b = card("T1: sampled actions"),
      c = card("T1 collision vs sampling steps"), d = card("T2: rollouts");
    const grid = h("div", { style: "display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px" }, a, b, c, d);
    container.replaceChildren(grid);
    flowtransport(result, a.lastChild, ctx);
    t1scatter(result, b.lastChild, ctx);
    stepsweep(result, c.lastChild, ctx);
    traj2d(result, d.lastChild, ctx);
  }

  window.LabViewers = { register: (name, fn) => { registry[name] = fn; }, render, jsonFallback };
  window.LabViewers.register("t1scatter", t1scatter);
  window.LabViewers.register("traj2d", traj2d);
  window.LabViewers.register("t1", t1);
  window.LabViewers.register("flowtransport", flowtransport);
  window.LabViewers.register("stepsweep", stepsweep);
  window.LabViewers.register("flow", flow);
})();
